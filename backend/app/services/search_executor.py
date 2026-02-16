from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any

from app.models.events import SSEEvent
from app.services import streaming
from app.tools import search_provider, web_utils


@dataclass(slots=True)
class StepSearchResult:
    step_index: int
    queries: list[str] = field(default_factory=list)
    results: list[dict[str, Any]] = field(default_factory=list)
    provider: str | None = None
    fallback_from: str | None = None
    fallback_reason: str | None = None


def build_step_queries(step: str, *, max_queries: int) -> list[str]:
    """Build deterministic, search-friendly query variants for one plan step."""
    cleaned = " ".join(step.split()).strip()
    if not cleaned:
        return []

    queries: list[str] = [cleaned]
    lowered = cleaned.lower()
    has_site_operator = "site:" in lowered

    quoted_entities = [
        " ".join(match.split()).strip()
        for match in re.findall(r'"([^"]{2,120})"', cleaned)
    ]
    titled_entities = [
        " ".join(match.split()).strip()
        for match in re.findall(
            r"\b[A-Z][A-Za-z0-9&'().-]*(?:\s+[A-Z][A-Za-z0-9&'().-]*){1,4}",
            cleaned,
        )
    ]
    entities = quoted_entities + titled_entities

    if not has_site_operator and entities:
        queries.append(f'"{entities[0]}" site:wikipedia.org')

    source_mentions = [
        " ".join(match.split()).strip()
        for match in re.findall(r"according to ([^,.;]+)", cleaned, flags=re.IGNORECASE)
    ]
    for source in source_mentions[:2]:
        queries.append(f'"{source}" {cleaned}')
        source_lower = source.lower()
        if "world population review" in source_lower:
            queries.append(f"{cleaned} site:worldpopulationreview.com")
        if "vision of humanity" in source_lower:
            queries.append(f"{cleaned} site:visionofhumanity.org")
        if "organized crime index" in source_lower or "organised crime index" in source_lower:
            queries.append(f"{cleaned} site:ocindex.net")
        if "migration observatory" in source_lower or "observatory of migration" in source_lower:
            queries.append(f"{cleaned} site:ox.ac.uk")

    years = list(dict.fromkeys(re.findall(r"\b(?:19|20)\d{2}\b", cleaned)))
    if years:
        queries.append(f"{cleaned} {' '.join(years[:2])} timeline")

    if "only provide" in lowered or "name the" in lowered or "list" in lowered:
        queries.append(f"{cleaned} official source")

    if len(cleaned.split()) > 12:
        stopwords = {
            "the",
            "that",
            "from",
            "with",
            "which",
            "were",
            "what",
            "when",
            "where",
            "according",
            "amongst",
            "their",
            "these",
            "those",
            "into",
            "this",
            "only",
            "provide",
        }
        terms = [
            token
            for token in re.findall(r"[A-Za-z0-9']+", cleaned)
            if len(token) > 3 and token.lower() not in stopwords
        ]
        if terms:
            queries.append(" ".join(terms[:10]))

    deduped: list[str] = []
    seen: set[str] = set()
    for query in queries:
        q = " ".join(query.split()).strip()
        if not q:
            continue
        key = q.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(q)
        if len(deduped) >= max(max_queries, 1):
            break
    return deduped


def _dedupe_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_url: dict[str, dict[str, Any]] = {}
    for item in results:
        url = item.get("url")
        if not isinstance(url, str):
            continue
        if not web_utils.is_valid_url(url):
            continue
        prev = by_url.get(url)
        if prev is None or float(item.get("score", 0.0) or 0.0) > float(
            prev.get("score", 0.0) or 0.0
        ):
            by_url[url] = item
    return sorted(
        by_url.values(),
        key=lambda item: float(item.get("score", 0.0) or 0.0),
        reverse=True,
    )


async def run_deterministic_searches(
    plan_steps: list[str],
    *,
    step_offset: int,
    max_parallel: int,
    max_queries_per_step: int,
    max_results_per_query: int,
) -> tuple[list[dict[str, Any]], list[SSEEvent]]:
    """Run deterministic web searches for each step with bounded parallelism."""
    events: list[SSEEvent] = []
    all_results: list[dict[str, Any]] = []
    semaphore = asyncio.Semaphore(max(max_parallel, 1))

    for index, step in enumerate(plan_steps):
        events.append(streaming.agent_started("search", step=step_offset + index, query=step))

    async def run_step(step: str, index: int) -> StepSearchResult:
        step_result = StepSearchResult(step_index=step_offset + index)
        queries = build_step_queries(step, max_queries=max_queries_per_step)
        step_result.queries = queries
        if not queries:
            return step_result

        merged: list[dict[str, Any]] = []
        providers_seen: set[str] = set()
        fallback_from: str | None = None
        fallback_reason: str | None = None

        for query in queries:
            async with semaphore:
                response = await search_provider.search(
                    query=query,
                    search_depth="advanced",
                    max_results=max_results_per_query,
                )
            providers_seen.add(response.provider)
            if response.fallback_from:
                fallback_from = response.fallback_from
            if response.fallback_reason:
                fallback_reason = response.fallback_reason
            merged.extend(search_provider.results_to_dicts(response.results))

        step_result.results = _dedupe_results(merged)
        step_result.provider = (
            next(iter(providers_seen))
            if len(providers_seen) == 1
            else ",".join(sorted(providers_seen))
            if providers_seen
            else None
        )
        step_result.fallback_from = fallback_from
        step_result.fallback_reason = fallback_reason
        return step_result

    raw_results = await asyncio.gather(
        *(run_step(step, index) for index, step in enumerate(plan_steps)),
        return_exceptions=True,
    )

    for index, item in enumerate(raw_results):
        step_index = step_offset + index
        if isinstance(item, Exception):
            events.append(streaming.error(str(item), agent="search"))
            events.append(streaming.agent_completed("search", step=step_index, success=False))
            continue

        all_results.extend(item.results)
        events.append(
            streaming.search_result(
                step_index,
                item.results,
                provider=item.provider,
                fallback_from=item.fallback_from,
                fallback_reason=item.fallback_reason,
            )
        )
        events.append(
            streaming.agent_completed(
                "search",
                step=step_index,
                success=True,
                queries_run=len(item.queries),
                results_count=len(item.results),
                provider=item.provider,
            )
        )

    return all_results, events
