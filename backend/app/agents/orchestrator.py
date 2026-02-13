from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any, AsyncGenerator
from uuid import UUID

from app.agents.analyzer_agent import AnalyzerAgent
from app.agents.search_agent import SearchAgent
from app.agents.scraper_agent import ScraperAgent
from app.llm_client import client as llm_client, get_model
from app.models.events import SSEEvent
from app.services import streaming
from app.services import supabase as db
from app.tools import web_utils


@dataclass
class ResearchNotes:
    """Compact working memory captured between research phases."""

    highlights: list[str] = field(default_factory=list)
    resolved_points: list[str] = field(default_factory=list)
    unresolved_points: list[str] = field(default_factory=list)
    follow_up_queries: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "highlights": self.highlights,
            "resolved_points": self.resolved_points,
            "unresolved_points": self.unresolved_points,
            "follow_up_queries": self.follow_up_queries,
        }


@dataclass
class SynthesisReview:
    """Internal quality review for deciding whether to run more research."""

    needs_more_research: bool = False
    reason: str = ""
    missing_points: list[str] = field(default_factory=list)
    follow_up_queries: list[str] = field(default_factory=list)


class ResearchOrchestrator:
    """Orchestrates the full research pipeline.

    Flow:
      1. Generate research plan (sub-queries/angles) via LLM
      2. Fan out: run search agents in parallel
      3. Collect results, identify top URLs
      4. Fan out: scrape top URLs in parallel
      5. Feed everything to analyzer agent for synthesis
      6. Yield final report

    All steps yield SSE events for real-time frontend updates.
    """

    def __init__(self, model: str | None = None, session_id: str | None = None):
        self.model = model or get_model()
        self.session_id = session_id
        self.client = None
        self.max_follow_up_rounds = 2
        self.max_follow_up_queries_per_round = 3
        self.max_post_synthesis_review_rounds = 1
        self.max_post_synthesis_follow_up_queries = 2

    async def _log_call(self, caller: str, response: Any, elapsed_ms: int) -> None:
        """Log an LLM call to the database and file logs."""
        from app.services import logger as log_service

        try:
            usage = response.usage if hasattr(response, "usage") else None
            input_tokens = getattr(usage, "input_tokens", 0) if usage else 0
            output_tokens = getattr(usage, "output_tokens", 0) if usage else 0

            await db.log_llm_call(
                model=self.model,
                caller=caller,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                duration_ms=elapsed_ms,
                session_id=UUID(self.session_id) if self.session_id else None,
            )
            # Also log to file for debugging
            log_service.log_llm_call(
                model=self.model,
                caller=caller,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                duration_ms=elapsed_ms,
            )
        except Exception as e:
            log_service.log_event(
                event_type="logging_error",
                message=f"Failed to log LLM call in {caller}",
                error=str(e),
                model=self.model,
            )

    async def _generate_plan(self, query: str) -> list[str]:
        """Use the configured model to break the research query into sub-queries."""
        def normalize(raw_steps: Any) -> list[str]:
            if not isinstance(raw_steps, list):
                return []
            cleaned: list[str] = []
            seen: set[str] = set()
            for item in raw_steps:
                if not isinstance(item, str):
                    continue
                step = " ".join(item.split()).strip()
                if not step:
                    continue
                key = step.lower()
                if key in seen:
                    continue
                seen.add(key)
                cleaned.append(step)
            return cleaned

        def is_chain_query(text: str) -> bool:
            q = text.lower()
            markers = [
                "who",
                "that",
                "which",
                "from 20",
                "from 19",
                "list",
                "all",
                "between",
                "relationship",
            ]
            return sum(1 for m in markers if m in q) >= 3

        def extract_key_phrases(text: str) -> list[str]:
            phrases: list[str] = []
            seen: set[str] = set()

            quoted = re.findall(r'"([^"]{2,120})"', text)
            titled = re.findall(
                r"\b[A-Z][A-Za-z0-9&'().-]*(?:\s+[A-Z][A-Za-z0-9&'().-]*){1,4}",
                text,
            )

            for phrase in quoted + titled:
                cleaned = " ".join(phrase.split()).strip(".,:;!?")
                if len(cleaned) < 3:
                    continue
                key = cleaned.lower()
                if key in seen:
                    continue
                seen.add(key)
                phrases.append(cleaned)

            return phrases[:4]

        def augment_plan_steps(base_steps: list[str], original_query: str) -> list[str]:
            steps: list[str] = []
            to_add: list[str] = list(base_steps)
            to_add.append(original_query)

            if is_chain_query(original_query):
                # Generic chain-task augmentation for multi-hop disambiguation.
                to_add.append(f"{original_query} site:wikipedia.org")

                key_phrases = extract_key_phrases(original_query)
                for phrase in key_phrases[:3]:
                    to_add.append(f'"{phrase}" site:wikipedia.org')

                if len(key_phrases) >= 2:
                    to_add.append(
                        f'"{key_phrases[0]}" "{key_phrases[1]}" relationship timeline'
                    )

                years = re.findall(r"\b(?:19|20)\d{2}\b", original_query)
                if years:
                    unique_years = " ".join(dict.fromkeys(years))
                    to_add.append(f"{original_query} {unique_years} timeline")

                to_add.append(f"{original_query} corroborating sources")

            seen = {s.lower() for s in steps}
            for step in to_add:
                key = step.lower()
                if key not in seen:
                    seen.add(key)
                    steps.append(step)

            return steps[:8]

        active_client = self.client or llm_client()
        t0 = time.monotonic()
        response = await active_client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=(
                f"You are a research planning expert. Today's date is {date.today().isoformat()}. "
                "Given a research query, break it down into "
                "3-5 specific sub-queries that together will provide comprehensive coverage of the "
                "topic. Each sub-query should explore a different angle or aspect. "
                "For chain questions, include explicit entity-resolution steps and at least one "
                "canonical-source step (e.g. site:wikipedia.org) before downstream steps. "
                "When the query asks about 'current' or 'latest' data, make sure sub-queries "
                f"include the current year ({date.today().year}) to find the most recent information.\n\n"
                "Respond with ONLY a JSON array of strings, no other text. Example:\n"
                '["sub-query 1", "sub-query 2", "sub-query 3"]'
            ),
            messages=[{"role": "user", "content": query}],
        )
        await self._log_call("orchestrator.plan", response, int((time.monotonic() - t0) * 1000))

        blocks = getattr(response, "content", None) or []
        text_parts: list[str] = []
        for block in blocks:
            btype = getattr(block, "type", None)
            btext = getattr(block, "text", None)
            is_text_like_type = btype in (None, "text") or not isinstance(btype, str)
            if is_text_like_type and isinstance(btext, str) and btext.strip():
                text_parts.append(btext)

        if not text_parts:
            return augment_plan_steps([], query)

        text = "\n".join(text_parts).strip()
        # Extract JSON array from response
        try:
            # Handle case where model wraps in markdown code block
            if text.startswith("```"):
                parts = text.split("```")
                if len(parts) >= 2:
                    text = parts[1]
                else:
                    raise json.JSONDecodeError("invalid fenced block", text, 0)
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()
            start = text.find("[")
            end = text.rfind("]")
            if start >= 0 and end > start:
                text = text[start : end + 1]

            parsed = normalize(json.loads(text))
            if is_chain_query(query):
                return augment_plan_steps(parsed, query)
            if len(parsed) < 3:
                return augment_plan_steps(parsed, query)
            return parsed[:8]
        except (json.JSONDecodeError, IndexError):
            # Fallback: build deterministic chain-aware steps.
            return augment_plan_steps([], query)

    async def _run_parallel_searches(
        self, plan_steps: list[str], *, step_offset: int = 0
    ) -> tuple[list[dict[str, Any]], list[SSEEvent]]:
        """Run search agents in parallel for all plan steps."""
        all_events: list[SSEEvent] = []
        all_results: list[dict[str, Any]] = []

        agents = [
            SearchAgent(
                model=self.model,
                step_index=step_offset + i,
                session_id=self.session_id,
            )
            for i in range(len(plan_steps))
        ]

        # Start events
        for i, step in enumerate(plan_steps):
            all_events.append(
                streaming.agent_started("search", step=step_offset + i, query=step)
            )

        # Run all searches in parallel
        async def run_search(agent: SearchAgent, query: str) -> list[SSEEvent]:
            events: list[SSEEvent] = []
            async for event in agent.run(
                query,
                context=f"This is step {agent.step_index} of the research plan.",
            ):
                events.append(event)
            return events

        tasks = [run_search(agent, step) for agent, step in zip(agents, plan_steps)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, (agent, result) in enumerate(zip(agents, results)):
            if isinstance(result, Exception):
                all_events.append(streaming.error(str(result), agent="search"))
            else:
                all_events.extend(result)
                all_results.extend(agent.all_results)
            all_events.append(streaming.agent_completed("search", step=step_offset + i))

        return all_results, all_events

    async def _select_top_urls(self, search_results: list[dict[str, Any]], max_urls: int = 8) -> list[str]:
        """Select the top URLs to scrape based on relevance scores."""
        # Deduplicate by URL, keep highest score
        seen: dict[str, float] = {}
        for r in search_results:
            url = r.get("url", "")
            score = r.get("score", 0.0)
            if url and web_utils.is_valid_url(url):
                if url not in seen or score > seen[url]:
                    seen[url] = score

        # Sort by score descending, take top N
        sorted_urls = sorted(seen.items(), key=lambda x: x[1], reverse=True)
        selected = [url for url, _ in sorted_urls[:max_urls]]

        # Ensure canonical references (Wikipedia) are represented in scrape set.
        wiki_candidates = [url for url, _ in sorted_urls if "wikipedia.org" in web_utils.extract_domain(url)]
        for wiki_url in wiki_candidates[:2]:
            if wiki_url in selected:
                continue
            if len(selected) < max_urls:
                selected.append(wiki_url)
                continue

            # Replace the lowest-ranked non-wiki URL so canonical entity pages are scraped.
            for idx in range(len(selected) - 1, -1, -1):
                if "wikipedia.org" not in web_utils.extract_domain(selected[idx]):
                    selected[idx] = wiki_url
                    break

        return selected

    def _extract_canonical_full_names(
        self, search_results: list[dict[str, Any]], scraped_content: dict[str, str]
    ) -> list[str]:
        """Extract likely full personal names from canonical-source text."""
        full_names: list[str] = []
        seen: set[str] = set()

        patterns = [
            re.compile(r"\b([A-Z][a-z]+(?: [A-Z][a-z]+){2,4})\s*\(born\b"),
            re.compile(r"\b([A-Z][a-z]+(?: [A-Z][a-z]+){2,4})\s+is an\b"),
        ]

        def collect_from_text(text: str) -> None:
            for pattern in patterns:
                for match in pattern.findall(text):
                    candidate = " ".join(match.split()).strip()
                    if len(candidate.split()) < 3:
                        continue
                    key = candidate.lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    full_names.append(candidate)

        for result in search_results:
            url = result.get("url", "")
            if "wikipedia.org" not in web_utils.extract_domain(url):
                continue
            title = result.get("title", "")
            snippet = result.get("content", "")
            collect_from_text(f"{title} {snippet}")

        for url, content in scraped_content.items():
            if "wikipedia.org" not in web_utils.extract_domain(url):
                continue
            collect_from_text(content[:12000])

        return full_names

    def _promote_full_name_mentions(self, report: str, full_names: list[str]) -> str:
        """Expand first short-name mention to include canonical full name when available."""
        updated = report
        for full_name in full_names:
            parts = full_name.split()
            if len(parts) < 3:
                continue
            short_name = f"{parts[0]} {parts[-1]}"
            if full_name.lower() in updated.lower():
                continue
            pattern = re.compile(rf"\b{re.escape(short_name)}\b", flags=re.IGNORECASE)
            if pattern.search(updated):
                updated = pattern.sub(f"{full_name} ({short_name})", updated, count=1)
        return updated

    async def _run_parallel_scrapes(
        self, urls: list[str]
    ) -> tuple[dict[str, str], list[SSEEvent]]:
        """Scrape multiple URLs in parallel using scraper agents."""
        all_events: list[SSEEvent] = []
        all_content: dict[str, str] = {}

        agent = ScraperAgent(model=self.model, session_id=self.session_id)

        for url in urls:
            all_events.append(streaming.agent_started("scraper", url=url))

        # Run scraper agent with all URLs
        prompt = "Scrape the following URLs to extract their content:\n" + "\n".join(
            f"- {url}" for url in urls
        )
        async for event in agent.run(prompt):
            all_events.append(event)

        all_content = agent.scraped_content
        # Emit completion per URL to keep start/completion lifecycle consistent.
        for url in urls:
            content = all_content.get(url, "")
            all_events.append(
                streaming.agent_completed(
                    "scraper",
                    url=url,
                    success=bool(content),
                    content_length=len(content),
                )
            )

        return all_content, all_events

    @staticmethod
    def _normalize_text_list(
        raw_values: Any, *, max_items: int, min_len: int = 1
    ) -> list[str]:
        if not isinstance(raw_values, list):
            return []
        cleaned: list[str] = []
        seen: set[str] = set()
        for item in raw_values:
            if not isinstance(item, str):
                continue
            value = " ".join(item.split()).strip()
            if len(value) < min_len:
                continue
            key = value.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(value)
            if len(cleaned) >= max_items:
                break
        return cleaned

    @staticmethod
    def _extract_response_text(response: Any) -> str:
        blocks = getattr(response, "content", None) or []
        text_parts: list[str] = []
        for block in blocks:
            btype = getattr(block, "type", None)
            btext = getattr(block, "text", None)
            is_text_like_type = btype in (None, "text") or not isinstance(btype, str)
            if is_text_like_type and isinstance(btext, str) and btext.strip():
                text_parts.append(btext)
        return "\n".join(text_parts).strip()

    @staticmethod
    def _extract_json_object(raw_text: str) -> dict[str, Any]:
        text = raw_text.strip()
        if text.startswith("```"):
            parts = text.split("```")
            if len(parts) >= 2:
                text = parts[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise json.JSONDecodeError("object not found", text, 0)
        parsed = json.loads(text[start : end + 1])
        if not isinstance(parsed, dict):
            raise json.JSONDecodeError("not an object", text, 0)
        return parsed

    @staticmethod
    def _merge_search_results(
        current: list[dict[str, Any]], additional: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        merged = list(current)
        seen_urls: set[str] = {
            r.get("url", "").strip().lower()
            for r in current
            if isinstance(r.get("url"), str) and r.get("url", "").strip()
        }
        for result in additional:
            url = result.get("url", "")
            key = url.strip().lower() if isinstance(url, str) else ""
            if key and key in seen_urls:
                continue
            if key:
                seen_urls.add(key)
            merged.append(result)
        return merged

    def _fallback_follow_up_queries(
        self, query: str, unresolved_points: list[str]
    ) -> list[str]:
        queries: list[str] = []
        for unresolved in unresolved_points:
            clipped = " ".join(unresolved.split())[:140].strip()
            if not clipped:
                continue
            queries.append(f"{query} {clipped}")
            if len(queries) >= self.max_follow_up_queries_per_round:
                break
        return self._normalize_text_list(queries, max_items=self.max_follow_up_queries_per_round, min_len=6)

    def _fallback_notes(
        self, query: str, search_results: list[dict[str, Any]]
    ) -> ResearchNotes:
        highlights: list[str] = []
        for result in search_results[:8]:
            title = result.get("title", "")
            snippet = result.get("content", "")
            if isinstance(title, str) and isinstance(snippet, str) and title and snippet:
                highlights.append(f"{title}: {snippet[:180]}")
            elif isinstance(title, str) and title:
                highlights.append(title)
        highlights = self._normalize_text_list(highlights, max_items=8, min_len=3)
        unresolved: list[str] = []
        if len(search_results) < 4:
            unresolved.append("Evidence coverage appears thin; additional corroboration may be needed.")
        return ResearchNotes(
            highlights=highlights,
            unresolved_points=unresolved,
            follow_up_queries=self._fallback_follow_up_queries(query, unresolved),
        )

    async def _capture_research_notes(
        self,
        query: str,
        search_results: list[dict[str, Any]],
        scraped_content: dict[str, str],
        *,
        previous_notes: ResearchNotes | None = None,
    ) -> ResearchNotes:
        """Summarize in-flight findings and unresolved points for follow-up search rounds."""
        active_client = self.client or llm_client()

        search_evidence: list[str] = []
        for i, result in enumerate(search_results[:14]):
            title = result.get("title", "Untitled")
            url = result.get("url", "")
            snippet = str(result.get("content", ""))[:320]
            search_evidence.append(
                f"{i + 1}. title={title}\nurl={url}\nsnippet={snippet}"
            )

        scrape_evidence: list[str] = []
        for i, (url, content) in enumerate(list(scraped_content.items())[:5]):
            excerpt = " ".join(content.split())[:4000]
            scrape_evidence.append(f"{i + 1}. url={url}\nexcerpt={excerpt}")

        previous_payload = previous_notes.to_dict() if previous_notes else {}

        t0 = time.monotonic()
        response = await active_client.messages.create(
            model=self.model,
            max_tokens=1400,
            system=(
                f"You create compact research working notes. Today's date is {date.today().isoformat()}. "
                "Given a query and intermediate evidence, return ONLY JSON with keys: "
                "highlights, resolved_points, unresolved_points, follow_up_queries. "
                "Each value must be an array of short strings. "
                "Use follow_up_queries for concrete next searches that can close unresolved points now. "
                "If the query is already fully answered with sufficient evidence, keep unresolved_points "
                "and follow_up_queries empty. Avoid generic advice and avoid markdown."
            ),
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"QUERY:\n{query}\n\n"
                        f"PREVIOUS_NOTES_JSON:\n{json.dumps(previous_payload)}\n\n"
                        f"SEARCH_EVIDENCE:\n{chr(10).join(search_evidence) or 'none'}\n\n"
                        f"SCRAPE_EVIDENCE:\n{chr(10).join(scrape_evidence) or 'none'}\n\n"
                        "Return JSON only."
                    ),
                }
            ],
        )
        await self._log_call(
            "orchestrator.notes",
            response,
            int((time.monotonic() - t0) * 1000),
        )

        text = self._extract_response_text(response)
        if not text:
            return self._fallback_notes(query, search_results)

        try:
            payload = self._extract_json_object(text)
        except json.JSONDecodeError:
            return self._fallback_notes(query, search_results)

        notes = ResearchNotes(
            highlights=self._normalize_text_list(
                payload.get("highlights"), max_items=10, min_len=3
            ),
            resolved_points=self._normalize_text_list(
                payload.get("resolved_points"), max_items=8, min_len=3
            ),
            unresolved_points=self._normalize_text_list(
                payload.get("unresolved_points"), max_items=6, min_len=3
            ),
            follow_up_queries=self._normalize_text_list(
                payload.get("follow_up_queries"),
                max_items=self.max_follow_up_queries_per_round,
                min_len=6,
            ),
        )

        if notes.unresolved_points and not notes.follow_up_queries:
            notes.follow_up_queries = self._fallback_follow_up_queries(
                query, notes.unresolved_points
            )

        return notes

    async def _persist_research_notes(
        self,
        notes: ResearchNotes,
        *,
        phase: str,
        iteration: int,
    ) -> None:
        """Persist notes snapshots for auditability and history hydration."""
        if not self.session_id:
            return
        from app.services import logger as log_service

        try:
            session_uuid = UUID(self.session_id)
        except ValueError:
            return

        try:
            row = await db.create_research_step(
                session_uuid,
                "notes",
                {
                    "phase": phase,
                    "iteration": iteration,
                    **notes.to_dict(),
                },
            )
            await db.update_research_step(UUID(row["id"]), "completed")
        except Exception as exc:
            log_service.log_event(
                event_type="notes_persist_error",
                message="Failed to persist research notes snapshot",
                error=str(exc),
                session_id=self.session_id,
            )

    async def _persist_synthesis_review(
        self,
        review: SynthesisReview,
        *,
        iteration: int,
    ) -> None:
        """Persist synthesis sufficiency checks for later inspection."""
        if not self.session_id:
            return
        from app.services import logger as log_service

        try:
            session_uuid = UUID(self.session_id)
        except ValueError:
            return

        try:
            row = await db.create_research_step(
                session_uuid,
                "synthesis_review",
                {
                    "iteration": iteration,
                    "needs_more_research": review.needs_more_research,
                    "reason": review.reason,
                    "missing_points": review.missing_points,
                    "follow_up_queries": review.follow_up_queries,
                },
            )
            await db.update_research_step(UUID(row["id"]), "completed")
        except Exception as exc:
            log_service.log_event(
                event_type="review_persist_error",
                message="Failed to persist synthesis review",
                error=str(exc),
                session_id=self.session_id,
            )

    @staticmethod
    def _notes_stabilized(previous: ResearchNotes, current: ResearchNotes) -> bool:
        prev_unresolved = {item.lower() for item in previous.unresolved_points}
        curr_unresolved = {item.lower() for item in current.unresolved_points}
        prev_follow = {item.lower() for item in previous.follow_up_queries}
        curr_follow = {item.lower() for item in current.follow_up_queries}
        return curr_unresolved == prev_unresolved and curr_follow.issubset(prev_follow)

    @staticmethod
    def _strip_deferred_sections(report: str) -> str:
        """Remove trailing sections that defer work to future investigation."""
        if not report:
            return report

        targets = (
            "areas for further investigation",
            "further investigation",
            "future work",
            "open questions",
        )
        lines = report.splitlines()
        cleaned: list[str] = []
        skipping = False

        for line in lines:
            stripped = line.strip()
            normalized = re.sub(r"^[#\s>*-]+", "", stripped).strip(" :").lower()
            heading_match = bool(re.match(r"^#{1,6}\s+", stripped))
            standalone_heading_like = bool(stripped) and len(stripped) <= 80

            is_deferred_heading = any(
                normalized == target or normalized.startswith(f"{target} ")
                for target in targets
            )

            if is_deferred_heading and (heading_match or standalone_heading_like):
                skipping = True
                continue

            if skipping and heading_match:
                skipping = False

            if not skipping:
                cleaned.append(line)

        return "\n".join(cleaned).strip()

    def _build_synthesis_context(
        self,
        query: str,
        search_results: list[dict[str, Any]],
        scraped_content: dict[str, str],
        notes: ResearchNotes | None,
    ) -> tuple[str, list[str]]:
        today = date.today()
        context_parts = [
            f"# Current Date\n{today.isoformat()} ({today.year})\n",
            f"# Original Research Query\n{query}\n",
        ]

        context_parts.append("# Search Results Summary")
        for i, result in enumerate(search_results[:20]):
            context_parts.append(
                f"\n## Source {i+1}: [{result.get('title', 'Untitled')}]({result.get('url', '')})\n"
                f"{result.get('content', '')[:500]}"
            )

        if scraped_content:
            context_parts.append("\n# Detailed Content from Top Sources")
            for url, content in list(scraped_content.items())[:8]:
                domain = web_utils.extract_domain(url)
                context_parts.append(
                    f"\n## Content from {domain}\nURL: {url}\n{content[:12000]}"
                )

        full_names = self._extract_canonical_full_names(search_results, scraped_content)
        if full_names:
            context_parts.append("\n# Canonical Name Hints")
            for name in full_names[:5]:
                context_parts.append(f"- {name}")

        if notes:
            context_parts.append("\n# Working Research Notes")
            if notes.highlights:
                context_parts.append("## Highlights")
                context_parts.extend(f"- {item}" for item in notes.highlights)
            if notes.resolved_points:
                context_parts.append("\n## Resolved Points")
                context_parts.extend(f"- {item}" for item in notes.resolved_points)
            if notes.unresolved_points:
                context_parts.append("\n## Unresolved Points")
                context_parts.extend(f"- {item}" for item in notes.unresolved_points)
            if notes.follow_up_queries:
                context_parts.append("\n## Follow-up Queries Already Planned/Run")
                context_parts.extend(f"- {item}" for item in notes.follow_up_queries)

        return "\n".join(context_parts), full_names

    @staticmethod
    def _build_synthesis_instruction(query: str) -> str:
        return (
            f"Based on all the research data provided, write a comprehensive "
            f"research report on: {query}\n\n"
            "Resolve the question as completely as possible in this report. "
            "Do not include sections titled 'Areas for Further Investigation', "
            "'Further Investigation', 'Open Questions', or 'Future Work'. "
            "If evidence has limits, state confidence and what was verified instead of deferring work. "
            "If sources provide a person's full name, include the full name explicitly. "
            "For songs, include release year when supported by sources. "
            "Include citations to sources using [Title](URL) format."
        )

    def _build_synthesis_messages(self, query: str, full_context: str) -> list[dict[str, Any]]:
        return [
            {"role": "user", "content": f"<context>\n{full_context}\n</context>"},
            {
                "role": "assistant",
                "content": "I've reviewed all the research materials. Let me now provide my analysis.",
            },
            {
                "role": "user",
                "content": self._build_synthesis_instruction(query),
            },
        ]

    async def _generate_draft_report(
        self,
        query: str,
        search_results: list[dict[str, Any]],
        scraped_content: dict[str, str],
        notes: ResearchNotes | None,
    ) -> str:
        """Create an internal non-stream draft report for sufficiency review."""
        active_client = self.client or llm_client()
        full_context, full_names = self._build_synthesis_context(
            query,
            search_results,
            scraped_content,
            notes,
        )

        t0 = time.monotonic()
        response = await active_client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=AnalyzerAgent.get_system_prompt(),
            messages=self._build_synthesis_messages(query, full_context),
        )
        await self._log_call(
            "orchestrator.synthesis_draft",
            response,
            int((time.monotonic() - t0) * 1000),
        )

        draft = self._extract_response_text(response)
        draft = self._promote_full_name_mentions(draft, full_names)
        return self._strip_deferred_sections(draft)

    async def _review_report_sufficiency(
        self,
        query: str,
        draft_report: str,
        notes: ResearchNotes | None,
    ) -> SynthesisReview:
        """Evaluate whether the draft answer is complete and identify targeted follow-up searches."""
        active_client = self.client or llm_client()
        notes_payload = notes.to_dict() if notes else {}

        t0 = time.monotonic()
        response = await active_client.messages.create(
            model=self.model,
            max_tokens=1200,
            system=(
                "You are a strict research QA reviewer. "
                "Given a query and a draft answer, decide if more web research is required "
                "to fully answer the user query. "
                "Respond ONLY JSON with keys: needs_more_research (boolean), reason (string), "
                "missing_points (array of strings), follow_up_queries (array of specific web search queries). "
                "Only request more research for factual coverage gaps, not writing style."
            ),
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"QUERY:\n{query}\n\n"
                        f"WORKING_NOTES_JSON:\n{json.dumps(notes_payload)}\n\n"
                        f"DRAFT_REPORT:\n{draft_report[:7000]}\n\n"
                        "Return JSON only."
                    ),
                }
            ],
        )
        await self._log_call(
            "orchestrator.synthesis_review",
            response,
            int((time.monotonic() - t0) * 1000),
        )

        text = self._extract_response_text(response)
        if text:
            try:
                payload = self._extract_json_object(text)
                follow_up_queries = self._normalize_text_list(
                    payload.get("follow_up_queries"),
                    max_items=self.max_post_synthesis_follow_up_queries,
                    min_len=6,
                )
                missing_points = self._normalize_text_list(
                    payload.get("missing_points"),
                    max_items=6,
                    min_len=3,
                )
                needs_more = bool(payload.get("needs_more_research"))
                reason = str(payload.get("reason", "")).strip()
                if needs_more and not follow_up_queries and notes:
                    follow_up_queries = self._normalize_text_list(
                        notes.follow_up_queries,
                        max_items=self.max_post_synthesis_follow_up_queries,
                        min_len=6,
                    )
                return SynthesisReview(
                    needs_more_research=needs_more and bool(follow_up_queries),
                    reason=reason,
                    missing_points=missing_points,
                    follow_up_queries=follow_up_queries,
                )
            except json.JSONDecodeError:
                pass

        draft_lower = draft_report.lower()
        uncertainty_markers = (
            "unable to determine",
            "not enough evidence",
            "insufficient evidence",
            "unclear from available sources",
            "could not verify",
        )
        has_uncertainty = any(marker in draft_lower for marker in uncertainty_markers)
        fallback_queries = []
        if notes:
            fallback_queries = self._normalize_text_list(
                notes.follow_up_queries,
                max_items=self.max_post_synthesis_follow_up_queries,
                min_len=6,
            )
        return SynthesisReview(
            needs_more_research=has_uncertainty and bool(fallback_queries),
            reason="Fallback sufficiency check triggered due uncertainty language in draft.",
            missing_points=[],
            follow_up_queries=fallback_queries,
        )

    async def _synthesize(
        self,
        query: str,
        search_results: list[dict],
        scraped_content: dict[str, str],
        notes: ResearchNotes | None = None,
    ) -> AsyncGenerator[SSEEvent, None]:
        """Feed all collected data to analyzer for synthesis, streaming the response."""
        full_context, full_names = self._build_synthesis_context(
            query,
            search_results,
            scraped_content,
            notes,
        )

        yield streaming.synthesis_started(len(search_results))

        # Use streaming for the synthesis to get incremental output
        active_client = self.client or llm_client()
        t0 = time.monotonic()
        async with active_client.messages.stream(
            model=self.model,
            max_tokens=8192,
            system=AnalyzerAgent.get_system_prompt(),
            messages=self._build_synthesis_messages(query, full_context),
        ) as stream:
            full_report = ""
            buffer = ""
            async for text in stream.text_stream:
                buffer += text
                full_report += text
                # Yield chunks periodically (every ~100 chars) for smooth streaming
                if len(buffer) >= 100:
                    yield streaming.synthesis_progress(buffer)
                    buffer = ""

            # Flush remaining buffer
            if buffer:
                yield streaming.synthesis_progress(buffer)

        # Log the synthesis call
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        final_msg = await stream.get_final_message()
        await self._log_call("orchestrator.synthesis", final_msg, elapsed_ms)
        usage = getattr(final_msg, "usage", None)
        tokens_used = (
            (getattr(usage, "input_tokens", 0) or 0)
            + (getattr(usage, "output_tokens", 0) or 0)
        )

        # Build source list
        sources = []
        seen_urls: set[str] = set()
        for r in search_results:
            url = r.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                sources.append({
                    "title": r.get("title", ""),
                    "url": url,
                    "domain": web_utils.extract_domain(url),
                })

        full_report = self._promote_full_name_mentions(full_report, full_names)
        full_report = self._strip_deferred_sections(full_report)

        yield streaming.research_complete(
            report=full_report,
            sources=sources[:20],
            tokens_used=tokens_used,
        )

    async def research(self, query: str) -> AsyncGenerator[SSEEvent, None]:
        """Execute the full research pipeline, yielding SSE events throughout."""
        try:
            # Step 1: Generate research plan
            plan_steps = await self._generate_plan(query)
            yield streaming.plan_created(plan_steps)

            # Step 2: Parallel searches
            search_step_offset = 0
            searched_query_keys = {
                " ".join(step.lower().split()) for step in plan_steps if step.strip()
            }
            search_results, search_events = await self._run_parallel_searches(
                plan_steps, step_offset=search_step_offset
            )
            search_step_offset += len(plan_steps)
            for event in search_events:
                yield event

            if not search_results:
                yield streaming.error("No search results found. Try refining your query.")
                return

            # Step 3: Select top URLs to scrape
            top_urls = await self._select_top_urls(search_results)

            # Step 4: Parallel scraping
            scraped_content: dict[str, str] = {}
            if top_urls:
                scraped_content, scrape_events = await self._run_parallel_scrapes(top_urls)
                for event in scrape_events:
                    yield event

            notes = await self._capture_research_notes(query, search_results, scraped_content)
            await self._persist_research_notes(notes, phase="initial", iteration=0)

            for round_index in range(1, self.max_follow_up_rounds + 1):
                follow_up_queries: list[str] = []
                for candidate in notes.follow_up_queries:
                    key = " ".join(candidate.lower().split())
                    if not key or key in searched_query_keys:
                        continue
                    searched_query_keys.add(key)
                    follow_up_queries.append(candidate)
                    if len(follow_up_queries) >= self.max_follow_up_queries_per_round:
                        break

                if not follow_up_queries:
                    break

                follow_up_results, follow_up_events = await self._run_parallel_searches(
                    follow_up_queries,
                    step_offset=search_step_offset,
                )
                search_step_offset += len(follow_up_queries)
                for event in follow_up_events:
                    yield event

                if follow_up_results:
                    search_results = self._merge_search_results(
                        search_results, follow_up_results
                    )
                    new_urls = await self._select_top_urls(
                        follow_up_results, max_urls=4
                    )
                    urls_to_scrape = [url for url in new_urls if url not in scraped_content]
                    if urls_to_scrape:
                        extra_scraped, extra_scrape_events = await self._run_parallel_scrapes(
                            urls_to_scrape
                        )
                        for event in extra_scrape_events:
                            yield event
                        scraped_content.update(extra_scraped)

                updated_notes = await self._capture_research_notes(
                    query,
                    search_results,
                    scraped_content,
                    previous_notes=notes,
                )
                await self._persist_research_notes(
                    updated_notes,
                    phase="follow_up",
                    iteration=round_index,
                )
                if self._notes_stabilized(notes, updated_notes):
                    notes = updated_notes
                    break
                notes = updated_notes
                if not notes.unresolved_points:
                    break

            for review_round in range(1, self.max_post_synthesis_review_rounds + 1):
                yield streaming.agent_progress(
                    "analyzer",
                    status="reviewing_draft",
                    iteration=review_round,
                )
                draft_report = await self._generate_draft_report(
                    query,
                    search_results,
                    scraped_content,
                    notes,
                )
                review = await self._review_report_sufficiency(
                    query,
                    draft_report,
                    notes,
                )
                await self._persist_synthesis_review(review, iteration=review_round)

                if not review.needs_more_research:
                    break

                yield streaming.agent_progress(
                    "analyzer",
                    status="additional_research_required",
                    iteration=review_round,
                    reason=review.reason,
                    missing_points=review.missing_points,
                )

                query_candidates = list(review.follow_up_queries)
                if notes:
                    query_candidates.extend(notes.follow_up_queries)

                post_queries: list[str] = []
                for candidate in query_candidates:
                    key = " ".join(candidate.lower().split())
                    if not key or key in searched_query_keys:
                        continue
                    searched_query_keys.add(key)
                    post_queries.append(candidate)
                    if len(post_queries) >= self.max_post_synthesis_follow_up_queries:
                        break

                if not post_queries:
                    break

                post_results, post_search_events = await self._run_parallel_searches(
                    post_queries,
                    step_offset=search_step_offset,
                )
                search_step_offset += len(post_queries)
                for event in post_search_events:
                    yield event

                if post_results:
                    search_results = self._merge_search_results(search_results, post_results)
                    post_urls = await self._select_top_urls(post_results, max_urls=4)
                    urls_to_scrape = [url for url in post_urls if url not in scraped_content]
                    if urls_to_scrape:
                        extra_scraped, extra_scrape_events = await self._run_parallel_scrapes(
                            urls_to_scrape
                        )
                        for event in extra_scrape_events:
                            yield event
                        scraped_content.update(extra_scraped)

                notes = await self._capture_research_notes(
                    query,
                    search_results,
                    scraped_content,
                    previous_notes=notes,
                )
                await self._persist_research_notes(
                    notes,
                    phase="post_synthesis_follow_up",
                    iteration=review_round,
                )

            # Step 5: Synthesis — streamed
            async for event in self._synthesize(
                query, search_results, scraped_content, notes
            ):
                yield event

        except Exception as e:
            yield streaming.error(f"Research failed: {e}")
