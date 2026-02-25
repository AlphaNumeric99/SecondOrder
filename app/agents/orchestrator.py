from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from datetime import date
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator
from uuid import UUID

from loguru import logger

from app.research_core.evidence.repository import EvidenceRepository
from app.research_core.extract.service import ExtractService
from app.research_core.models.interfaces import ScrapeRequest
from app.research_core.scrape.service import ScrapeService
from app.agents.analyzer_agent import AnalyzerAgent
from app.agents.search_agent import SearchAgent
from app.agents.url_selector import URLSelector
from app.agents.verification_agent import VerificationAgent
from app.config import settings
from app.llm_client import client as llm_client, get_model
from app.models.execution import VerificationResult
from app.models.memory import MemoryRecord
from app.models.research_plan import ResearchPlan, ResearchStep, StepStatus
from app.models.events import SSEEvent
from app.services.execution_compiler import compile_execution_graph, graph_summary
from app.services.env_safety import sanitize_ssl_keylogfile
from app.services.memory_store import get_memory_store
from app.services.prompt_store import render_prompt
from app.services.search_executor import run_deterministic_searches
from app.services import streaming
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
        planner_setting = getattr(settings, "planner_model", "")
        planner_override = planner_setting.strip() if isinstance(planner_setting, str) else ""
        self.planner_model = planner_override or self.model
        self.session_id = session_id
        self.client = None
        self.search_executor_mode = str(settings.search_executor_mode).lower().strip()
        self.search_max_queries_per_step = max(int(settings.search_max_queries_per_step), 1)
        self.search_max_results_per_query = max(int(settings.search_max_results_per_query), 1)
        self.search_max_parallel_requests = max(int(settings.search_max_parallel_requests), 1)
        self.scrape_max_parallel_requests = max(int(settings.scrape_max_parallel_requests), 1)
        self.extract_in_thread = bool(settings.extract_in_thread)
        self.synthesis_context_char_budget = max(
            int(settings.synthesis_context_char_budget), 6000
        )
        self.prompt_bookend_enabled = bool(settings.prompt_bookend_enabled)
        self.prompt_bookend_pattern = str(settings.prompt_bookend_pattern).lower().strip()
        self.review_mode = str(settings.review_mode).lower().strip()
        self.review_min_supported_ratio = float(settings.review_min_supported_ratio)
        self.review_min_citations = max(int(settings.review_min_citations), 1)
        self.max_follow_up_rounds = max(int(settings.max_follow_up_rounds), 0)
        self.max_follow_up_queries_per_round = max(
            int(settings.max_follow_up_queries_per_round), 1
        )
        self.max_post_synthesis_review_rounds = max(
            int(settings.max_post_synthesis_review_rounds), 0
        )
        self.max_post_synthesis_follow_up_queries = max(
            int(settings.max_post_synthesis_follow_up_queries), 1
        )
        self.shadow_mode = bool(settings.shadow_mode)
        self.max_parallel_search = int(settings.max_parallel_search)
        self.max_parallel_extract = int(settings.max_parallel_extract)
        self.max_parallel_verify = int(settings.max_parallel_verify)
        self.scrape_headless_default = bool(settings.scrape_headless_default)
        self.scrape_quality_threshold = float(settings.scrape_quality_threshold)
        self.scrape_pipeline_max_parallel = max(int(settings.scrape_pipeline_max_parallel), 1)
        self.scrape_retry_max = max(int(settings.scrape_retry_max), 0)
        self.scrape_provider = str(settings.scrape_provider).lower().strip()
        self.firecrawl_base_url = str(settings.firecrawl_base_url).strip()
        self.firecrawl_api_key = str(settings.firecrawl_api_key).strip()
        self.jina_reader_base_url = str(settings.jina_reader_base_url).strip()
        self._scrape_service: ScrapeService | None = None
        self._extract_service: ExtractService | None = None
        self._evidence_repository: EvidenceRepository | None = None

    async def _log_call(
        self,
        caller: str,
        response: Any,
        elapsed_ms: int,
        model_name: str | None = None,
    ) -> None:
        """Log an LLM call to the database and file logs."""
        from app.services import logger as log_service
        from app.services import database as db

        try:
            used_model = model_name or self.model
            usage = response.usage if hasattr(response, "usage") else None
            input_tokens = getattr(usage, "input_tokens", 0) if usage else 0
            output_tokens = getattr(usage, "output_tokens", 0) if usage else 0

            await db.log_llm_call(
                model=used_model,
                caller=caller,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                duration_ms=elapsed_ms,
                session_id=UUID(self.session_id) if self.session_id else None,
            )
            # Also log to file for debugging
            log_service.log_llm_call(
                model=used_model,
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
                model=model_name or self.model,
            )

    async def _generate_plan(self, query: str) -> ResearchPlan:
        """Use the configured model to break the research query into a structured plan."""

        def create_structured_plan(steps: list[dict[str, Any]], original_query: str) -> ResearchPlan:
            """Create a ResearchPlan from parsed step data."""
            from uuid import uuid4

            now = datetime.now(timezone.utc).isoformat()
            plan_id = str(uuid4())

            research_steps = []
            for idx, step_data in enumerate(steps):
                step_id = step_data.get("id", f"step-{idx + 1}")
                research_steps.append(
                    ResearchStep(
                        id=step_id,
                        query=step_data.get("query", ""),
                        purpose=step_data.get("purpose", ""),
                        dependencies=step_data.get("dependencies", []),
                        status=StepStatus.PENDING,
                    )
                )

            return ResearchPlan(
                id=plan_id,
                original_query=original_query,
                steps=research_steps,
                version=1,
                created_at=now,
                updated_at=now,
            )

        def create_fallback_plan(original_query: str) -> ResearchPlan:
            """Create a fallback plan using deterministic chain-aware steps."""
            from uuid import uuid4

            now = datetime.now(timezone.utc).isoformat()
            plan_id = str(uuid4())

            # Fallback: create basic steps from query analysis
            steps = augment_plan_steps([], original_query)

            research_steps = []
            for idx, step_query in enumerate(steps):
                research_steps.append(
                    ResearchStep(
                        id=f"step-{idx + 1}",
                        query=step_query,
                        purpose=f"Research step {idx + 1}",
                        dependencies=[],
                        status=StepStatus.PENDING,
                    )
                )

            return ResearchPlan(
                id=plan_id,
                original_query=original_query,
                steps=research_steps,
                version=1,
                created_at=now,
                updated_at=now,
            )
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

        def has_recency_request(text: str) -> bool:
            lowered = text.lower()
            markers = (
                "current",
                "latest",
                "today",
                "as of now",
                "most recent",
                "recent",
            )
            return any(marker in lowered for marker in markers)

        def sanitize_plan_steps(raw_steps: list[str], original_query: str) -> list[str]:
            query_years = set(re.findall(r"\b(?:19|20)\d{2}\b", original_query))
            recency_requested = has_recency_request(original_query)
            blocked_fragments = (
                "strategies are players using",
                "analyze the social or economic reasons",
                "broader implications",
            )

            filtered: list[str] = []
            seen: set[str] = set()
            for step in raw_steps:
                candidate = " ".join(step.split()).strip()
                if not candidate:
                    continue
                lowered = candidate.lower()
                if any(fragment in lowered for fragment in blocked_fragments):
                    continue

                if not recency_requested:
                    step_years = set(re.findall(r"\b(?:19|20)\d{2}\b", candidate))
                    if step_years and (step_years - query_years):
                        continue

                key = lowered
                if key in seen:
                    continue
                seen.add(key)
                filtered.append(candidate)

            return filtered

        def extract_key_phrases(text: str) -> list[str]:
            phrases: list[str] = []
            seen: set[str] = set()

            quoted = re.findall(r'"([^"]{2,120})"', text)
            titled: list[str] = []
            for segment in re.split(r"[?!;:\n]", text):
                titled.extend(
                    re.findall(
                        r"\b[A-Z][A-Za-z0-9&'().-]*(?:\s+[A-Z][A-Za-z0-9&'().-]*){1,4}",
                        segment,
                    )
                )

            for phrase in quoted + titled:
                cleaned = " ".join(phrase.split()).strip(".,:;!?")
                if len(cleaned) < 3:
                    continue
                if ")" in cleaned or "(" in cleaned:
                    continue
                key = cleaned.lower()
                if key in seen:
                    continue
                seen.add(key)
                phrases.append(cleaned)

            return phrases[:4]

        def extract_role_terms(text: str) -> list[str]:
            lowered = text.lower()
            role_aliases: list[tuple[str, str]] = [
                ("drummer", "drummer"),
                ("drums", "drummer"),
                ("guitarist", "guitarist"),
                ("guitar", "guitarist"),
                ("bassist", "bassist"),
                ("bass", "bassist"),
                ("keyboardist", "keyboardist"),
                ("keyboard", "keyboardist"),
                ("vocalist", "vocalist"),
                ("singer", "vocalist"),
                ("researcher", "researcher"),
                ("author", "author"),
                ("director", "director"),
                ("founder", "founder"),
                ("ceo", "ceo"),
                ("captain", "captain"),
                ("player", "player"),
            ]
            roles: list[str] = []
            seen: set[str] = set()
            for token, normalized in role_aliases:
                if token in lowered and normalized not in seen:
                    seen.add(normalized)
                    roles.append(normalized)
            return roles[:3]

        def extract_target_terms(text: str) -> list[str]:
            lowered = text.lower()
            candidates = (
                "songs",
                "song",
                "albums",
                "album",
                "books",
                "book",
                "papers",
                "paper",
                "movies",
                "movie",
                "products",
                "product",
            )
            targets: list[str] = []
            for candidate in candidates:
                if re.search(rf"\b{re.escape(candidate)}\b", lowered):
                    targets.append(candidate)
            deduped = list(dict.fromkeys(targets))
            return deduped[:2]

        def extract_entity_term(text: str) -> str:
            lowered = text.lower()
            candidates = (
                "band",
                "team",
                "company",
                "organization",
                "lab",
                "artist",
                "researcher",
                "person",
                "player",
            )
            for candidate in candidates:
                if re.search(rf"\b{re.escape(candidate)}s?\b", lowered):
                    return candidate
            return "entity"

        def extract_year_window(text: str) -> str:
            lowered = text.lower()
            from_to = re.search(r"\bfrom\s+((?:19|20)\d{2})\s+to\s+((?:19|20)\d{2})\b", lowered)
            if from_to:
                return f"{from_to.group(1)} {from_to.group(2)}"
            between = re.search(
                r"\bbetween\s+((?:19|20)\d{2})\s+and\s+((?:19|20)\d{2})\b",
                lowered,
            )
            if between:
                return f"{between.group(1)} {between.group(2)}"
            years = re.findall(r"\b(?:19|20)\d{2}\b", text)
            if years:
                unique_years = list(dict.fromkeys(years))
                return " ".join(unique_years[:2])
            return ""

        def extract_anchor_year(text: str) -> str:
            in_clause = re.search(
                r'"[^"]+"\s+in\s+the\s+[^.]*?\b((?:19|20)\d{2})\b',
                text,
                flags=re.IGNORECASE,
            )
            if in_clause:
                return in_clause.group(1)

            best_of = re.search(
                r"\bbest\s+of\b[^.]*?\b((?:19|20)\d{2})\b",
                text,
                flags=re.IGNORECASE,
            )
            if best_of:
                return best_of.group(1)

            years = re.findall(r"\b(?:19|20)\d{2}\b", text)
            if years:
                return years[-1]
            return ""

        def dedupe_steps(candidates: list[str]) -> list[str]:
            deduped: list[str] = []
            seen: set[str] = set()
            for candidate in candidates:
                normalized = " ".join(candidate.split()).strip()
                if not normalized:
                    continue
                key = normalized.lower()
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(normalized)
            return deduped[:8]

        def build_chain_decomposition_steps(
            original_query: str,
            base_steps: list[str],
        ) -> list[str]:
            quoted_phrases = re.findall(r'"([^"]{2,120})"', original_query)
            award_phrase = ""
            for phrase in quoted_phrases:
                lowered = phrase.lower()
                if any(token in lowered for token in ("best", "award", "winner", "prize")):
                    award_phrase = phrase
                    break
            if not award_phrase and quoted_phrases:
                award_phrase = quoted_phrases[0]

            key_phrases = extract_key_phrases(original_query)
            context_phrases = [
                phrase
                for phrase in key_phrases
                if phrase.lower() != award_phrase.lower()
            ]
            anchor_bits: list[str] = []
            if award_phrase:
                anchor_bits.append(f'"{award_phrase}"')
            for phrase in context_phrases[:2]:
                anchor_bits.append(f'"{phrase}"')
            if not anchor_bits:
                anchor_bits.append(original_query)

            years = re.findall(r"\b(?:19|20)\d{2}\b", original_query)
            anchor_year = extract_anchor_year(original_query)
            if anchor_year:
                year_fragment = anchor_year
            else:
                year_fragment = " ".join(list(dict.fromkeys(years))[:2])
            entity_term = extract_entity_term(original_query)
            role_terms = extract_role_terms(original_query)
            role_term = role_terms[0] if role_terms else "member"
            year_window = extract_year_window(original_query)
            target_terms = extract_target_terms(original_query)
            target_term = target_terms[0] if target_terms else "key outputs"

            anchor_query = " ".join(anchor_bits).strip()
            candidates: list[str] = [
                f"{anchor_query} {year_fragment} winner {entity_term}".strip(),
                f"{anchor_query} {year_fragment} winner {entity_term} official source".strip(),
                f"{anchor_query} winning {entity_term} members lineup timeline".strip(),
                f"{anchor_query} {role_term} {year_window} timeline".strip(),
                f"{anchor_query} {role_term} {target_term}".strip(),
            ]

            if base_steps:
                candidates.extend(base_steps[:2])
            candidates.append(f"{original_query} corroborating sources")
            return dedupe_steps(candidates)

        def is_low_signal_chain_plan(
            steps: list[str],
            original_query: str,
        ) -> bool:
            if len(steps) < 4:
                return True
            lowered_steps = [step.lower() for step in steps]
            has_anchor = any(
                any(token in step for token in ("winner", "won", "award", "best"))
                for step in lowered_steps
            )
            has_timeline = any(
                any(token in step for token in ("timeline", "members", "lineup", "history"))
                for step in lowered_steps
            )
            roles = extract_role_terms(original_query)
            has_role = True
            if roles:
                has_role = any(any(role in step for role in roles) for step in lowered_steps)
            targets = extract_target_terms(original_query)
            has_target = True
            if targets:
                has_target = any(any(target in step for target in targets) for step in lowered_steps)
            return not (has_anchor and has_timeline and has_role and has_target)

        def augment_plan_steps(base_steps: list[str], original_query: str) -> list[str]:
            if is_chain_query(original_query):
                if is_low_signal_chain_plan(base_steps, original_query):
                    return build_chain_decomposition_steps(original_query, base_steps)
                return dedupe_steps(base_steps)

            to_add: list[str] = list(base_steps)
            if len(to_add) < 3:
                to_add.append(original_query)
                to_add.append(f"{original_query} corroborating sources")
            return dedupe_steps(to_add)

        active_client = self.client or llm_client()
        t0 = time.monotonic()
        response = await active_client.messages.create(
            model=self.planner_model,
            max_tokens=2048,
            system=render_prompt(
                "orchestrator.plan_system",
                today_iso=date.today().isoformat(),
                current_year=date.today().year,
            ),
            messages=[{"role": "user", "content": query}],
        )
        await self._log_call(
            "orchestrator.plan",
            response,
            int((time.monotonic() - t0) * 1000),
            model_name=self.planner_model,
        )

        blocks = getattr(response, "content", None) or []
        text_parts: list[str] = []
        for block in blocks:
            btype = getattr(block, "type", None)
            btext = getattr(block, "text", None)
            is_text_like_type = btype in (None, "text") or not isinstance(btype, str)
            if is_text_like_type and isinstance(btext, str) and btext.strip():
                text_parts.append(btext)

        if not text_parts:
            return create_fallback_plan(query)

        text = "\n".join(text_parts).strip()
        # Extract JSON from response - handle both old array format and new object format
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

            # Try to parse as the new structured format first
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict) and "steps" in parsed:
                    # New structured format
                    steps_data = parsed.get("steps", [])
                    if steps_data:
                        return create_structured_plan(steps_data, query)
            except json.JSONDecodeError:
                pass

            # Fallback to old array format for backwards compatibility
            start = text.find("[")
            end = text.rfind("]")
            if start >= 0 and end > start:
                text = text[start : end + 1]

            parsed = sanitize_plan_steps(normalize(json.loads(text)), query)
            if is_chain_query(query):
                steps = augment_plan_steps(parsed, query)
            elif len(parsed) < 3:
                steps = augment_plan_steps(parsed, query)
            else:
                steps = parsed[:8]

            # Convert to structured format
            research_steps = []
            for idx, step_query in enumerate(steps):
                research_steps.append(
                    ResearchStep(
                        id=f"step-{idx + 1}",
                        query=step_query,
                        purpose=f"Research step {idx + 1}",
                        dependencies=[],
                        status=StepStatus.PENDING,
                    )
                )

            from uuid import uuid4
            now = datetime.now(timezone.utc).isoformat()
            return ResearchPlan(
                id=str(uuid4()),
                original_query=query,
                steps=research_steps,
                version=1,
                created_at=now,
                updated_at=now,
            )
        except (json.JSONDecodeError, IndexError):
            # Fallback: build deterministic chain-aware steps.
            return create_fallback_plan(query)

    async def _run_parallel_searches(
        self, plan_steps: list[str], *, step_offset: int = 0
    ) -> tuple[list[dict[str, Any]], list[SSEEvent]]:
        """Run searches with project-configured execution strategy."""
        if self.search_executor_mode == "agent_loop":
            return await self._run_parallel_searches_agent_loop(
                plan_steps, step_offset=step_offset
            )
        return await run_deterministic_searches(
            plan_steps,
            step_offset=step_offset,
            max_parallel=self.search_max_parallel_requests,
            max_queries_per_step=self.search_max_queries_per_step,
            max_results_per_query=self.search_max_results_per_query,
        )

    async def _run_parallel_searches_bounded(
        self,
        plan_steps: list[str],
        *,
        step_offset: int = 0,
        max_parallel: int,
    ) -> tuple[list[dict[str, Any]], list[SSEEvent]]:
        """Run searches with explicit bounded parallelism."""
        if self.search_executor_mode == "agent_loop":
            return await self._run_parallel_searches_bounded_agent_loop(
                plan_steps,
                step_offset=step_offset,
                max_parallel=max_parallel,
            )
        return await run_deterministic_searches(
            plan_steps,
            step_offset=step_offset,
            max_parallel=max_parallel,
            max_queries_per_step=self.search_max_queries_per_step,
            max_results_per_query=self.search_max_results_per_query,
        )

    async def _run_parallel_searches_agent_loop(
        self, plan_steps: list[str], *, step_offset: int = 0
    ) -> tuple[list[dict[str, Any]], list[SSEEvent]]:
        """Legacy SearchAgent tool-loop path retained for rollback."""
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

        for i, step in enumerate(plan_steps):
            all_events.append(
                streaming.agent_started("search", step=step_offset + i, query=step)
            )

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

    async def _run_parallel_searches_bounded_agent_loop(
        self,
        plan_steps: list[str],
        *,
        step_offset: int = 0,
        max_parallel: int,
    ) -> tuple[list[dict[str, Any]], list[SSEEvent]]:
        """Legacy SearchAgent tool-loop path with explicit concurrency bound."""
        all_events: list[SSEEvent] = []
        all_results: list[dict[str, Any]] = []
        semaphore = asyncio.Semaphore(max(max_parallel, 1))

        for i, step in enumerate(plan_steps):
            all_events.append(
                streaming.agent_started("search", step=step_offset + i, query=step)
            )

        async def run_one(step: str, idx: int) -> tuple[list[dict[str, Any]], list[SSEEvent]]:
            async with semaphore:
                agent = SearchAgent(
                    model=self.model,
                    step_index=step_offset + idx,
                    session_id=self.session_id,
                )
                events: list[SSEEvent] = []
                async for event in agent.run(
                    step, context=f"This is step {agent.step_index} of the research plan."
                ):
                    events.append(event)
                return agent.all_results, events

        raw_results = await asyncio.gather(
            *(run_one(step, idx) for idx, step in enumerate(plan_steps)),
            return_exceptions=True,
        )

        for idx, item in enumerate(raw_results):
            if isinstance(item, Exception):
                all_events.append(streaming.error(str(item), agent="search"))
            else:
                results, events = item
                all_results.extend(results)
                all_events.extend(events)
            all_events.append(streaming.agent_completed("search", step=step_offset + idx))

        return all_results, all_events

    @staticmethod
    def _query_terms(query: str) -> tuple[list[str], list[str]]:
        stopwords = {
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "by",
            "for",
            "from",
            "how",
            "in",
            "is",
            "it",
            "its",
            "name",
            "of",
            "on",
            "or",
            "that",
            "the",
            "their",
            "then",
            "these",
            "this",
            "those",
            "to",
            "two",
            "was",
            "were",
            "what",
            "when",
            "which",
            "who",
            "with",
            "won",
            "only",
            "provide",
            "according",
        }
        lowered = (query or "").lower()
        tokens = re.findall(r"[a-z0-9]{3,}", lowered)
        terms = [t for t in tokens if t not in stopwords]
        # Keep first-seen order but deduplicate.
        deduped_terms: list[str] = []
        seen_terms: set[str] = set()
        for term in terms:
            if term in seen_terms:
                continue
            seen_terms.add(term)
            deduped_terms.append(term)

        raw_phrases = re.findall(r'"([^"]{3,120})"', query or "")
        phrases = [" ".join(p.lower().split()).strip() for p in raw_phrases if p.strip()]
        return deduped_terms, phrases

    async def _select_top_urls(
        self,
        search_results: list[dict[str, Any]],
        max_urls: int = 8,
        *,
        query: str = "",
    ) -> list[str]:
        """Select the top URLs to scrape based on relevance scores.

        Delegates to URLSelector for core logic.
        """
        selector = URLSelector()
        return selector.select_top_urls(search_results, query, max_urls)

        counts: dict[str, int] = {}
        for result in search_results[:80]:
            title = str(result.get("title", "") or "").strip()
            url = str(result.get("url", "") or "").strip()
            domain = web_utils.extract_domain(url).lower()
            if not title:
                continue

            candidates = self._extract_title_case_phrases(title)
            if "wikipedia.org" in domain:
                wiki_title = re.sub(r"\s*-\s*Wikipedia\s*$", "", title, flags=re.IGNORECASE).strip()
                if wiki_title:
                    candidates.append(wiki_title)

            bonus = 2 if "wikipedia.org" in domain else 1
            for candidate in candidates:
                normalized = " ".join(candidate.split()).strip().lower()
                if not normalized:
                    continue
                if self._is_generic_entity_phrase(normalized):
                    continue
                counts[normalized] = counts.get(normalized, 0) + bonus

        ranked = sorted(counts.items(), key=lambda item: (item[1], len(item[0])), reverse=True)
        hints: list[str] = []
        for hint, score in ranked:
            if score < 2:
                continue
            hints.append(hint)
            if len(hints) >= 6:
                break
        return hints

    @staticmethod
    def _is_chain_like_query(query: str) -> bool:
        lowered = (query or "").lower()
        markers = (
            "who",
            "that",
            "which",
            "from 20",
            "from 19",
            "played",
            "won",
            "name the two",
        )
        return sum(1 for marker in markers if marker in lowered) >= 3

    def _extract_chain_components(self, query: str) -> dict[str, list[str]]:
        """Extract chain components from multi-hop queries for relation-aware retrieval."""
        lowered = query.lower()

        # Known relation patterns for chain queries
        relation_patterns = [
            "played drums",
            "played",
            "won",
            "from 20",
            "from 19",
            "best of",
            "viral on",
            "drummer in",
            "band that",
            "artist who",
        ]

        relations = [r for r in relation_patterns if r in lowered]

        # Extract potential entity mentions - look for quoted phrases and capitalized sequences
        primary_candidates = []

        # Look for quoted phrases (exact matches)
        import re
        quoted = re.findall(r'"([^"]+)"', query)
        primary_candidates.extend([q.lower() for q in quoted])

        # For queries like "artist who from 2016 to 2021 played drums in a band that won..."
        # extract the key entities: artist, band, award
        if "band that won" in lowered or "band who won" in lowered:
            # Extract band name after "in a band that won"
            match = re.search(r'in a (?:band that|band who) won ["\']?([^"\']+)["\']?', lowered)
            if match:
                primary_candidates.append(match.group(1).strip())

        return {
            "primary": primary_candidates[0] if primary_candidates else "",
            "relations": relations,
            "all_candidates": primary_candidates,
        }

    @staticmethod
    def _is_generic_entity_phrase(value: str) -> bool:
        lowered = value.lower().strip()
        if not lowered:
            return True
        generic_fragments = (
            "best ",
            "award",
            "timeline",
            "wikipedia",
            "newsroom",
            "top ",
            "list of ",
            "people and places",
            "share to",
            "email facebook",
        )
        if any(fragment in lowered for fragment in generic_fragments):
            return True
        words = lowered.split()
        if len(words) == 1:
            token = words[0]
            if token in {"band", "artist", "music", "songs", "video", "viral"}:
                return True
            if token in {"film", "drums"}:
                return True
            if len(token) < 3:
                return True
        if all(word in {"share", "to", "email", "facebook", "x"} for word in words):
            return True
        return False

    @staticmethod
    def _extract_title_case_phrases(text: str) -> list[str]:
        phrases = re.findall(
            r"\b[A-Z][A-Za-z0-9&'().-]*(?:\s+[A-Z][A-Za-z0-9&'().-]*){1,4}",
            text,
        )
        phrases.extend(
            re.findall(
                r"\b([A-Z][A-Za-z0-9&'().-]{2,})\s*\((?:musician|artist|drummer)\b",
                text,
                flags=re.IGNORECASE,
            )
        )
        cleaned: list[str] = []
        seen: set[str] = set()
        for phrase in phrases:
            value = " ".join(phrase.split()).strip(".,:;!?()[]")
            if not value:
                continue
            if len(value.split()) < 2 and len(value) < 4:
                continue
            if ResearchOrchestrator._is_generic_entity_phrase(value):
                continue
            key = value.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(value)
        return cleaned

    def _build_chain_bootstrap_queries(
        self,
        query: str,
        search_results: list[dict[str, Any]],
    ) -> list[str]:
        if not self._is_chain_like_query(query):
            return []
        lowered_query = (query or "").lower()

        quoted_phrases = [
            " ".join(p.lower().split()).strip()
            for p in re.findall(r'"([^"]{3,120})"', query or "")
            if p.strip()
        ]
        years = list(dict.fromkeys(re.findall(r"\b(?:19|20)\d{2}\b", query or "")))

        blocked_substrings = (
            "wikipedia",
            "people and places",
            "newsroom",
        )

        candidate_counts: dict[str, int] = {}
        for result in search_results[:50]:
            title = str(result.get("title", "") or "")
            snippet = str(result.get("content", "") or "")
            url = str(result.get("url", "") or "")
            haystack = f"{title}\n{snippet}"
            lowered_haystack = haystack.lower()
            domain = web_utils.extract_domain(url).lower()
            trusted_domains = (
                "wikipedia.org",
                "musicbrainz.org",
                "last.fm",
                "allmusic.com",
                "discogs.com",
            )
            has_quoted_hit = bool(
                quoted_phrases and any(phrase in lowered_haystack for phrase in quoted_phrases)
            )
            if not has_quoted_hit and not any(token in domain for token in trusted_domains):
                continue
            if not has_quoted_hit and quoted_phrases and "wikipedia.org" not in domain:
                continue
            bonus = 0
            if has_quoted_hit:
                bonus += 2
            if "wikipedia.org" in domain:
                bonus += 1

            if "wikipedia.org" in domain:
                wiki_title = re.sub(r"\s*-\s*Wikipedia\s*$", "", title, flags=re.IGNORECASE).strip()
                if wiki_title:
                    candidate_counts[wiki_title] = candidate_counts.get(wiki_title, 0) + 2

            for phrase in self._extract_title_case_phrases(title):
                key = phrase.lower()
                if key in lowered_query:
                    continue
                if any(blocked in key for blocked in blocked_substrings):
                    continue
                if self._is_generic_entity_phrase(phrase):
                    continue
                if len(phrase) > 64:
                    continue
                score = 1 + bonus
                lowered_title = title.lower()
                if re.search(r"\((?:musician|band|artist|drummer)\)", lowered_title):
                    score += 2
                candidate_counts[phrase] = candidate_counts.get(phrase, 0) + score

        if not candidate_counts:
            return []

        ranked_candidates = [
            item
            for item in sorted(
                candidate_counts.items(),
                key=lambda item: (item[1], len(item[0])),
                reverse=True,
            )
            if item[1] >= 2
        ]
        if not ranked_candidates:
            return []

        wants_membership = "member" in lowered_query or "played" in lowered_query or "drum" in lowered_query
        year_hint = " ".join(years[:2]).strip()
        quoted_hint = quoted_phrases[0] if quoted_phrases else ""

        queries: list[str] = []
        for candidate, _ in ranked_candidates[:3]:
            queries.append(f'"{candidate}" site:wikipedia.org')
            if wants_membership:
                suffix = f" {year_hint}" if year_hint else ""
                queries.append(f'"{candidate}" members{suffix}'.strip())
                queries.append(f'"{candidate}" lineup history{suffix}'.strip())
            if "song" in lowered_query or "music" in lowered_query:
                queries.append(f'"{candidate}" notable songs')

        if quoted_hint:
            if year_hint:
                queries.append(f'"{quoted_hint}" relationship timeline {year_hint}')
            queries.append(f'"{quoted_hint}" official source')

        deduped: list[str] = []
        seen: set[str] = set()
        for q in queries:
            key = " ".join(q.lower().split())
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(q)
            if len(deduped) >= 6:
                break
        return deduped

    def _extract_band_candidates_from_results(
        self,
        search_results: list[dict[str, Any]],
        query: str = "",
    ) -> list[str]:
        counts: dict[str, int] = {}
        quoted_phrases = [
            " ".join(p.lower().split()).strip()
            for p in re.findall(r'"([^"]{3,120})"', query or "")
            if p.strip()
        ]
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
        query_terms = [
            token
            for token in re.findall(r"[a-z0-9]{4,}", (query or "").lower())
            if token not in stopwords
        ][:12]
        trusted_domains = (
            "wikipedia.org",
            "musicbrainz.org",
            "allmusic.com",
            "discogs.com",
            "last.fm",
            "genius.com",
        )
        for result in search_results[:120]:
            title = str(result.get("title", "") or "").strip()
            snippet = str(result.get("content", "") or "").strip()
            url = str(result.get("url", "") or "").strip()
            if not title:
                continue
            lowered_title = title.lower()
            lowered_haystack = f"{title}\n{snippet}\n{url}".lower()
            domain = web_utils.extract_domain(url).lower()
            phrase_hit = any(phrase in lowered_haystack for phrase in quoted_phrases)
            term_hits = sum(1 for term in query_terms if term in lowered_haystack)
            domain_hit = any(token in domain for token in trusted_domains)
            contextual = phrase_hit or term_hits >= 2 or domain_hit
            if not contextual:
                continue
            wiki_band_match = re.match(r"(.+?)\s*\((?:band)\)\s*-\s*wikipedia\s*$", title, flags=re.IGNORECASE)
            if wiki_band_match:
                candidate = " ".join(wiki_band_match.group(1).split()).strip(".,:;!?()[]")
                if candidate and not self._is_generic_entity_phrase(candidate):
                    counts[candidate] = counts.get(candidate, 0) + 4

            for phrase in self._extract_title_case_phrases(title):
                if self._is_generic_entity_phrase(phrase):
                    continue
                score = 1
                if "band" in lowered_title:
                    score += 1
                if "musician" in lowered_title:
                    score += 1
                counts[phrase] = counts.get(phrase, 0) + score

        ranked = sorted(counts.items(), key=lambda item: (item[1], len(item[0])), reverse=True)
        selected: list[str] = []
        for candidate, score in ranked:
            if score < 2:
                continue
            selected.append(candidate)
            if len(selected) >= 4:
                break
        return selected

    def _extract_band_candidates_from_scraped(
        self,
        query: str,
        scraped_content: dict[str, str],
    ) -> list[str]:
        quoted_phrases = [
            " ".join(p.split()).strip()
            for p in re.findall(r'"([^"]{3,120})"', query or "")
            if p.strip()
        ]
        counts: dict[str, int] = {}

        def normalize_candidate(raw_value: str) -> str:
            candidate = " ".join(raw_value.split()).strip(".,:;!?()[]")
            candidate = re.sub(r"^[#*\-]+", "", candidate).strip(".,:;!?()[] ")
            lowered_candidate = candidate.lower()
            share_index = lowered_candidate.find(" share ")
            if share_index >= 0:
                candidate = candidate[:share_index].strip()
            candidate = re.split(
                r"\b(?:Share|Read|Advertisement|Copy Link|People & Places)\b",
                candidate,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0].strip(".,:;!?()[] ")
            return candidate

        for _, content in scraped_content.items():
            if not content:
                continue
            text = " ".join(content[:120000].split())
            found_phrase_match = False
            for phrase in quoted_phrases:
                phrase_patterns = [
                    re.compile(
                        rf"(?:#{{1,6}}\s*)?{re.escape(phrase)}(?:\s*#{{1,6}})?\s+([A-Z][A-Za-z0-9&'().-]*(?:\s+(?!Share\b)[A-Z][A-Za-z0-9&'().-]*){{0,5}})",
                        flags=re.IGNORECASE,
                    ),
                    re.compile(
                        rf"{re.escape(phrase)}[^A-Za-z0-9]+([A-Z][A-Za-z0-9&'().-]*(?:\s+[A-Z][A-Za-z0-9&'().-]*){{0,5}})",
                        flags=re.IGNORECASE,
                    ),
                ]
                for pattern in phrase_patterns:
                    for match in pattern.findall(text):
                        candidate = normalize_candidate(match)
                        if self._is_generic_entity_phrase(candidate):
                            continue
                        counts[candidate] = counts.get(candidate, 0) + 8
                        found_phrase_match = True

            if found_phrase_match:
                continue

            for fallback_phrase in self._extract_title_case_phrases(text[:30000]):
                if self._is_generic_entity_phrase(fallback_phrase):
                    continue
                counts[fallback_phrase] = counts.get(fallback_phrase, 0) + 1

        normalized_counts: dict[str, int] = {}
        for candidate, score in counts.items():
            cleaned = re.sub(r"\s+Share\b.*$", "", candidate, flags=re.IGNORECASE).strip(
                ".,:;!?()[] "
            )
            if not cleaned:
                continue
            if self._is_generic_entity_phrase(cleaned):
                continue
            normalized_counts[cleaned] = max(normalized_counts.get(cleaned, 0), score)

        ranked = sorted(
            normalized_counts.items(),
            key=lambda item: (item[1], len(item[0])),
            reverse=True,
        )
        min_score = 1
        if ranked and ranked[0][1] >= 4:
            min_score = 2
        selected: list[str] = []
        for candidate, score in ranked:
            if score < min_score:
                continue
            selected.append(candidate)
            if len(selected) >= 3:
                break
        return selected

    def _extract_person_candidates_from_content(
        self,
        search_results: list[dict[str, Any]],
        scraped_content: dict[str, str],
        *,
        band_candidates: list[str] | None = None,
    ) -> list[str]:
        name_pattern = re.compile(r"\b([A-Z][a-z]+(?: [A-Z][a-z]+){1,2})\b")
        counts: dict[str, int] = {}
        banned_first_tokens = {
            "Retrieved",
            "Category",
            "Wikipedia",
            "People",
            "Pages",
            "Music",
            "Best",
            "Share",
            "Jump",
            "Edit",
        }
        band_hints = [
            " ".join(candidate.lower().split()).strip()
            for candidate in (band_candidates or [])
            if candidate
        ]

        def is_band_context(text: str) -> bool:
            if not band_hints:
                return True
            lowered = text.lower()
            for hint in band_hints:
                if hint in lowered:
                    return True
                slug = re.sub(r"[^a-z0-9]+", "-", hint).strip("-")
                if slug and slug in lowered:
                    return True
            return False

        def include_name(name: str) -> bool:
            first = name.split()[0]
            if first in banned_first_tokens:
                return False
            if self._is_generic_entity_phrase(name):
                return False
            return True

        for result in search_results[:120]:
            title = str(result.get("title", "") or "")
            snippet = str(result.get("content", "") or "")
            url = str(result.get("url", "") or "")
            combined = f"{title}\n{snippet}"
            if "drum" not in combined.lower():
                continue
            if not is_band_context(f"{title}\n{snippet}\n{url}"):
                continue
            for match in name_pattern.finditer(combined):
                name = match.group(1)
                start, end = match.span(1)
                window = combined[max(0, start - 120) : min(len(combined), end + 120)].lower()
                if (
                    "drum" not in window
                    and "member" not in window
                    and "lineup" not in window
                ):
                    continue
                if not include_name(name):
                    continue
                counts[name] = counts.get(name, 0) + 1

        for url, content in list(scraped_content.items())[:20]:
            text = content[:18000]
            if "drum" not in text.lower():
                continue
            if not is_band_context(f"{url}\n{text[:4000]}"):
                continue
            for match in name_pattern.finditer(text):
                name = match.group(1)
                start, end = match.span(1)
                window = text[max(0, start - 160) : min(len(text), end + 160)].lower()
                if (
                    "drum" not in window
                    and "member" not in window
                    and "lineup" not in window
                ):
                    continue
                if not include_name(name):
                    continue
                counts[name] = counts.get(name, 0) + 2

        ranked = sorted(counts.items(), key=lambda item: (item[1], len(item[0])), reverse=True)
        selected: list[str] = []
        for candidate, score in ranked:
            if score < 2:
                continue
            selected.append(candidate)
            if len(selected) >= 4:
                break
        return selected

    def _build_chain_enrichment_queries(
        self,
        query: str,
        search_results: list[dict[str, Any]],
        scraped_content: dict[str, str],
    ) -> list[str]:
        if not self._is_chain_like_query(query):
            return []

        lowered_query = (query or "").lower()
        years = list(dict.fromkeys(re.findall(r"\b(?:19|20)\d{2}\b", query or "")))
        year_hint = " ".join(years[:2]).strip()
        entity_candidates_from_scraped = self._extract_band_candidates_from_scraped(
            query,
            scraped_content,
        )
        entity_candidates_from_results = self._extract_band_candidates_from_results(
            search_results,
            query=query,
        )
        entity_candidates: list[str] = []
        if entity_candidates_from_scraped:
            for candidate in entity_candidates_from_scraped:
                if candidate in entity_candidates:
                    continue
                entity_candidates.append(candidate)
                if len(entity_candidates) >= 3:
                    break
        else:
            for candidate in entity_candidates_from_results:
                if candidate in entity_candidates:
                    continue
                entity_candidates.append(candidate)
                if len(entity_candidates) >= 4:
                    break
        person_candidates = self._extract_person_candidates_from_content(
            search_results,
            scraped_content,
            band_candidates=entity_candidates,
        )

        queries: list[str] = []
        for entity in entity_candidates[:3]:
            queries.append(f'"{entity}" site:wikipedia.org')
            suffix = f" {year_hint}" if year_hint else ""
            queries.append(f'"{entity}" members timeline{suffix}'.strip())
            queries.append(f'"{entity}" lineup history{suffix}'.strip())
            if "song" in lowered_query or "music" in lowered_query:
                queries.append(f'"{entity}" notable songs')
            if "award" in lowered_query or "won" in lowered_query:
                queries.append(f'"{entity}" awards history')

            parts = [part for part in re.findall(r"[A-Za-z0-9]+", entity) if part]
            if 2 <= len(parts) <= 5:
                acronym = "".join(part[0].upper() for part in parts if part[0].isalpha())
                if len(acronym) >= 2:
                    queries.append(f'"{acronym}" membership timeline')
                    if year_hint:
                        queries.append(f'"{acronym}" {year_hint}')

            for person in person_candidates[:2]:
                queries.append(f'"{entity}" "{person}"')

        for person in person_candidates[:3]:
            queries.append(f'"{person}" site:wikipedia.org')
            queries.append(f'"{person}" biography')
            if year_hint:
                queries.append(f'"{person}" career timeline {year_hint}')

        deduped: list[str] = []
        seen: set[str] = set()
        for query_item in queries:
            key = " ".join(query_item.lower().split())
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(query_item)
            if len(deduped) >= 10:
                break
        return deduped

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
        """Scrape URLs with configured concurrency using direct Hasdata calls."""
        return await self._run_parallel_scrapes_bounded(
            urls,
            max_parallel=self.scrape_max_parallel_requests,
        )

    async def _run_parallel_scrapes_bounded(
        self, urls: list[str], *, max_parallel: int
    ) -> tuple[dict[str, str], list[SSEEvent]]:
        """Scrape URLs with explicit concurrency limits."""
        return await self._run_parallel_scrapes_primary(urls, max_parallel=max_parallel)

    def _get_scrape_services(self) -> tuple[ScrapeService, ExtractService, EvidenceRepository]:
        if self._scrape_service is None:
            self._scrape_service = ScrapeService(
                retry_max=self.scrape_retry_max,
                provider=self.scrape_provider,
                firecrawl_base_url=self.firecrawl_base_url,
                firecrawl_api_key=self.firecrawl_api_key,
                jina_reader_base_url=self.jina_reader_base_url,
            )
        if self._extract_service is None:
            self._extract_service = ExtractService(
                max_chars=settings.extractor_max_page_chars,
            )
        if self._evidence_repository is None:
            self._evidence_repository = EvidenceRepository()
        return (
            self._scrape_service,
            self._extract_service,
            self._evidence_repository,
        )

    async def _run_parallel_scrapes_primary(
        self,
        urls: list[str],
        *,
        max_parallel: int,
    ) -> tuple[dict[str, str], list[SSEEvent]]:
        sanitize_ssl_keylogfile()
        all_events: list[SSEEvent] = [
            streaming.agent_started("scraper", url=url) for url in urls
        ]
        content_map: dict[str, str] = {}
        scrape_service, extract_service, evidence_repo = self._get_scrape_services()
        semaphore = asyncio.Semaphore(max(min(max_parallel, self.scrape_pipeline_max_parallel), 1))

        async def run_one(url: str) -> None:
            async with semaphore:
                all_events.append(
                    streaming.agent_progress(
                        "scraper",
                        url=url,
                        status="scraping",
                        provider=self.scrape_provider,
                    )
                )
                extracted = ""
                try:
                    request = ScrapeRequest(
                        url=url,
                        render_mode="headless_default"
                        if self.scrape_headless_default
                        else "http_only",
                        timeout_profile="standard",
                        domain_policy_id="default",
                    )
                    artifact = await scrape_service.scrape(request)
                    raw_html = Path(artifact.rendered_html_path).read_text(
                        encoding="utf-8",
                        errors="ignore",
                    )

                    if self.extract_in_thread:
                        extraction = await asyncio.to_thread(
                            extract_service.extract,
                            url=url,
                            raw_html=raw_html,
                            quality_threshold=self.scrape_quality_threshold,
                        )
                    else:
                        extraction = extract_service.extract(
                            url=url,
                            raw_html=raw_html,
                            quality_threshold=self.scrape_quality_threshold,
                        )

                    evidence_repo.persist_artifact(artifact)
                    evidence_repo.persist_extraction(extraction)
                    records = evidence_repo.build_records(
                        url=url,
                        extraction=extraction,
                        metadata={
                            "policy_applied": artifact.policy_applied,
                            "status_code": artifact.status_code,
                            "provider": self.scrape_provider,
                        },
                    )
                    evidence_repo.persist_records(url=url, records=records)
                    extracted = web_utils.clean_content(
                        extraction.content_text,
                        max_length=settings.extractor_max_page_chars,
                    )
                    if extracted:
                        all_events.append(streaming.scrape_result(url, extracted[:500]))
                    all_events.append(
                        streaming.agent_completed(
                            "scraper",
                            url=url,
                            success=bool(extracted),
                            content_length=len(extracted),
                            method=extraction.method,
                            quality_score=extraction.quality_score,
                            evidence_chunks=len(records),
                            provider=self.scrape_provider,
                        )
                    )
                except Exception as exc:
                    all_events.append(
                        streaming.error(f"Failed to scrape {url}: {exc}", agent="scraper")
                    )
                    all_events.append(
                        streaming.agent_completed(
                            "scraper",
                            url=url,
                            success=False,
                            content_length=0,
                        )
                    )
                content_map[url] = extracted

        await asyncio.gather(*(run_one(url) for url in urls), return_exceptions=True)
        return content_map, all_events

    @staticmethod
    def _chunk_text(text: str, *, chunk_size: int = 1800, overlap: int = 200) -> list[str]:
        if not text.strip():
            return []
        normalized = " ".join(text.split())
        chunks: list[str] = []
        start = 0
        step = max(chunk_size - overlap, 200)
        while start < len(normalized):
            chunk = normalized[start : start + chunk_size].strip()
            if chunk:
                chunks.append(chunk)
            start += step
        return chunks

    async def _upsert_memory_chunks(
        self,
        *,
        scraped_content: dict[str, str],
        default_step_id: str = "extract",
    ) -> tuple[dict[str, int], list[SSEEvent]]:
        if not self.session_id:
            return {"inserted_chunks": 0, "deduplicated_chunks": 0, "documents_processed": 0}, []
        store = get_memory_store()
        now = datetime.now(timezone.utc).isoformat()
        records: list[MemoryRecord] = []
        dedupe_seen: set[str] = set()

        for url, text in scraped_content.items():
            chunks = self._chunk_text(text)
            for index, chunk in enumerate(chunks):
                chunk_hash = hashlib.sha1(f"{url}|{chunk}".encode("utf-8")).hexdigest()
                record_id = f"{chunk_hash}_{index}"
                if record_id in dedupe_seen:
                    continue
                dedupe_seen.add(record_id)
                records.append(
                    MemoryRecord(
                        id=record_id,
                        session_id=self.session_id,
                        step_id=default_step_id,
                        url=url,
                        text=chunk,
                        chunk_hash=chunk_hash,
                        created_at=now,
                    )
                )

        result = await store.upsert(records)
        payload = {
            "inserted_chunks": int(result.inserted),
            "deduplicated_chunks": int(result.deduplicated),
            "documents_processed": len(scraped_content),
        }
        return payload, [streaming.memory_upserted(**payload)]

    async def _run_verification_stage(
        self, *, query: str, plan_steps: list[str]
    ) -> tuple[list[VerificationResult], list[SSEEvent]]:
        if not self.session_id:
            return [], []
        store = get_memory_store()
        verifier = VerificationAgent()
        tasks = verifier.build_tasks(query, plan_steps, max_tasks=10)
        events: list[SSEEvent] = []
        for task in tasks:
            events.append(
                streaming.verification_started(
                    task_id=task.id,
                    claim=task.claim,
                    step_id=task.step_id,
                )
            )
        results = await verifier.verify_tasks(
            query=query,
            tasks=tasks,
            session_id=self.session_id,
            memory_store=store,
            max_parallel=self.max_parallel_verify,
        )
        for result in results:
            events.append(
                streaming.verification_completed(
                    task_id=result.task_id,
                    status=result.status,
                    score=result.score,
                    reason=result.reason,
                    citations=result.citations,
                )
            )
        return results, events

    @staticmethod
    def _verification_summary(results: list[VerificationResult]) -> dict[str, Any]:
        summary = {
            "total": len(results),
            "supported": 0,
            "partially_supported": 0,
            "unsupported": 0,
            "avg_score": 0.0,
        }
        if not results:
            return summary
        total_score = 0.0
        for result in results:
            summary[result.status] = summary.get(result.status, 0) + 1
            total_score += float(result.score)
        summary["avg_score"] = round(total_score / len(results), 4)
        return summary

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

    @staticmethod
    def _sanitize_search_results(
        results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        noisy_domains = {
            "reddit.com",
            "www.reddit.com",
            "tiktok.com",
            "www.tiktok.com",
            "youtube.com",
            "www.youtube.com",
            "facebook.com",
            "www.facebook.com",
            "tokchart.com",
            "huggingface.co",
        }
        by_url: dict[str, dict[str, Any]] = {}
        for result in results:
            url = result.get("url")
            if not isinstance(url, str) or not web_utils.is_valid_url(url):
                continue
            domain = web_utils.extract_domain(url).lower()
            if domain in noisy_domains:
                continue
            previous = by_url.get(url)
            if previous is None or float(result.get("score", 0.0) or 0.0) > float(
                previous.get("score", 0.0) or 0.0
            ):
                by_url[url] = result

        ordered = sorted(
            by_url.values(),
            key=lambda item: float(item.get("score", 0.0) or 0.0),
            reverse=True,
        )
        return ordered

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
        for i, result in enumerate(search_results[:12]):
            title = result.get("title", "Untitled")
            url = result.get("url", "")
            snippet = str(result.get("content", ""))[:260]
            search_evidence.append(
                f"{i + 1}. title={title}\nurl={url}\nsnippet={snippet}"
            )

        scrape_evidence: list[str] = []
        for i, (url, content) in enumerate(list(scraped_content.items())[:5]):
            excerpt = " ".join(content.split())[:1800]
            scrape_evidence.append(f"{i + 1}. url={url}\nexcerpt={excerpt}")

        previous_payload = previous_notes.to_dict() if previous_notes else {}
        search_evidence_text = chr(10).join(search_evidence) or "none"
        scrape_evidence_text = chr(10).join(scrape_evidence) or "none"

        t0 = time.monotonic()
        response = await active_client.messages.create(
            model=self.model,
            max_tokens=1400,
            system=render_prompt(
                "orchestrator.notes_system",
                today_iso=date.today().isoformat(),
            ),
            messages=[
                {
                    "role": "user",
                    "content": self._bookend_query_context(
                        query,
                        render_prompt(
                            "orchestrator.notes_user",
                            query=query,
                            previous_notes_json=json.dumps(previous_payload),
                            search_evidence=search_evidence_text,
                            scrape_evidence=scrape_evidence_text,
                        ),
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
        from app.services import database as db

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
        from app.services import database as db

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

    def _should_run_synthesis_review(
        self,
        *,
        notes: ResearchNotes | None,
        search_results: list[dict[str, Any]],
        verification_results: list[VerificationResult] | None = None,
    ) -> bool:
        mode = self.review_mode
        if mode == "off":
            return False
        if mode == "always":
            return True

        if notes and notes.unresolved_points:
            return True
        if len(search_results) < self.review_min_citations:
            return True

        if verification_results:
            total = len(verification_results)
            if total > 0:
                supported = sum(1 for r in verification_results if r.status == "supported")
                ratio = supported / total
                if ratio < self.review_min_supported_ratio:
                    return True

        return False

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
        budget = self.synthesis_context_char_budget
        context_parts: list[str] = []
        used_chars = 0

        def append_with_budget(text: str) -> None:
            nonlocal used_chars
            if not text or used_chars >= budget:
                return
            remaining = budget - used_chars
            chunk = text if len(text) <= remaining else text[:remaining]
            context_parts.append(chunk)
            used_chars += len(chunk)

        append_with_budget(f"# Current Date\n{today.isoformat()} ({today.year})\n")
        append_with_budget(f"\n# Original Research Query\n{query}\n")

        append_with_budget("\n# Search Results Summary\n")
        for index, result in enumerate(search_results[:20]):
            title = str(result.get("title", "Untitled"))
            url = str(result.get("url", ""))
            snippet = str(result.get("content", ""))[:300]
            append_with_budget(
                f"\n## Source {index + 1}: [{title}]({url})\n{snippet}\n"
            )

        if scraped_content:
            score_by_url: dict[str, float] = {}
            for result in search_results:
                url = result.get("url")
                if not isinstance(url, str):
                    continue
                score = float(result.get("score", 0.0) or 0.0)
                if url not in score_by_url or score > score_by_url[url]:
                    score_by_url[url] = score

            ordered_scraped = sorted(
                scraped_content.items(),
                key=lambda item: score_by_url.get(item[0], 0.0),
                reverse=True,
            )
            append_with_budget("\n# Detailed Content from Top Sources\n")
            for url, content in ordered_scraped[:10]:
                domain = web_utils.extract_domain(url)
                excerpt = content[:2200]
                append_with_budget(
                    f"\n## Content from {domain}\nURL: {url}\n{excerpt}\n"
                )

        full_names = self._extract_canonical_full_names(search_results, scraped_content)
        if full_names:
            append_with_budget("\n# Canonical Name Hints\n")
            for name in full_names[:5]:
                append_with_budget(f"- {name}\n")

        if notes:
            append_with_budget("\n# Working Research Notes\n")
            if notes.highlights:
                append_with_budget("## Highlights\n")
                for item in notes.highlights[:8]:
                    append_with_budget(f"- {item}\n")
            if notes.resolved_points:
                append_with_budget("\n## Resolved Points\n")
                for item in notes.resolved_points[:6]:
                    append_with_budget(f"- {item}\n")
            if notes.unresolved_points:
                append_with_budget("\n## Unresolved Points\n")
                for item in notes.unresolved_points[:6]:
                    append_with_budget(f"- {item}\n")
            if notes.follow_up_queries:
                append_with_budget("\n## Follow-up Queries Already Planned/Run\n")
                for item in notes.follow_up_queries[:6]:
                    append_with_budget(f"- {item}\n")

        return "".join(context_parts), full_names

    @staticmethod
    def _build_synthesis_instruction(query: str) -> str:
        lowered = query.lower()
        strict_markers = (
            "only provide",
            "do not list any other information",
            "do not provide any other information",
            "return only",
        )
        if any(marker in lowered for marker in strict_markers):
            if "bulleted list" in lowered:
                return render_prompt(
                    "orchestrator.synthesis_instruction_strict_bulleted",
                    query=query,
                )
            return render_prompt(
                "orchestrator.synthesis_instruction_strict_default",
                query=query,
            )

        return render_prompt(
            "orchestrator.synthesis_instruction_default",
            query=query,
        )

    def _bookend_query_context(self, query: str, context: str) -> str:
        query_text = (query or "").strip()
        context_text = (context or "").strip()
        if not self.prompt_bookend_enabled or not query_text or not context_text:
            return context

        query_block = f"<query>\n{query_text}\n</query>"
        context_block = f"<context>\n{context_text}\n</context>"

        pattern = self.prompt_bookend_pattern
        if pattern == "context_query_query":
            return f"{context_block}\n\n{query_block}\n\n{query_block}"
        if pattern == "query_context":
            return f"{query_block}\n\n{context_block}"
        if pattern == "context_query":
            return f"{context_block}\n\n{query_block}"

        # Default and recommended layout: <QUERY><CONTEXT><QUERY>
        return f"{query_block}\n\n{context_block}\n\n{query_block}"

    def _build_synthesis_messages(self, query: str, full_context: str) -> list[dict[str, Any]]:
        if self.prompt_bookend_enabled:
            wrapped = self._bookend_query_context(query, full_context)
            instruction = self._build_synthesis_instruction(query)
            return [{"role": "user", "content": f"{wrapped}\n\n{instruction}"}]
        return [
            {"role": "user", "content": f"<context>\n{full_context}\n</context>"},
            {
                "role": "assistant",
                "content": render_prompt("orchestrator.synthesis_context_ack"),
            },
            {
                "role": "user",
                "content": self._build_synthesis_instruction(query),
            },
        ]

    @staticmethod
    def _build_source_list(
        search_results: list[dict[str, Any]],
        *,
        extra_urls: list[str] | None = None,
    ) -> list[dict[str, str]]:
        sources: list[dict[str, str]] = []
        seen_urls: set[str] = set()
        for result in search_results:
            url = str(result.get("url", "") or "")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            sources.append(
                {
                    "title": str(result.get("title", "") or ""),
                    "url": url,
                    "domain": web_utils.extract_domain(url),
                }
            )

        for url in extra_urls or []:
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            title = url.rstrip("/").split("/")[-1] or web_utils.extract_domain(url)
            sources.append(
                {
                    "title": title,
                    "url": url,
                    "domain": web_utils.extract_domain(url),
                }
            )

        return sources

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
            max_tokens=2600,
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
            system=render_prompt("orchestrator.synthesis_review_system"),
            messages=[
                {
                    "role": "user",
                    "content": self._bookend_query_context(
                        query,
                        render_prompt(
                            "orchestrator.synthesis_review_user",
                            query=query,
                            working_notes_json=json.dumps(notes_payload),
                            draft_report=draft_report[:7000],
                        ),
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
        pipeline_started_at: float | None = None,
    ) -> AsyncGenerator[SSEEvent, None]:
        """Feed all collected data to analyzer for synthesis, streaming the response."""
        full_context, full_names = self._build_synthesis_context(
            query,
            search_results,
            scraped_content,
            notes,
        )

        logger.info(f"Starting synthesis stage with {len(search_results)} search results and {len(scraped_content)} scraped pages")
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

        sources = self._build_source_list(search_results)

        full_report = self._promote_full_name_mentions(full_report, full_names)
        full_report = self._strip_deferred_sections(full_report)
        runtime_ms: int | None = None
        if pipeline_started_at is not None:
            runtime_ms = int((time.monotonic() - pipeline_started_at) * 1000)

        logger.info(f"Research complete! Runtime: {runtime_ms}ms, Tokens: {tokens_used}, Sources: {len(sources)}")
        yield streaming.research_complete(
            report=full_report,
            sources=sources[:20],
            tokens_used=tokens_used,
            runtime_ms=runtime_ms,
        )

    # V1 legacy path removed

    async def _revise_plan(
        self,
        query: str,
        current_plan: ResearchPlan,
        revision_context: dict[str, Any],
    ) -> ResearchPlan:
        """
        Revise the research plan mid-research based on accumulated evidence.

        Called when:
        - Search results are poor quality
        - Evidence contradicts initial assumptions
        - New information suggests different approach needed

        Args:
            query: Original research query
            current_plan: The current research plan
            revision_context: Context including evidence gathered, failed steps, etc.

        Returns:
            Updated ResearchPlan with revised steps
        """
        # Build context for the revision prompt
        evidence_summary = revision_context.get("evidence_summary", "")
        failed_steps = revision_context.get("failed_steps", [])
        low_quality_steps = revision_context.get("low_quality_steps", [])

        revision_prompt = f"""Original query: {query}

Current plan has {len(current_plan.steps)} steps.
Steps that need revision: {low_quality_steps + failed_steps}

Evidence gathered so far:
{evidence_summary}

The current plan is not working well. Please revise the plan by:
1. Identifying which steps are not producing useful results
2. Creating new, more effective search queries
3. Adding steps to address gaps in the evidence

Respond with ONLY valid JSON in this format:
{{
  "revised_steps": [
    {{"id": "step-1", "query": "...", "purpose": "...", "dependencies": [], "revised_from": "old-step-id"}},
    ...
  ],
  "reason": "brief explanation of why the plan was revised"
}}"""

        active_client = self.client or llm_client()
        t0 = time.monotonic()

        try:
            response = await active_client.messages.create(
                model=self.planner_model,
                max_tokens=2048,
                system=render_prompt(
                    "orchestrator.plan_system",
                    today_iso=date.today().isoformat(),
                    current_year=date.today().year,
                ),
                messages=[{"role": "user", "content": revision_prompt}],
            )
            await self._log_call(
                "orchestrator.plan_revision",
                response,
                int((time.monotonic() - t0) * 1000),
                model_name=self.planner_model,
            )

            # Parse response
            blocks = getattr(response, "content", None) or []
            text_parts = []
            for block in blocks:
                btype = getattr(block, "type", None)
                btext = getattr(block, "text", None)
                is_text_like_type = btype in (None, "text") or not isinstance(btype, str)
                if is_text_like_type and isinstance(btext, str) and btext.strip():
                    text_parts.append(btext)

            if not text_parts:
                return current_plan

            text = "\n".join(text_parts).strip()

            # Extract JSON from response
            try:
                # Handle markdown code blocks
                if text.startswith("```"):
                    parts = text.split("```")
                    if len(parts) >= 2:
                        text = parts[1]
                    if text.startswith("json"):
                        text = text[4:]
                    text = text.strip()

                parsed = json.loads(text)
                revised_steps_data = parsed.get("revised_steps", [])
                revision_reason = parsed.get("reason", "Plan revised based on evidence")

                # Update current plan with revised steps
                now = datetime.now(timezone.utc).isoformat()

                # Mark old steps as revised
                revised_step_ids = {step["id"] for step in revised_steps_data if "revised_from" in step}

                for step in current_plan.steps:
                    if step.id in revised_step_ids:
                        step.status = StepStatus.REVISED

                # Add new revised steps
                new_steps = []
                for step_data in revised_steps_data:
                    revised_from = step_data.get("revised_from")
                    new_step = ResearchStep(
                        id=step_data.get("id", f"step-{len(current_plan.steps) + len(new_steps) + 1}"),
                        query=step_data.get("query", ""),
                        purpose=step_data.get("purpose", ""),
                        dependencies=step_data.get("dependencies", []),
                        status=StepStatus.PENDING,
                        revised_from=revised_from,
                        revision_reason=revision_reason,
                    )
                    new_steps.append(new_step)

                # Create new plan version
                return ResearchPlan(
                    id=current_plan.id,  # Keep same plan ID
                    original_query=current_plan.original_query,
                    steps=current_plan.steps + new_steps,
                    version=current_plan.version + 1,
                    created_at=current_plan.created_at,
                    updated_at=now,
                )

            except (json.JSONDecodeError, KeyError, IndexError):
                return current_plan

        except Exception:
            return current_plan

    async def _execute_research(self, query: str) -> AsyncGenerator[SSEEvent, None]:
        """Hybrid execution: staged backbone with bounded parallel mesh internals."""
        logger.info(f"Starting research execution for query: {query[:100]}...")
        try:
            pipeline_started_at = time.monotonic()

            # Generate research plan
            logger.info("Generating research plan...")
            research_plan = await self._generate_plan(query)
            logger.info(f"Research plan generated with {len(research_plan.steps)} steps")
            logger.debug(f"Research plan: {research_plan.model_dump_json()}")
            yield streaming.plan_created(research_plan)

            # Extract steps as list for methods that expect list[str]
            plan_step_queries = [step.query for step in research_plan.steps]

            execution_graph = compile_execution_graph(query, plan_step_queries)
            graph_meta = {
                **graph_summary(execution_graph),
                "mode": "hybrid",
                "shadow_mode": self.shadow_mode,
            }
            logger.info(f"Execution graph compiled: {graph_meta}")
            yield streaming.execution_compiled(graph_meta)

            # Stage: search
            logger.info("Starting search stage...")
            search_stage_started = time.monotonic()
            yield streaming.mesh_stage_started(
                "search",
                node_count=graph_meta.get("search_nodes", len(plan_step_queries)),
                max_parallel=self.max_parallel_search,
            )
            search_results, search_events = await self._run_parallel_searches_bounded(
                plan_step_queries,
                step_offset=0,
                max_parallel=self.max_parallel_search,
            )
            search_results = self._sanitize_search_results(search_results)
            for event in search_events:
                yield event
            if not search_results:
                yield streaming.error("No search results found. Try refining your query.")
                return

            bootstrap_queries = self._build_chain_bootstrap_queries(query, search_results)
            bootstrap_query_count = 0
            bootstrap_results: list[dict[str, Any]] = []
            if bootstrap_queries:
                bootstrap_results, bootstrap_events = await self._run_parallel_searches_bounded(
                    bootstrap_queries,
                    step_offset=len(plan_step_queries),
                    max_parallel=self.max_parallel_search,
                )
                bootstrap_query_count = len(bootstrap_queries)
                for event in bootstrap_events:
                    yield event
                if bootstrap_results:
                    search_results = self._sanitize_search_results(
                        self._merge_search_results(search_results, bootstrap_results)
                    )

            logger.info(f"Search stage completed: {len(search_results)} results from {len(plan_step_queries)} queries")
            yield streaming.mesh_stage_completed(
                "search",
                results_count=len(search_results),
                query_count=len(plan_step_queries),
                duration_ms=int((time.monotonic() - search_stage_started) * 1000),
            )
            searched_query_keys = {
                " ".join(step.lower().split()) for step in plan_step_queries if step.strip()
            }
            for bootstrap_query in bootstrap_queries:
                searched_query_keys.add(" ".join(bootstrap_query.lower().split()))
            search_step_offset = len(plan_step_queries) + bootstrap_query_count

            # Stage: extraction
            logger.info(f"Starting extraction stage with {len(search_results)} search results")
            top_urls = await self._select_top_urls(search_results, query=query)
            logger.info(f"Selected {len(top_urls)} top URLs for extraction")
            if bootstrap_results:
                bootstrap_top_urls = await self._select_top_urls(
                    bootstrap_results,
                    max_urls=4,
                    query=query,
                )
                merged_top_urls: list[str] = []
                for url in bootstrap_top_urls + top_urls:
                    if url in merged_top_urls:
                        continue
                    merged_top_urls.append(url)
                    if len(merged_top_urls) >= max(len(top_urls), 8):
                        break
                top_urls = merged_top_urls
            extract_stage_started = time.monotonic()
            yield streaming.mesh_stage_started(
                "extract",
                url_count=len(top_urls),
                max_parallel=self.max_parallel_extract,
            )
            scraped_content: dict[str, str] = {}
            if top_urls:
                scraped_content, scrape_events = await self._run_parallel_scrapes_bounded(
                    top_urls,
                    max_parallel=self.max_parallel_extract,
                )
                for event in scrape_events:
                    yield event
            successful_scrapes = sum(1 for content in scraped_content.values() if content)
            yield streaming.mesh_stage_completed(
                "extract",
                attempted=len(top_urls),
                successful=successful_scrapes,
                url_count=len(top_urls),
                duration_ms=int((time.monotonic() - extract_stage_started) * 1000),
            )

            enrichment_queries = self._build_chain_enrichment_queries(
                query,
                search_results,
                scraped_content,
            )
            enrichment_queries = [
                q for q in enrichment_queries if " ".join(q.lower().split()) not in searched_query_keys
            ]
            if enrichment_queries:
                for q in enrichment_queries:
                    searched_query_keys.add(" ".join(q.lower().split()))
                enrichment_results, enrichment_events = await self._run_parallel_searches_bounded(
                    enrichment_queries,
                    step_offset=search_step_offset,
                    max_parallel=self.max_parallel_search,
                )
                search_step_offset += len(enrichment_queries)
                for event in enrichment_events:
                    yield event
                if enrichment_results:
                    search_results = self._sanitize_search_results(
                        self._merge_search_results(search_results, enrichment_results)
                    )
                    enrichment_urls = await self._select_top_urls(
                        enrichment_results,
                        max_urls=4,
                        query=query,
                    )
                    urls_to_scrape = [url for url in enrichment_urls if url not in scraped_content]
                    if urls_to_scrape:
                        extra_scraped, extra_scrape_events = await self._run_parallel_scrapes_bounded(
                            urls_to_scrape,
                            max_parallel=self.max_parallel_extract,
                        )
                        for event in extra_scrape_events:
                            yield event
                        scraped_content.update(extra_scraped)

            # Memory upsert summary
            memory_payload, memory_events = await self._upsert_memory_chunks(
                scraped_content=scraped_content
            )
            for event in memory_events:
                yield event

            # Stage: verification
            verify_stage_started = time.monotonic()
            yield streaming.mesh_stage_started(
                "verify",
                max_parallel=self.max_parallel_verify,
            )
            verification_task = asyncio.create_task(
                self._run_verification_stage(
                    query=query,
                    plan_steps=plan_step_queries,
                )
            )
            notes_task = asyncio.create_task(
                self._capture_research_notes(query, search_results, scraped_content)
            )

            verification_outcome, notes_outcome = await asyncio.gather(
                verification_task,
                notes_task,
                return_exceptions=True,
            )
            if isinstance(verification_outcome, Exception):
                verification_results = []
                verification_events = [
                    streaming.error(
                        f"Verification stage failed: {verification_outcome}",
                        agent="verification",
                    )
                ]
            else:
                verification_results, verification_events = verification_outcome

            if isinstance(notes_outcome, Exception):
                notes = self._fallback_notes(query, search_results)
            else:
                notes = notes_outcome

            for event in verification_events:
                yield event
            verification_payload = self._verification_summary(verification_results)
            yield streaming.mesh_stage_completed(
                "verify",
                **verification_payload,
                duration_ms=int((time.monotonic() - verify_stage_started) * 1000),
            )

            # Continue with deterministic notes/follow-up/review pipeline.
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

                follow_up_results, follow_up_events = await self._run_parallel_searches_bounded(
                    follow_up_queries,
                    step_offset=search_step_offset,
                    max_parallel=self.max_parallel_search,
                )
                search_step_offset += len(follow_up_queries)
                for event in follow_up_events:
                    yield event

                if follow_up_results:
                    search_results = self._sanitize_search_results(
                        self._merge_search_results(search_results, follow_up_results)
                    )
                    new_urls = await self._select_top_urls(
                        follow_up_results,
                        max_urls=4,
                        query=query,
                    )
                    urls_to_scrape = [url for url in new_urls if url not in scraped_content]
                    if urls_to_scrape:
                        extra_scraped, extra_scrape_events = await self._run_parallel_scrapes_bounded(
                            urls_to_scrape,
                            max_parallel=self.max_parallel_extract,
                        )
                        for event in extra_scrape_events:
                            yield event
                        scraped_content.update(extra_scraped)

                        extra_memory_payload, extra_memory_events = await self._upsert_memory_chunks(
                            scraped_content=extra_scraped,
                            default_step_id=f"extract_follow_up_{round_index}",
                        )
                        memory_payload["inserted_chunks"] += extra_memory_payload["inserted_chunks"]
                        memory_payload["deduplicated_chunks"] += extra_memory_payload["deduplicated_chunks"]
                        memory_payload["documents_processed"] += extra_memory_payload["documents_processed"]
                        for event in extra_memory_events:
                            yield event

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

            if self._should_run_synthesis_review(
                notes=notes,
                search_results=search_results,
                verification_results=verification_results,
            ):
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
                    review = await self._review_report_sufficiency(query, draft_report, notes)
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

                    post_results, post_events = await self._run_parallel_searches_bounded(
                        post_queries,
                        step_offset=search_step_offset,
                        max_parallel=self.max_parallel_search,
                    )
                    search_step_offset += len(post_queries)
                    for event in post_events:
                        yield event

                    if post_results:
                        search_results = self._sanitize_search_results(
                            self._merge_search_results(search_results, post_results)
                        )
                        post_urls = await self._select_top_urls(
                            post_results,
                            max_urls=4,
                            query=query,
                        )
                        urls_to_scrape = [url for url in post_urls if url not in scraped_content]
                        if urls_to_scrape:
                            extra_scraped, extra_scrape_events = await self._run_parallel_scrapes_bounded(
                                urls_to_scrape,
                                max_parallel=self.max_parallel_extract,
                            )
                            for event in extra_scrape_events:
                                yield event
                            scraped_content.update(extra_scraped)

                            extra_memory_payload, extra_memory_events = await self._upsert_memory_chunks(
                                scraped_content=extra_scraped,
                                default_step_id=f"extract_review_{review_round}",
                            )
                            memory_payload["inserted_chunks"] += extra_memory_payload["inserted_chunks"]
                            memory_payload["deduplicated_chunks"] += extra_memory_payload["deduplicated_chunks"]
                            memory_payload["documents_processed"] += extra_memory_payload["documents_processed"]
                            for event in extra_memory_events:
                                yield event

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

            if not self.shadow_mode and verification_results:
                notes.highlights.append(
                    "Verification summary: "
                    f"{verification_payload.get('supported', 0)} supported, "
                    f"{verification_payload.get('partially_supported', 0)} partially supported, "
                    f"{verification_payload.get('unsupported', 0)} unsupported."
                )
                notes.highlights = self._normalize_text_list(notes.highlights, max_items=12, min_len=3)

            async for event in self._synthesize(
                query,
                search_results,
                scraped_content,
                notes,
                pipeline_started_at=pipeline_started_at,
            ):
                yield event
        except Exception as e:
            logger.exception(f"Research failed with error: {e}")
            yield streaming.error(f"Hybrid research failed: {e}")

    async def research(self, query: str) -> AsyncGenerator[SSEEvent, None]:
        async for event in self._execute_research(query):
            yield event
