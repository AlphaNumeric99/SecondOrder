from __future__ import annotations

import asyncio
import json
import time
from datetime import date
from typing import Any, AsyncGenerator
from uuid import UUID

from app.agents.analyzer_agent import AnalyzerAgent
from app.agents.search_agent import SearchAgent
from app.agents.scraper_agent import ScraperAgent
from app.config import settings
from app.llm_client import client as llm_client, get_model
from app.models.events import SSEEvent
from app.services import streaming
from app.services import supabase as db
from app.tools import web_utils


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
                "When the query asks about 'current' or 'latest' data, make sure sub-queries "
                f"include the current year ({date.today().year}) to find the most recent information.\n\n"
                "Respond with ONLY a JSON array of strings, no other text. Example:\n"
                '["sub-query 1", "sub-query 2", "sub-query 3"]'
            ),
            messages=[{"role": "user", "content": query}],
        )
        await self._log_call("orchestrator.plan", response, int((time.monotonic() - t0) * 1000))

        text = response.content[0].text.strip()
        # Extract JSON array from response
        try:
            # Handle case where model wraps in markdown code block
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()
            return json.loads(text)
        except (json.JSONDecodeError, IndexError):
            # Fallback: use the original query as a single plan step
            return [query]

    async def _run_parallel_searches(
        self, plan_steps: list[str]
    ) -> tuple[list[dict[str, Any]], list[SSEEvent]]:
        """Run search agents in parallel for all plan steps."""
        all_events: list[SSEEvent] = []
        all_results: list[dict[str, Any]] = []

        agents = [SearchAgent(model=self.model, step_index=i, session_id=self.session_id) for i in range(len(plan_steps))]

        # Start events
        for i, step in enumerate(plan_steps):
            all_events.append(streaming.agent_started("search", step=i, query=step))

        # Run all searches in parallel
        async def run_search(agent: SearchAgent, query: str) -> list[SSEEvent]:
            events: list[SSEEvent] = []
            async for event in agent.run(
                f"Search for: {query}",
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
            all_events.append(streaming.agent_completed("search", step=i))

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
        return [url for url, _ in sorted_urls[:max_urls]]

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
        all_events.append(streaming.agent_completed("scraper", urls_scraped=len(all_content)))

        return all_content, all_events

    async def _synthesize(
        self, query: str, search_results: list[dict], scraped_content: dict[str, str]
    ) -> AsyncGenerator[SSEEvent, None]:
        """Feed all collected data to analyzer for synthesis, streaming the response."""
        # Build context for the analyzer
        today = date.today()
        context_parts = [
            f"# Current Date\n{today.isoformat()} ({today.year})\n",
            f"# Original Research Query\n{query}\n",
        ]

        context_parts.append("# Search Results Summary")
        for i, r in enumerate(search_results[:20]):  # Cap to prevent context bloat
            context_parts.append(
                f"\n## Source {i+1}: [{r.get('title', 'Untitled')}]({r.get('url', '')})\n"
                f"{r.get('content', '')[:500]}"
            )

        if scraped_content:
            context_parts.append("\n# Detailed Content from Top Sources")
            for url, content in list(scraped_content.items())[:8]:
                domain = web_utils.extract_domain(url)
                context_parts.append(
                    f"\n## Content from {domain}\nURL: {url}\n{content[:3000]}"
                )

        full_context = "\n".join(context_parts)

        yield streaming.synthesis_started(len(search_results))

        # Use streaming for the synthesis to get incremental output
        active_client = self.client or llm_client()
        t0 = time.monotonic()
        async with active_client.messages.stream(
            model=self.model,
            max_tokens=8192,
            system=AnalyzerAgent.get_system_prompt(),
            messages=[
                {"role": "user", "content": f"<context>\n{full_context}\n</context>"},
                {
                    "role": "assistant",
                    "content": "I've reviewed all the research materials. Let me now provide my analysis.",
                },
                {
                    "role": "user",
                    "content": (
                        f"Based on all the research data provided, write a comprehensive "
                        f"research report on: {query}\n\n"
                        f"Include citations to sources using [Title](URL) format."
                    ),
                },
            ],
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

        yield streaming.research_complete(
            report=full_report,
            sources=sources[:20],
            tokens_used=getattr(stream, "usage", {}).get("total_tokens", 0) if hasattr(stream, "usage") else 0,
        )

    async def research(self, query: str) -> AsyncGenerator[SSEEvent, None]:
        """Execute the full research pipeline, yielding SSE events throughout."""
        try:
            # Step 1: Generate research plan
            plan_steps = await self._generate_plan(query)
            yield streaming.plan_created(plan_steps)

            # Step 2: Parallel searches
            search_results, search_events = await self._run_parallel_searches(plan_steps)
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

            # Step 5: Synthesis — streamed
            async for event in self._synthesize(query, search_results, scraped_content):
                yield event

        except Exception as e:
            yield streaming.error(f"Research failed: {e}")
