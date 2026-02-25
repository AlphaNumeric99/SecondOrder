from __future__ import annotations

import argparse
import asyncio
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any
from urllib.parse import urlparse

from app.agents.orchestrator import ResearchOrchestrator
from app.llm_client import client as llm_client

QUERY_1 = (
    'Name the two songs that went viral on TikTok by the artist who (from 2016 to 2021) '
    'played drums in a band that won "Best Holy Local Band" in the Phoenix New Times Best of Phoenix 2006.'
)


def _safe_slug(value: str, max_len: int = 80) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("._")
    if not slug:
        slug = "item"
    return slug[:max_len]


def _now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _extract_response_text(response: Any) -> str:
    blocks = getattr(response, "content", None) or []
    text_parts: list[str] = []
    for block in blocks:
        btext = getattr(block, "text", None)
        if isinstance(btext, str) and btext.strip():
            text_parts.append(btext)
    return "\n".join(text_parts).strip()


def _json_dump(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


class TracingStream:
    def __init__(self, inner_stream: Any, call_record: dict[str, Any]):
        self._inner_stream = inner_stream
        self._record = call_record
        self._chunks: list[str] = []
        self._ctx: Any = None

    async def __aenter__(self) -> "TracingStream":
        self._ctx = await self._inner_stream.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._ctx is not None:
            await self._ctx.__aexit__(exc_type, exc, tb)

    @property
    def text_stream(self):
        async def _gen():
            if self._ctx is None:
                return
            async for chunk in self._ctx.text_stream:
                self._chunks.append(chunk)
                yield chunk

        return _gen()

    async def get_final_message(self):
        if self._ctx is None:
            raise RuntimeError("TracingStream not entered")
        final_message = await self._ctx.get_final_message()
        usage = getattr(final_message, "usage", None)
        self._record["output_text"] = "".join(self._chunks)
        self._record["usage"] = {
            "input_tokens": getattr(usage, "input_tokens", 0) if usage else 0,
            "output_tokens": getattr(usage, "output_tokens", 0) if usage else 0,
        }
        return final_message


class TracingMessages:
    def __init__(self, inner_messages: Any, call_log: list[dict[str, Any]]):
        self._inner_messages = inner_messages
        self._call_log = call_log

    async def create(
        self,
        *,
        model: str,
        max_tokens: int,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ):
        call_record: dict[str, Any] = {
            "type": "create",
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
            "tools": tools or [],
        }
        response = await self._inner_messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            tools=tools,
        )
        usage = getattr(response, "usage", None)
        call_record["output_text"] = _extract_response_text(response)
        call_record["usage"] = {
            "input_tokens": getattr(usage, "input_tokens", 0) if usage else 0,
            "output_tokens": getattr(usage, "output_tokens", 0) if usage else 0,
        }
        self._call_log.append(call_record)
        return response

    def stream(
        self,
        *,
        model: str,
        max_tokens: int,
        system: str,
        messages: list[dict[str, Any]],
    ):
        call_record: dict[str, Any] = {
            "type": "stream",
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
        }
        self._call_log.append(call_record)
        stream_ctx = self._inner_messages.stream(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        )
        return TracingStream(stream_ctx, call_record)


def _compact_event(event: Any) -> dict[str, Any]:
    return {
        "event": event.event.value,
        "data": event.data,
    }


def _extract_search_snapshot(events: list[dict[str, Any]]) -> dict[str, Any]:
    search_queries: list[dict[str, Any]] = []
    search_results: list[dict[str, Any]] = []
    for event in events:
        etype = event["event"]
        data = event.get("data", {}) or {}
        if etype == "agent_started" and data.get("agent") == "search":
            search_queries.append(
                {
                    "step": data.get("step"),
                    "query": data.get("query"),
                }
            )
        if etype == "search_result":
            search_results.append(data)
    return {
        "queries": search_queries,
        "results": search_results,
    }


async def run_trace(
    query: str,
    output_root: Path,
    run_name: str,
    model: str | None,
) -> Path:
    run_dir = output_root / f"{run_name}_{_now_tag()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "scraped_data").mkdir(parents=True, exist_ok=True)

    llm_calls: list[dict[str, Any]] = []
    search_batches: list[dict[str, Any]] = []
    scrape_batches: list[dict[str, Any]] = []
    scraped_content_all: dict[str, str] = {}

    base_client = llm_client()
    tracing_client = SimpleNamespace(messages=TracingMessages(base_client.messages, llm_calls))

    orchestrator = ResearchOrchestrator(model=model)
    orchestrator.client = tracing_client

    original_run_parallel_searches = orchestrator._run_parallel_searches
    original_run_parallel_searches_bounded = orchestrator._run_parallel_searches_bounded
    original_run_parallel_scrapes = orchestrator._run_parallel_scrapes
    original_run_parallel_scrapes_bounded = orchestrator._run_parallel_scrapes_bounded

    async def traced_run_parallel_searches(self, plan_steps: list[str], step_offset: int = 0):
        results, events = await original_run_parallel_searches(plan_steps, step_offset=step_offset)
        search_batches.append(
            {
                "method": "_run_parallel_searches",
                "step_offset": step_offset,
                "plan_steps": plan_steps,
                "results_count": len(results),
                "events": [_compact_event(event) for event in events],
            }
        )
        return results, events

    async def traced_run_parallel_searches_bounded(
        self,
        plan_steps: list[str],
        *,
        step_offset: int,
        max_parallel: int,
    ):
        results, events = await original_run_parallel_searches_bounded(
            plan_steps,
            step_offset=step_offset,
            max_parallel=max_parallel,
        )
        search_batches.append(
            {
                "method": "_run_parallel_searches_bounded",
                "step_offset": step_offset,
                "max_parallel": max_parallel,
                "plan_steps": plan_steps,
                "results_count": len(results),
                "events": [_compact_event(event) for event in events],
            }
        )
        return results, events

    async def traced_run_parallel_scrapes(self, urls: list[str]):
        scraped, events = await original_run_parallel_scrapes(urls)
        scraped_content_all.update(scraped)
        scrape_batches.append(
            {
                "method": "_run_parallel_scrapes",
                "urls": urls,
                "scraped_count": len(scraped),
                "events": [_compact_event(event) for event in events],
                "content_lengths": {url: len(content) for url, content in scraped.items()},
            }
        )
        return scraped, events

    async def traced_run_parallel_scrapes_bounded(self, urls: list[str], *, max_parallel: int):
        scraped, events = await original_run_parallel_scrapes_bounded(
            urls,
            max_parallel=max_parallel,
        )
        scraped_content_all.update(scraped)
        scrape_batches.append(
            {
                "method": "_run_parallel_scrapes_bounded",
                "max_parallel": max_parallel,
                "urls": urls,
                "scraped_count": len(scraped),
                "events": [_compact_event(event) for event in events],
                "content_lengths": {url: len(content) for url, content in scraped.items()},
            }
        )
        return scraped, events

    orchestrator._run_parallel_searches = MethodType(traced_run_parallel_searches, orchestrator)
    orchestrator._run_parallel_searches_bounded = MethodType(
        traced_run_parallel_searches_bounded,
        orchestrator,
    )
    orchestrator._run_parallel_scrapes = MethodType(traced_run_parallel_scrapes, orchestrator)
    orchestrator._run_parallel_scrapes_bounded = MethodType(
        traced_run_parallel_scrapes_bounded,
        orchestrator,
    )

    all_events: list[dict[str, Any]] = []
    async for event in orchestrator.research(query):
        all_events.append(_compact_event(event))

    plan_steps: list[str] = []
    final_report = ""
    final_sources: list[dict[str, Any]] = []
    for event in all_events:
        if event["event"] == "plan_created":
            plan_steps = list((event.get("data") or {}).get("steps") or [])
            break
    for event in reversed(all_events):
        if event["event"] == "research_complete":
            final_report = str((event.get("data") or {}).get("report") or "")
            final_sources = list((event.get("data") or {}).get("sources") or [])
            break

    scraped_index: list[dict[str, Any]] = []
    for idx, (url, content) in enumerate(scraped_content_all.items(), start=1):
        parsed = urlparse(url)
        slug = _safe_slug((parsed.netloc or "url") + "_" + (parsed.path or "root"))
        file_name = f"{idx:03d}_{slug}.txt"
        file_path = run_dir / "scraped_data" / file_name
        file_path.write_text(content, encoding="utf-8")
        scraped_index.append(
            {
                "url": url,
                "file": str(Path("scraped_data") / file_name),
                "length": len(content),
            }
        )

    search_snapshot = _extract_search_snapshot(all_events)

    (run_dir / "query.txt").write_text(query, encoding="utf-8")
    (run_dir / "final_report.md").write_text(final_report, encoding="utf-8")
    _json_dump(run_dir / "plan.json", {"steps": plan_steps})
    _json_dump(run_dir / "events.json", {"events": all_events})
    _json_dump(run_dir / "search_trace.json", search_snapshot)
    _json_dump(run_dir / "search_batches.json", {"batches": search_batches})
    _json_dump(run_dir / "scrape_batches.json", {"batches": scrape_batches})
    _json_dump(run_dir / "scraped_index.json", {"items": scraped_index})
    _json_dump(run_dir / "llm_calls.json", {"calls": llm_calls})
    _json_dump(run_dir / "sources.json", {"sources": final_sources})
    _json_dump(
        run_dir / "run_summary.json",
        {
            "query": query,
            "events_count": len(all_events),
            "plan_steps_count": len(plan_steps),
            "search_batches_count": len(search_batches),
            "scrape_batches_count": len(scrape_batches),
            "scraped_docs_count": len(scraped_index),
            "llm_calls_count": len(llm_calls),
            "final_sources_count": len(final_sources),
            "final_report_chars": len(final_report),
        },
    )

    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a full research trace and persist step-by-step artifacts.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=QUERY_1,
        help="Research query to run. Defaults to Query 1 from TESTS.md.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("trace_runs"),
        help="Root directory where trace folders are created.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="query_trace",
        help="Prefix for the run directory name.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional model override for the orchestrator runtime model.",
    )
    return parser.parse_args()


async def _main() -> None:
    args = parse_args()
    run_dir = await run_trace(
        query=args.query,
        output_root=args.output_root,
        run_name=args.run_name,
        model=args.model,
    )
    print(f"Trace saved to: {run_dir}")


if __name__ == "__main__":
    asyncio.run(_main())

