from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.agents.parallel_orchestrator import ParallelAgentOrchestrator
from app.llm_client import client as llm_client


QUERY_1 = (
    "Name the two songs that went viral on TikTok by the artist who (from 2016 to 2021) "
    'played drums in a band that won "Best Holy Local Band" in the Phoenix New Times Best of Phoenix 2006.'
)


def _safe_slug(value: str, max_len: int = 80) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("._")
    if not slug:
        slug = "item"
    return slug[:max_len]


def _now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _json_dump(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


async def run_research(query: str, model: str | None = None):
    print(f"Research query: {query}")
    print("-" * 50)

    orchestrator = ParallelAgentOrchestrator(model=model)

    events = []
    async for event in orchestrator.run(query):
        events.append(event)
        event_type = event.event.value
        data = event.data

        if event_type == "agent_started":
            print(f"\n[~] Agent started: {data.get('query', '')[:60]}...")

        elif event_type == "agent_completed":
            print(f"  [+] Agent completed")

        elif event_type == "message":
            msg = data.get("content", "")
            if data.get("type") == "final":
                print(f"\n\n{'=' * 50}")
                print("FINAL ANSWER:")
                print(f"{'=' * 50}")
                print(msg)
            else:
                print(f"\n[*] {msg}")

        elif event_type == "error":
            print(f"\n[!] Error: {data.get('message', 'Unknown error')}")


async def main():
    parser = argparse.ArgumentParser(description="SecondOrder Research CLI")
    parser.add_argument("--query", "-q", help="Research query")
    parser.add_argument("--model", "-m", help="Model to use")
    parser.add_argument(
        "--mcp-config",
        help="Path to MCP config file",
    )

    args = parser.parse_args()

    query = args.query or QUERY_1
    model = args.model
    mcp_config = args.mcp_config

    await run_research(query, model)


if __name__ == "__main__":
    asyncio.run(main())
