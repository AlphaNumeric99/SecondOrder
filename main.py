"""SecondOrder - Deep Research Tool

Simple CLI for running research queries.
"""

import argparse
import asyncio
import sys
from pathlib import Path

from app.agents.orchestrator import ResearchOrchestrator


async def run_research(query: str, model: str | None = None):
    """Run research on the given query."""
    print(f"Research query: {query}")
    print("-" * 50)

    orchestrator = ResearchOrchestrator(model=model)

    async for event in orchestrator.research(query):
        event_type = event.event.value
        data = event.data

        if event_type == "plan_created":
            steps = data.get("steps", [])
            print(f"\n[*] Research Plan ({len(steps)} steps):")
            for i, step in enumerate(steps, 1):
                print(f"  {i}. {step.get('query', '')[:80]}...")
                print(f"     Purpose: {step.get('purpose', 'N/A')}")

        elif event_type == "mesh_stage_started":
            print(f"\n[~] Starting {data.get('stage')} stage...")

        elif event_type == "mesh_stage_completed":
            print(f"  [+] {data.get('stage')} complete: {data.get('results_count')} results")

        elif event_type == "synthesis_started":
            print(f"\n[+] Synthesizing report...")

        elif event_type == "synthesis_progress":
            print(".", end="", flush=True)

        elif event_type == "research_complete":
            print(f"\n\n[*] Research Complete!")
            print(f"   Runtime: {data.get('runtime_ms')}ms")
            print(f"   Tokens: {data.get('tokens_used')}")
            print(f"   Sources: {len(data.get('sources', []))}")
            print(f"\n{'='*50}")
            print("REPORT:")
            print(f"{'='*50}")
            print(data.get("report", ""))

        elif event_type == "error":
            print(f"\n[!] Error: {data.get('message', 'Unknown error')}")


def main():
    parser = argparse.ArgumentParser(description="SecondOrder Deep Research Tool")
    parser.add_argument("--query", "-q", required=True, help="Research query")
    parser.add_argument("--model", "-m", help="Model to use (default: from config)")

    args = parser.parse_args()

    asyncio.run(run_research(args.query, args.model))


if __name__ == "__main__":
    main()
