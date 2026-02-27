"""SecondOrder - Deep Research Tool

Multi-agent research system using Mini-Agent architecture.
"""

import argparse
import asyncio
import sys
from pathlib import Path

from app.agents.parallel_orchestrator import ParallelAgentOrchestrator


async def run_research(
    query: str, model: str | None = None, mcp_config: str | None = None
):
    print(f"Research query: {query}")
    print("-" * 50)

    orchestrator = ParallelAgentOrchestrator(
        model=model,
        mcp_config_path=mcp_config,
    )

    async for event in orchestrator.run(query):
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


def main():
    parser = argparse.ArgumentParser(description="SecondOrder Deep Research Tool")
    parser.add_argument("--query", "-q", required=True, help="Research query")
    parser.add_argument("--model", "-m", help="Model to use (default: from config)")
    parser.add_argument(
        "--mcp-config",
        help="Path to MCP config file (default: mcp.json in current dir)",
    )

    args = parser.parse_args()

    asyncio.run(run_research(args.query, args.model, args.mcp_config))


if __name__ == "__main__":
    main()
