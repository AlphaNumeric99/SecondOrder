"""SecondOrder - Deep Research Tool

Multi-agent research system using Mini-Agent architecture.
"""

import argparse
import asyncio
import re
import sys
from pathlib import Path

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def strip_emoji(text: str) -> str:
    emoji_pattern = re.compile(
        "["
        "\U0001f600-\U0001f64f"  # emoticons
        "\U0001f300-\U0001f5ff"  # symbols & pictographs
        "\U0001f680-\U0001f6ff"  # transport & map symbols
        "\U0001f1e0-\U0001f1ff"  # flags
        "\U00002702-\U000027b0"  # dingbats
        "\U000024c2-\U0001f251"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(r"", text)


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
        event_type = event.event.value if hasattr(event.event, "value") else event.event
        data = event.data

        if event_type == "agent_started":
            print(f"\n[~] Agent started: {data.get('query', '')[:60]}...")

        elif event_type == "agent_completed":
            print(f"  [+] Agent completed")

        elif event_type == "message":
            msg = data.get("content", "")
            msg = strip_emoji(msg)
            if data.get("type") == "final":
                print(f"\n\n{'=' * 50}")
                print("FINAL ANSWER:")
                print(f"{'=' * 50}")
                print(msg)
            else:
                print(f"\n[*] {msg}")

        elif event_type == "error":
            err_msg = strip_emoji(data.get("message", "Unknown error"))
            print(f"\n[!] Error: {err_msg}")


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
