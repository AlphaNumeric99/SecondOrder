"""CLI runner for SecondOrder benchmarks.

Usage:
  python -m benchmarks.run draco --limit 10 --model claude-sonnet-4-5-20250929
  python -m benchmarks.run researchrubrics --limit 5
  python -m benchmarks.run deepsearchqa --limit 20
  python -m benchmarks.run all --limit 5
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Ensure app imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from benchmarks.draco import DRACOBenchmark
from benchmarks.research_rubrics import ResearchRubricsBenchmark
from benchmarks.deepsearchqa import DeepSearchQABenchmark

BENCHMARKS = {
    "draco": DRACOBenchmark,
    "researchrubrics": ResearchRubricsBenchmark,
    "deepsearchqa": DeepSearchQABenchmark,
}


async def main():
    parser = argparse.ArgumentParser(description="Run SecondOrder benchmarks")
    parser.add_argument(
        "benchmark",
        choices=list(BENCHMARKS.keys()) + ["all"],
        help="Which benchmark to run",
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=None,
        help="Max number of tasks to run (default: all)",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="claude-sonnet-4-5-20250929",
        help="Model to use for research",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="benchmarks/results",
        help="Output directory for results",
    )

    args = parser.parse_args()

    if args.benchmark == "all":
        benchmarks_to_run = list(BENCHMARKS.values())
    else:
        benchmarks_to_run = [BENCHMARKS[args.benchmark]]

    for BenchmarkClass in benchmarks_to_run:
        bench = BenchmarkClass()
        await bench.run(
            model=args.model,
            limit=args.limit,
            output_dir=args.output,
        )


if __name__ == "__main__":
    asyncio.run(main())
