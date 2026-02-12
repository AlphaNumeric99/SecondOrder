"""Base benchmark framework for evaluating SecondOrder's research quality."""
from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class BenchmarkTask:
    id: str
    query: str
    domain: str
    reference: str  # Gold answer or rubric JSON
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    task_id: str
    score: float  # 0.0 - 1.0 normalized
    details: dict[str, Any] = field(default_factory=dict)
    response: str = ""
    elapsed_seconds: float = 0.0


@dataclass
class BenchmarkReport:
    benchmark_name: str
    model: str
    timestamp: str
    total_tasks: int
    completed_tasks: int
    avg_score: float
    scores_by_domain: dict[str, float]
    results: list[EvalResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark_name,
            "model": self.model,
            "timestamp": self.timestamp,
            "total_tasks": self.total_tasks,
            "completed_tasks": self.completed_tasks,
            "avg_score": round(self.avg_score, 4),
            "scores_by_domain": {k: round(v, 4) for k, v in self.scores_by_domain.items()},
            "results": [
                {
                    "task_id": r.task_id,
                    "score": round(r.score, 4),
                    "elapsed_seconds": round(r.elapsed_seconds, 2),
                    "details": r.details,
                }
                for r in self.results
            ],
        }

    def save(self, output_dir: str | Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{self.benchmark_name}_{self.model}_{self.timestamp}.json"
        path = output_dir / filename
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    def print_summary(self) -> None:
        print(f"\n{'='*60}")
        print(f"  {self.benchmark_name} Benchmark Results")
        print(f"{'='*60}")
        print(f"  Model:     {self.model}")
        print(f"  Tasks:     {self.completed_tasks}/{self.total_tasks}")
        print(f"  Avg Score: {self.avg_score:.1%}")
        print(f"\n  Scores by Domain:")
        for domain, score in sorted(self.scores_by_domain.items(), key=lambda x: -x[1]):
            print(f"    {domain:30s} {score:.1%}")
        print(f"{'='*60}\n")


class Benchmark(ABC):
    """Base class for all benchmarks."""

    name: str = "base"

    @abstractmethod
    async def load_tasks(self, limit: int | None = None) -> list[BenchmarkTask]:
        """Load benchmark tasks from dataset."""
        ...

    @abstractmethod
    async def evaluate(self, task: BenchmarkTask, response: str) -> EvalResult:
        """Evaluate a single response against the task's reference/rubric."""
        ...

    async def run(
        self,
        model: str = "openai/gpt-4o-mini",
        limit: int | None = None,
        output_dir: str = "benchmarks/results",
    ) -> BenchmarkReport:
        """Run the full benchmark: load tasks, generate responses, evaluate."""
        from app.agents.orchestrator import ResearchOrchestrator

        tasks = await self.load_tasks(limit=limit)
        results: list[EvalResult] = []

        print(f"\nRunning {self.name} benchmark ({len(tasks)} tasks, model={model})")

        for i, task in enumerate(tasks):
            print(f"  [{i+1}/{len(tasks)}] {task.domain}: {task.query[:80]}...")

            # Run research
            start = time.time()
            orchestrator = ResearchOrchestrator(model=model)
            report_text = ""

            try:
                async for event in orchestrator.research(task.query):
                    if event.event.value == "research_complete":
                        report_text = event.data.get("report", "")
                    elif event.event.value == "synthesis_progress":
                        report_text += event.data.get("chunk", "")
            except Exception as e:
                print(f"    ERROR: {e}")
                results.append(EvalResult(
                    task_id=task.id, score=0.0,
                    details={"error": str(e)}, elapsed_seconds=time.time() - start,
                ))
                continue

            elapsed = time.time() - start

            # Evaluate
            result = await self.evaluate(task, report_text)
            result.elapsed_seconds = elapsed
            result.response = report_text[:500]  # Truncate for storage
            results.append(result)
            print(f"    Score: {result.score:.1%} ({elapsed:.1f}s)")

        # Aggregate
        scores_by_domain: dict[str, list[float]] = {}
        for task, result in zip(tasks[: len(results)], results):
            scores_by_domain.setdefault(task.domain, []).append(result.score)

        avg_domain_scores = {d: sum(s) / len(s) for d, s in scores_by_domain.items()}
        avg_score = sum(r.score for r in results) / len(results) if results else 0.0

        report = BenchmarkReport(
            benchmark_name=self.name,
            model=model,
            timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
            total_tasks=len(tasks),
            completed_tasks=len(results),
            avg_score=avg_score,
            scores_by_domain=avg_domain_scores,
            results=results,
        )

        path = report.save(output_dir)
        report.print_summary()
        print(f"  Results saved to: {path}")
        return report
