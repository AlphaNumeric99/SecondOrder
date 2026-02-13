"""DRACO Benchmark — Perplexity's Deep Research Accuracy, Completeness, and Objectivity.

100 tasks across 10 domains, each with expert-crafted rubrics (~40 criteria).
Dataset: https://huggingface.co/datasets/perplexity-ai/draco
"""
from __future__ import annotations

import json
from typing import Any

from app.config import settings
from app.llm_client import get_client
from benchmarks.base import Benchmark, BenchmarkTask, EvalResult

JUDGE_SYSTEM = """You are an expert research evaluator. You will be given:
1. A research QUERY
2. A RUBRIC with weighted criteria organized into sections
3. A RESPONSE (research report) to evaluate

For each criterion in the rubric, determine if the response SATISFIES it.
Score each criterion as 1 (satisfied) or 0 (not satisfied).

Respond with a JSON object:
{
  "criteria_scores": {
    "<criterion_id>": {"satisfied": true/false, "reasoning": "brief explanation"}
  },
  "total_weighted_score": <float>,
  "max_possible_score": <float>,
  "normalized_score": <float between 0 and 1>
}

Be strict but fair. A criterion is satisfied only if the response clearly addresses it."""


class DRACOBenchmark(Benchmark):
    name = "DRACO"

    async def load_tasks(self, limit: int | None = None) -> list[BenchmarkTask]:
        """Load DRACO tasks from HuggingFace dataset (cached locally)."""
        data_path = "benchmarks/data/draco.json"

        try:
            with open(data_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except FileNotFoundError:
            print("  Downloading DRACO dataset from HuggingFace...")
            raw = await self._download_dataset()
            with open(data_path, "w", encoding="utf-8") as f:
                json.dump(raw, f)

        tasks = []
        for item in raw:
            tasks.append(BenchmarkTask(
                id=item["id"],
                query=item["problem"],
                domain=item["domain"],
                reference=item["answer"],  # JSON rubric
            ))

        if limit:
            tasks = tasks[:limit]
        return tasks

    async def _download_dataset(self) -> list[dict[str, Any]]:
        """Download DRACO from HuggingFace API."""
        import httpx

        url = "https://datasets-server.huggingface.co/rows?dataset=perplexity-ai%2Fdraco&config=default&split=test&offset=0&length=100"
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()

        return [row["row"] for row in data["rows"]]

    async def evaluate(self, task: BenchmarkTask, response: str) -> EvalResult:
        """Evaluate response against DRACO rubric using LLM-as-judge."""
        client = get_client()

        # Parse rubric from reference
        try:
            rubric = json.loads(task.reference) if isinstance(task.reference, str) else task.reference
        except json.JSONDecodeError:
            rubric = {"raw": task.reference}

        rubric_text = json.dumps(rubric, indent=2) if isinstance(rubric, dict) else str(rubric)

        judge_response = await client.messages.create(
            model=settings.benchmark_judge_model,
            max_tokens=4096,
            system=JUDGE_SYSTEM,
            messages=[{
                "role": "user",
                "content": (
                    f"## QUERY\n{task.query}\n\n"
                    f"## RUBRIC\n{rubric_text[:6000]}\n\n"
                    f"## RESPONSE\n{response[:8000]}"
                ),
            }],
        )

        judge_text = judge_response.content[0].text

        # Parse judge output
        try:
            # Extract JSON from response
            json_start = judge_text.find("{")
            json_end = judge_text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                scores = json.loads(judge_text[json_start:json_end])
                normalized = scores.get("normalized_score", 0.0)
            else:
                normalized = 0.0
                scores = {"raw": judge_text}
        except (json.JSONDecodeError, KeyError):
            normalized = 0.0
            scores = {"raw": judge_text}

        return EvalResult(
            task_id=task.id,
            score=float(normalized),
            details={
                "domain": task.domain,
                "weighted_score": scores.get("total_weighted_score", 0),
                "max_score": scores.get("max_possible_score", 0),
                "criteria_count": len(scores.get("criteria_scores", {})),
            },
        )
