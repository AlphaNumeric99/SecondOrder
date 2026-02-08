"""ResearchRubrics Benchmark — Scale AI's deep research evaluation.

2,500+ expert-written rubrics across 9 domains.
Dataset: https://huggingface.co/datasets/ScaleAI/researchrubrics
GitHub: https://github.com/scaleapi/researchrubrics
"""
from __future__ import annotations

import json
from typing import Any

import anthropic

from app.config import settings
from benchmarks.base import Benchmark, BenchmarkTask, EvalResult

JUDGE_SYSTEM = """You are an expert research evaluator using the ResearchRubrics methodology.

You will be given:
1. A research QUERY
2. A list of RUBRIC CRITERIA (each with a description and weight)
3. A RESPONSE (research report) to evaluate

For each rubric criterion, assess if the response satisfies it:
- "satisfied" (1.0) — the response clearly meets this criterion
- "not_satisfied" (0.0) — the response does not meet this criterion

Calculate the compliance score: sum(weight * score) / sum(positive_weights)

Respond with JSON:
{
  "criteria_results": [
    {"id": <index>, "criterion": "brief description", "satisfied": true/false, "weight": <number>, "reasoning": "brief explanation"}
  ],
  "compliance_score": <float between 0 and 1>,
  "strengths": ["list of strengths"],
  "weaknesses": ["list of weaknesses"]
}

Be rigorous. Factual accuracy and reasoning quality are paramount."""


class ResearchRubricsBenchmark(Benchmark):
    name = "ResearchRubrics"

    async def load_tasks(self, limit: int | None = None) -> list[BenchmarkTask]:
        """Load ResearchRubrics tasks from local cache or HuggingFace."""
        data_path = "benchmarks/data/researchrubrics.jsonl"

        try:
            tasks = []
            with open(data_path) as f:
                for line in f:
                    if line.strip():
                        tasks.append(json.loads(line))
        except FileNotFoundError:
            print("  Downloading ResearchRubrics dataset from HuggingFace...")
            tasks = await self._download_dataset()
            with open(data_path, "w") as f:
                for item in tasks:
                    f.write(json.dumps(item) + "\n")

        benchmark_tasks = []
        for item in tasks:
            benchmark_tasks.append(BenchmarkTask(
                id=item.get("sample_id", item.get("id", "")),
                query=item["prompt"],
                domain=item.get("domain", "unknown"),
                reference=json.dumps(item.get("rubrics", [])),
                metadata={
                    "conceptual_breadth": item.get("conceptual_breadth"),
                    "logical_nesting": item.get("logical_nesting"),
                    "exploration": item.get("exploration"),
                },
            ))

        if limit:
            benchmark_tasks = benchmark_tasks[:limit]
        return benchmark_tasks

    async def _download_dataset(self) -> list[dict[str, Any]]:
        """Download ResearchRubrics from HuggingFace (requires auth for gated dataset)."""
        import httpx
        import os

        hf_token = os.getenv("HF_TOKEN", "")
        headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}

        url = "https://datasets-server.huggingface.co/rows?dataset=ScaleAI%2Fresearchrubrics&config=default&split=train&offset=0&length=100"
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.get(url, headers=headers)
            if resp.status_code == 401:
                raise RuntimeError(
                    "ResearchRubrics is a gated dataset. To download:\n"
                    "  1. Accept access at https://huggingface.co/datasets/ScaleAI/researchrubrics\n"
                    "  2. Set HF_TOKEN in backend/.env\n"
                    "  OR manually download processed_data.jsonl to benchmarks/data/researchrubrics.jsonl"
                )
            resp.raise_for_status()
            data = resp.json()

        return [row["row"] for row in data["rows"]]

    async def evaluate(self, task: BenchmarkTask, response: str) -> EvalResult:
        """Evaluate response against ResearchRubrics criteria."""
        client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

        rubrics = json.loads(task.reference) if isinstance(task.reference, str) else task.reference
        rubric_text = json.dumps(rubrics, indent=2)[:6000]

        judge_response = await client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=4096,
            system=JUDGE_SYSTEM,
            messages=[{
                "role": "user",
                "content": (
                    f"## QUERY\n{task.query}\n\n"
                    f"## RUBRIC CRITERIA\n{rubric_text}\n\n"
                    f"## RESPONSE\n{response[:8000]}"
                ),
            }],
        )

        judge_text = judge_response.content[0].text

        try:
            json_start = judge_text.find("{")
            json_end = judge_text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                scores = json.loads(judge_text[json_start:json_end])
                compliance = scores.get("compliance_score", 0.0)
            else:
                compliance = 0.0
                scores = {"raw": judge_text}
        except (json.JSONDecodeError, KeyError):
            compliance = 0.0
            scores = {"raw": judge_text}

        return EvalResult(
            task_id=task.id,
            score=float(compliance),
            details={
                "domain": task.domain,
                "strengths": scores.get("strengths", []),
                "weaknesses": scores.get("weaknesses", []),
                "criteria_evaluated": len(scores.get("criteria_results", [])),
            },
        )
