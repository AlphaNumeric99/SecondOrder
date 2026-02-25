"""DeepSearchQA Benchmark — Google DeepMind's multi-step research evaluation.

900 tasks across 17 fields, testing exhaustive answer generation.
Dataset: https://huggingface.co/datasets/google/deepsearchqa
Evaluation: LLM-as-judge comparing response against gold answers.
"""
from __future__ import annotations

import json
from typing import Any

from app.config import settings
from app.llm_client import get_client
from benchmarks.base import Benchmark, BenchmarkTask, EvalResult

JUDGE_SYSTEM = """You are an expert evaluator for the DeepSearchQA benchmark.

You will be given:
1. A QUESTION requiring exhaustive research
2. A GOLD ANSWER (the reference answer set)
3. An ANSWER TYPE: "Single Answer" or "Set Answer"
4. A RESPONSE (the system's research output)

Your task is to extract the factual answers from the RESPONSE and compare them against the GOLD ANSWER.

For "Set Answer" tasks, compute:
- precision: fraction of extracted answers that are correct
- recall: fraction of gold answers that were found
- f1: 2 * (precision * recall) / (precision + recall)

For "Single Answer" tasks:
- Score 1.0 if the response contains the correct answer, 0.0 otherwise

Respond with JSON:
{
  "extracted_answers": ["list of answers found in response"],
  "gold_answers": ["list of gold answers"],
  "correct_matches": ["answers that match gold"],
  "precision": <float>,
  "recall": <float>,
  "f1": <float>,
  "reasoning": "brief explanation of matching decisions"
}

Be flexible with matching — accept equivalent phrasings, abbreviations, and minor variations.
But do NOT accept factually incorrect matches."""


class DeepSearchQABenchmark(Benchmark):
    name = "DeepSearchQA"

    async def load_tasks(self, limit: int | None = None) -> list[BenchmarkTask]:
        """Load DeepSearchQA tasks from local cache or HuggingFace."""
        data_path = "benchmarks/data/deepsearchqa.json"

        try:
            with open(data_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except FileNotFoundError:
            print("  Downloading DeepSearchQA dataset from HuggingFace...")
            raw = await self._download_dataset()
            with open(data_path, "w", encoding="utf-8") as f:
                json.dump(raw, f)

        tasks = []
        for i, item in enumerate(raw):
            tasks.append(BenchmarkTask(
                id=f"dsqa-{i}",
                query=item["problem"],
                domain=item.get("problem_category", "unknown"),
                reference=item["answer"],
                metadata={"answer_type": item.get("answer_type", "Set Answer")},
            ))

        if limit:
            tasks = tasks[:limit]
        return tasks

    async def _download_dataset(self) -> list[dict[str, Any]]:
        """Download DeepSearchQA from HuggingFace API."""
        import httpx

        all_rows: list[dict] = []
        offset = 0
        batch_size = 100

        async with httpx.AsyncClient(timeout=120.0) as client:
            while True:
                url = (
                    f"https://datasets-server.huggingface.co/rows?"
                    f"dataset=google%2Fdeepsearchqa&config=default&split=eval"
                    f"&offset={offset}&length={batch_size}"
                )
                resp = await client.get(url)
                resp.raise_for_status()
                data = resp.json()
                rows = [row["row"] for row in data["rows"]]
                if not rows:
                    break
                all_rows.extend(rows)
                offset += batch_size
                if len(rows) < batch_size:
                    break

        return all_rows

    async def evaluate(self, task: BenchmarkTask, response: str) -> EvalResult:
        """Evaluate response using LLM-as-judge for answer extraction and matching."""
        client = get_client()
        answer_type = task.metadata.get("answer_type", "Set Answer")

        judge_response = await client.messages.create(
            model=settings.benchmark_judge_model,
            max_tokens=2048,
            system=JUDGE_SYSTEM,
            messages=[{
                "role": "user",
                "content": (
                    f"## QUESTION\n{task.query}\n\n"
                    f"## GOLD ANSWER\n{task.reference}\n\n"
                    f"## ANSWER TYPE\n{answer_type}\n\n"
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
                f1 = scores.get("f1", 0.0)
            else:
                f1 = 0.0
                scores = {"raw": judge_text}
        except (json.JSONDecodeError, KeyError):
            f1 = 0.0
            scores = {"raw": judge_text}

        return EvalResult(
            task_id=task.id,
            score=float(f1),
            details={
                "domain": task.domain,
                "answer_type": answer_type,
                "precision": scores.get("precision", 0.0),
                "recall": scores.get("recall", 0.0),
                "f1": f1,
                "extracted_count": len(scores.get("extracted_answers", [])),
                "gold_count": len(scores.get("gold_answers", [])),
                "correct_count": len(scores.get("correct_matches", [])),
            },
        )
