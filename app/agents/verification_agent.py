from __future__ import annotations

import asyncio
from typing import Any

from app.models.execution import VerificationResult, VerificationTask
from app.models.memory import MemoryQuery
from app.services.memory_store import MemoryStore


class VerificationAgent:
    """Verify candidate claims against retrieved memory chunks."""

    name = "verification"

    @staticmethod
    def build_tasks(
        query: str,
        plan_steps: list[str],
        *,
        max_tasks: int = 10,
    ) -> list[VerificationTask]:
        tasks: list[VerificationTask] = []
        seen: set[str] = set()
        candidates = [query, *plan_steps]
        for idx, candidate in enumerate(candidates):
            claim = " ".join(candidate.split()).strip()
            if len(claim) < 6:
                continue
            key = claim.lower()
            if key in seen:
                continue
            seen.add(key)
            tasks.append(
                VerificationTask(
                    id=f"verify_task_{idx}",
                    claim=claim,
                    step_id=f"plan_{idx}" if idx > 0 else None,
                )
            )
            if len(tasks) >= max_tasks:
                break
        return tasks

    async def verify_tasks(
        self,
        *,
        query: str,
        tasks: list[VerificationTask],
        session_id: str,
        memory_store: MemoryStore,
        max_parallel: int,
    ) -> list[VerificationResult]:
        semaphore = asyncio.Semaphore(max(max_parallel, 1))
        is_list_query = _looks_like_list_query(query)

        async def run_task(task: VerificationTask) -> VerificationResult:
            async with semaphore:
                hits = await memory_store.query(
                    MemoryQuery(
                        session_id=session_id,
                        text=task.claim,
                        top_k=6,
                    )
                )
                return _score_task(task, hits, is_list_query=is_list_query)

        return await asyncio.gather(*(run_task(task) for task in tasks))


def _looks_like_list_query(query: str) -> bool:
    lowered = query.lower()
    markers = ("list", "name the", "only provide", "all ", "which ")
    return any(marker in lowered for marker in markers)


def _score_task(
    task: VerificationTask,
    hits: list[Any],
    *,
    is_list_query: bool,
) -> VerificationResult:
    high_hits = [hit for hit in hits if float(getattr(hit, "score", 0.0)) >= 0.72]
    medium_hits = [hit for hit in hits if float(getattr(hit, "score", 0.0)) >= 0.55]

    if is_list_query:
        supported = len(high_hits) >= 2
    else:
        supported = len(high_hits) >= 1

    if supported:
        status = "supported"
        reason = "Sufficient high-confidence evidence found in memory."
    elif medium_hits:
        status = "partially_supported"
        reason = "Some evidence found, but confidence/coverage is limited."
    else:
        status = "unsupported"
        reason = "No relevant evidence found in memory for this claim."

    citations: list[dict[str, Any]] = []
    for hit in hits[:3]:
        metadata = getattr(hit, "metadata", {}) or {}
        citations.append(
            {
                "chunk_id": getattr(hit, "id", ""),
                "score": round(float(getattr(hit, "score", 0.0)), 4),
                "url": metadata.get("url"),
                "step_id": metadata.get("step_id"),
            }
        )

    avg_score = 0.0
    if hits:
        avg_score = sum(float(getattr(hit, "score", 0.0)) for hit in hits[:3]) / min(
            len(hits), 3
        )

    return VerificationResult(
        task_id=task.id,
        status=status,
        score=avg_score,
        reason=reason,
        citations=citations,
    )
