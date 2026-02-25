from __future__ import annotations

import pytest

from app.agents.verification_agent import VerificationAgent
from app.models.memory import MemoryHit


class _FakeMemoryStore:
    def __init__(self, hits_by_query: dict[str, list[MemoryHit]]):
        self.hits_by_query = hits_by_query

    async def query(self, query):
        return self.hits_by_query.get(query.text, [])


@pytest.mark.asyncio
async def test_verification_agent_scores_supported_and_partial():
    agent = VerificationAgent()
    tasks = agent.build_tasks("name the countries", ["claim one", "claim two"], max_tasks=2)

    store = _FakeMemoryStore(
        {
            tasks[0].claim: [
                MemoryHit(id="h1", text="x", score=0.9, metadata={"url": "https://a"}),
                MemoryHit(id="h2", text="y", score=0.8, metadata={"url": "https://b"}),
            ],
            tasks[1].claim: [
                MemoryHit(id="h3", text="z", score=0.6, metadata={"url": "https://c"}),
            ],
        }
    )

    results = await agent.verify_tasks(
        query="Name all countries that match",
        tasks=tasks,
        session_id="s1",
        memory_store=store,  # type: ignore[arg-type]
        max_parallel=2,
    )

    assert results[0].status == "supported"
    assert results[1].status == "partially_supported"
    assert results[0].citations
