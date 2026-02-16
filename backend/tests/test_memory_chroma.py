from __future__ import annotations

import pytest

from app.models.memory import MemoryQuery, MemoryRecord
from app.services.memory_chroma import ChromaMemoryStore


class _FakeEmbedder:
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            length = float(len(text) or 1)
            vectors.append([length, 1.0])
        return vectors

    async def embed_text(self, text: str) -> list[float]:
        length = float(len(text) or 1)
        return [length, 1.0]


def _record(record_id: str, text: str, *, session_id: str = "s1") -> MemoryRecord:
    return MemoryRecord(
        id=record_id,
        session_id=session_id,
        step_id="extract",
        url="https://example.com",
        text=text,
        chunk_hash=record_id,
        created_at="2026-02-16T00:00:00+00:00",
    )


async def _build_store(tmp_path) -> ChromaMemoryStore:
    store = ChromaMemoryStore(str(tmp_path / "chroma"), ttl_hours=168)
    store._embedder = _FakeEmbedder()  # type: ignore[attr-defined]
    # Force fallback mode so tests do not require chromadb runtime.
    store._client = None
    return store

@pytest.mark.asyncio
async def test_memory_upsert_and_query_returns_ranked_hits(tmp_path):
    store = await _build_store(tmp_path)
    await store.upsert([_record("r1", "short"), _record("r2", "longer evidence text")])

    hits = await store.query(MemoryQuery(session_id="s1", text="evidence", top_k=2))

    assert len(hits) == 2
    assert hits[0].score >= hits[1].score


@pytest.mark.asyncio
async def test_memory_upsert_deduplicates_by_id(tmp_path):
    store = await _build_store(tmp_path)

    first = await store.upsert([_record("same", "v1")])
    second = await store.upsert([_record("same", "v2")])

    assert first.inserted == 1
    assert second.deduplicated == 1


@pytest.mark.asyncio
async def test_memory_query_applies_metadata_filters(tmp_path):
    store = await _build_store(tmp_path)
    await store.upsert(
        [
            _record("r1", "alpha text"),
            MemoryRecord(
                id="r2",
                session_id="s1",
                step_id="extract_follow_up",
                url="https://example.com/2",
                text="beta text",
                chunk_hash="r2",
                created_at="2026-02-16T00:00:00+00:00",
            ),
        ]
    )

    hits = await store.query(
        MemoryQuery(
            session_id="s1",
            text="text",
            top_k=5,
            filters={"step_id": "extract_follow_up"},
        )
    )

    assert len(hits) == 1
    assert hits[0].metadata["step_id"] == "extract_follow_up"
