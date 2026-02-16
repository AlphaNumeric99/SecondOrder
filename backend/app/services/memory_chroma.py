from __future__ import annotations

import asyncio
import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from app.models.memory import MemoryHit, MemoryQuery, MemoryRecord, MemoryUpsertResult
from app.services.embeddings_local import LocalEmbeddingService


class ChromaMemoryStore:
    def __init__(self, persist_dir: str, ttl_hours: int):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_hours = ttl_hours
        self._embedder = LocalEmbeddingService()
        self._client: Any | None = None
        self._fallback_records: dict[str, dict[str, tuple[MemoryRecord, list[float]]]] = {}
        self._registry_path = self.persist_dir / "_session_registry.json"
        self._registry_lock = asyncio.Lock()
        self._client_lock = asyncio.Lock()

    async def upsert(self, records: list[MemoryRecord]) -> MemoryUpsertResult:
        if not records:
            return MemoryUpsertResult()
        session_id = records[0].session_id
        vectors = await self._embedder.embed_texts([record.text for record in records])
        await self._touch_session(session_id)

        client = await self._get_client()
        if client is None:
            return self._fallback_upsert(records, vectors)

        def _sync_upsert() -> None:
            collection = client.get_or_create_collection(
                name=_collection_name(session_id),
                metadata={"session_id": session_id},
            )
            ids = [record.id for record in records]
            docs = [record.text for record in records]
            metas = [_metadata_for_record(record) for record in records]
            collection.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=vectors)

        await asyncio.to_thread(_sync_upsert)
        return MemoryUpsertResult(inserted=len(records), deduplicated=0)

    async def query(self, query: MemoryQuery) -> list[MemoryHit]:
        vector = await self._embedder.embed_text(query.text)
        client = await self._get_client()
        if client is None:
            return self._fallback_query(query, vector)

        def _sync_query() -> list[MemoryHit]:
            collection = client.get_or_create_collection(
                name=_collection_name(query.session_id),
                metadata={"session_id": query.session_id},
            )
            include = ["documents", "metadatas", "distances"]
            kwargs: dict[str, Any] = {
                "query_embeddings": [vector],
                "n_results": max(int(query.top_k), 1),
                "include": include,
            }
            if query.filters:
                kwargs["where"] = query.filters
            result = collection.query(**kwargs)

            docs = (result.get("documents") or [[]])[0]
            metas = (result.get("metadatas") or [[]])[0]
            distances = (result.get("distances") or [[]])[0]
            ids = (result.get("ids") or [[]])[0]
            hits: list[MemoryHit] = []
            for idx, doc in enumerate(docs):
                if not isinstance(doc, str):
                    continue
                metadata = metas[idx] if idx < len(metas) and isinstance(metas[idx], dict) else {}
                distance = float(distances[idx]) if idx < len(distances) else 1.0
                score = 1.0 / (1.0 + max(distance, 0.0))
                hit_id = ids[idx] if idx < len(ids) and isinstance(ids[idx], str) else f"hit_{idx}"
                hits.append(MemoryHit(id=hit_id, text=doc, score=score, metadata=metadata))
            return hits

        return await asyncio.to_thread(_sync_query)

    async def delete_session(self, session_id: str) -> None:
        await self._remove_session_from_registry(session_id)
        client = await self._get_client()
        if client is None:
            self._fallback_records.pop(session_id, None)
            return

        def _sync_delete() -> None:
            client.delete_collection(_collection_name(session_id))

        try:
            await asyncio.to_thread(_sync_delete)
        except Exception:
            return

    async def cleanup_expired_sessions(self, ttl_hours: int) -> list[str]:
        registry = await self._load_registry()
        now = datetime.now(UTC)
        max_age = timedelta(hours=max(int(ttl_hours), 1))
        expired = [
            session_id
            for session_id, updated_at in registry.items()
            if _parse_iso(updated_at) is not None and now - _parse_iso(updated_at) > max_age
        ]
        for session_id in expired:
            await self.delete_session(session_id)
        return expired

    async def _get_client(self) -> Any | None:
        async with self._client_lock:
            if self._client is not None:
                return self._client
            try:
                import chromadb
            except Exception:
                self._client = None
                return None
            self._client = chromadb.PersistentClient(path=str(self.persist_dir))
            return self._client

    async def _load_registry(self) -> dict[str, str]:
        async with self._registry_lock:
            if not self._registry_path.exists():
                return {}
            try:
                payload = json.loads(self._registry_path.read_text(encoding="utf-8"))
            except Exception:
                return {}
            if not isinstance(payload, dict):
                return {}
            return {k: v for k, v in payload.items() if isinstance(k, str) and isinstance(v, str)}

    async def _save_registry(self, payload: dict[str, str]) -> None:
        async with self._registry_lock:
            self._registry_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    async def _touch_session(self, session_id: str) -> None:
        payload = await self._load_registry()
        payload[session_id] = datetime.now(UTC).isoformat()
        await self._save_registry(payload)

    async def _remove_session_from_registry(self, session_id: str) -> None:
        payload = await self._load_registry()
        payload.pop(session_id, None)
        await self._save_registry(payload)

    def _fallback_upsert(
        self, records: list[MemoryRecord], vectors: list[list[float]]
    ) -> MemoryUpsertResult:
        inserted = 0
        deduplicated = 0
        for record, vector in zip(records, vectors):
            session_store = self._fallback_records.setdefault(record.session_id, {})
            if record.id in session_store:
                deduplicated += 1
            else:
                inserted += 1
            session_store[record.id] = (record, vector)
        return MemoryUpsertResult(inserted=inserted, deduplicated=deduplicated)

    def _fallback_query(self, query: MemoryQuery, vector: list[float]) -> list[MemoryHit]:
        session_store = self._fallback_records.get(query.session_id, {})
        hits: list[MemoryHit] = []
        for record_id, (record, record_vec) in session_store.items():
            if query.filters and not _metadata_matches(_metadata_for_record(record), query.filters):
                continue
            score = _cosine_similarity(vector, record_vec)
            hits.append(
                MemoryHit(
                    id=record_id,
                    text=record.text,
                    score=score,
                    metadata=_metadata_for_record(record),
                )
            )
        hits.sort(key=lambda hit: (-hit.score, hit.id))
        return hits[: max(int(query.top_k), 1)]


def _metadata_for_record(record: MemoryRecord) -> dict[str, Any]:
    metadata = {
        "session_id": record.session_id,
        "step_id": record.step_id,
        "url": record.url,
        "chunk_hash": record.chunk_hash,
        "created_at": record.created_at,
    }
    if record.provider:
        metadata["provider"] = record.provider
    metadata.update(record.metadata)
    return metadata


def _collection_name(session_id: str) -> str:
    return f"session_{session_id.replace('-', '_')}"


def _parse_iso(raw: str) -> datetime | None:
    try:
        return datetime.fromisoformat(raw)
    except Exception:
        return None


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _metadata_matches(metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
    for key, expected in filters.items():
        if metadata.get(key) != expected:
            return False
    return True
