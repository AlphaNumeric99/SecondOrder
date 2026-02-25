from __future__ import annotations

from typing import Protocol

from app.config import settings
from app.models.memory import MemoryHit, MemoryQuery, MemoryRecord, MemoryUpsertResult
from app.services.memory_chroma import ChromaMemoryStore


class MemoryStore(Protocol):
    async def upsert(self, records: list[MemoryRecord]) -> MemoryUpsertResult: ...
    async def query(self, query: MemoryQuery) -> list[MemoryHit]: ...
    async def delete_session(self, session_id: str) -> None: ...
    async def cleanup_expired_sessions(self, ttl_hours: int) -> list[str]: ...


_store: MemoryStore | None = None


def get_memory_store() -> MemoryStore:
    global _store
    if _store is None:
        backend = settings.memory_backend.lower().strip()
        if backend != "chromadb":
            raise ValueError(f"Unsupported MEMORY_BACKEND: {settings.memory_backend}")
        _store = ChromaMemoryStore(
            persist_dir=settings.chroma_persist_dir,
            ttl_hours=int(settings.chroma_session_ttl_hours),
        )
    return _store
