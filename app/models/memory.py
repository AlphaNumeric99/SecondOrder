from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class MemoryRecord:
    id: str
    session_id: str
    step_id: str
    url: str
    text: str
    chunk_hash: str
    created_at: str
    provider: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MemoryQuery:
    session_id: str
    text: str
    top_k: int = 5
    filters: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MemoryHit:
    id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MemoryUpsertResult:
    inserted: int = 0
    deduplicated: int = 0
