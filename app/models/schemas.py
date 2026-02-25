from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel


# --- Requests ---


class ResearchRequest(BaseModel):
    query: str
    model: str | None = None


# --- Responses ---


class SessionResponse(BaseModel):
    id: UUID
    title: str | None
    model: str
    created_at: datetime
    updated_at: datetime


class MessageResponse(BaseModel):
    id: UUID
    session_id: UUID
    role: str
    content: str
    metadata: dict
    created_at: datetime


class SessionDetailResponse(BaseModel):
    session: SessionResponse
    messages: list[MessageResponse]
    research: dict[str, Any] | None = None


class ResearchStartResponse(BaseModel):
    session_id: UUID


class ModelInfo(BaseModel):
    id: str
    name: str
    description: str


class ModelsResponse(BaseModel):
    models: list[ModelInfo]
