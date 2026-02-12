from __future__ import annotations

import json
from typing import Any
from uuid import UUID

from supabase import create_client, Client

from app.config import settings
from app.services.env_safety import sanitize_ssl_keylogfile


def get_client() -> Client:
    sanitize_ssl_keylogfile()
    return create_client(settings.supabase_url, settings.supabase_anon_key)


_client: Client | None = None


def client() -> Client:
    global _client
    if _client is None:
        _client = get_client()
    return _client


# --- Sessions ---


async def create_session(model: str, title: str | None = None) -> dict[str, Any]:
    data = {"model": model}
    if title:
        data["title"] = title
    result = client().table("sessions").insert(data).execute()
    return result.data[0]


async def get_sessions() -> list[dict[str, Any]]:
    result = client().table("sessions").select("*").order("created_at", desc=True).execute()
    return result.data


async def get_session(session_id: UUID) -> dict[str, Any] | None:
    result = client().table("sessions").select("*").eq("id", str(session_id)).execute()
    return result.data[0] if result.data else None


async def update_session(session_id: UUID, **kwargs: Any) -> dict[str, Any]:
    result = client().table("sessions").update(kwargs).eq("id", str(session_id)).execute()
    return result.data[0]


async def delete_session(session_id: UUID) -> None:
    client().table("sessions").delete().eq("id", str(session_id)).execute()


# --- Messages ---


async def create_message(
    session_id: UUID, role: str, content: str, metadata: dict | None = None
) -> dict[str, Any]:
    data = {
        "session_id": str(session_id),
        "role": role,
        "content": content,
        "metadata": json.dumps(metadata or {}),
    }
    result = client().table("messages").insert(data).execute()
    return result.data[0]


async def get_messages(session_id: UUID) -> list[dict[str, Any]]:
    result = (
        client()
        .table("messages")
        .select("*")
        .eq("session_id", str(session_id))
        .order("created_at")
        .execute()
    )
    return result.data


# --- Research Steps ---


async def create_research_step(
    session_id: UUID, step_type: str, data: dict | None = None
) -> dict[str, Any]:
    row = {
        "session_id": str(session_id),
        "step_type": step_type,
        "data": json.dumps(data or {}),
    }
    result = client().table("research_steps").insert(row).execute()
    return result.data[0]


async def update_research_step(step_id: UUID, status: str, data: dict | None = None) -> None:
    update = {"status": status}
    if data is not None:
        update["data"] = json.dumps(data)
    client().table("research_steps").update(update).eq("id", str(step_id)).execute()


# --- LLM Calls ---


async def log_llm_call(
    model: str,
    caller: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    duration_ms: int | None = None,
    session_id: UUID | None = None,
    status: str = "success",
    error: str | None = None,
    metadata: dict | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "model": model,
        "caller": caller,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "status": status,
        "metadata": json.dumps(metadata or {}),
    }
    if session_id:
        row["session_id"] = str(session_id)
    if duration_ms is not None:
        row["duration_ms"] = duration_ms
    if error:
        row["error"] = error
    result = client().table("llm_calls").insert(row).execute()
    return result.data[0]
