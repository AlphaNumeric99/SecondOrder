from __future__ import annotations

import asyncio
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


async def _execute(query: Any) -> Any:
    """Run blocking Supabase query execution in a worker thread."""
    return await asyncio.to_thread(query.execute)


def _coerce_json_object(value: Any) -> dict[str, Any]:
    """Normalize legacy JSON-string fields into dictionaries."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


# --- Sessions ---


async def create_session(model: str, title: str | None = None) -> dict[str, Any]:
    data = {"model": model}
    if title:
        data["title"] = title
    result = await _execute(client().table("sessions").insert(data))
    return result.data[0]


async def get_sessions() -> list[dict[str, Any]]:
    result = await _execute(client().table("sessions").select("*").order("created_at", desc=True))
    return result.data or []


async def get_session(session_id: UUID) -> dict[str, Any] | None:
    result = await _execute(client().table("sessions").select("*").eq("id", str(session_id)))
    return result.data[0] if result.data else None


async def update_session(session_id: UUID, **kwargs: Any) -> dict[str, Any]:
    result = await _execute(client().table("sessions").update(kwargs).eq("id", str(session_id)))
    return result.data[0]


async def delete_session(session_id: UUID) -> None:
    await _execute(client().table("sessions").delete().eq("id", str(session_id)))


# --- Messages ---


async def create_message(
    session_id: UUID, role: str, content: str, metadata: dict | None = None
) -> dict[str, Any]:
    data = {
        "session_id": str(session_id),
        "role": role,
        "content": content,
        "metadata": metadata or {},
    }
    result = await _execute(client().table("messages").insert(data))
    return result.data[0]


async def get_messages(session_id: UUID) -> list[dict[str, Any]]:
    result = await _execute(
        client()
        .table("messages")
        .select("*")
        .eq("session_id", str(session_id))
        .order("created_at")
    )
    rows = result.data or []
    for row in rows:
        row["metadata"] = _coerce_json_object(row.get("metadata"))
    return rows


# --- Research Steps ---


async def create_research_step(
    session_id: UUID, step_type: str, data: dict | None = None
) -> dict[str, Any]:
    row = {
        "session_id": str(session_id),
        "step_type": step_type,
        "data": data or {},
    }
    result = await _execute(client().table("research_steps").insert(row))
    return result.data[0]


async def update_research_step(step_id: UUID, status: str, data: dict | None = None) -> None:
    update = {"status": status}
    if data is not None:
        update["data"] = data
    await _execute(client().table("research_steps").update(update).eq("id", str(step_id)))


async def get_research_steps(session_id: UUID) -> list[dict[str, Any]]:
    result = await _execute(
        client()
        .table("research_steps")
        .select("*")
        .eq("session_id", str(session_id))
        .order("created_at")
    )
    rows = result.data or []
    for row in rows:
        row["data"] = _coerce_json_object(row.get("data"))
    return rows


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
        "metadata": metadata or {},
    }
    if session_id:
        row["session_id"] = str(session_id)
    if duration_ms is not None:
        row["duration_ms"] = duration_ms
    if error:
        row["error"] = error
    result = await _execute(client().table("llm_calls").insert(row))
    return result.data[0]
