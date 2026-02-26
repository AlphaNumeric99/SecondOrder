"""PostgreSQL database service using asyncpg."""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID

import asyncpg

from app.config import settings


# Connection pool
_pool: asyncpg.Pool | None = None


def _db_available() -> bool:
    """Check if database is configured and available."""
    return bool(settings.database_url)


async def _get_pool() -> asyncpg.Pool:
    """Get or create the database connection pool."""
    global _pool
    if not _db_available():
        raise RuntimeError("Database not configured. Set DATABASE_URL in .env")
    if _pool is None:
        _pool = await asyncpg.create_pool(
            settings.database_url,
            min_size=1,
            max_size=10,
        )
    return _pool


async def close_pool() -> None:
    """Close the database connection pool."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


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
    """Create a new session."""
    pool = await _get_pool()
    async with pool.acquire() as conn:
        result = await conn.fetchrow(
            """
            INSERT INTO sessions (model, title)
            VALUES ($1, $2)
            RETURNING id, model, title, created_at, updated_at
            """,
            model,
            title,
        )
        return dict(result)


async def get_sessions() -> list[dict[str, Any]]:
    """Get all sessions ordered by creation date."""
    pool = await _get_pool()
    async with pool.acquire() as conn:
        results = await conn.fetch(
            """
            SELECT id, model, title, created_at, updated_at
            FROM sessions
            ORDER BY created_at DESC
            """
        )
        return [dict(r) for r in results]


async def get_session(session_id: UUID) -> dict[str, Any] | None:
    """Get a session by ID."""
    pool = await _get_pool()
    async with pool.acquire() as conn:
        result = await conn.fetchrow(
            """
            SELECT id, model, title, created_at, updated_at
            FROM sessions
            WHERE id = $1
            """,
            session_id,
        )
        return dict(result) if result else None


async def update_session(session_id: UUID, **kwargs: Any) -> dict[str, Any]:
    """Update a session."""
    pool = await _get_pool()

    # Build dynamic update query
    allowed = {"title", "model"}
    updates = {k: v for k, v in kwargs.items() if k in allowed}
    if not updates:
        return await get_session(session_id)

    set_clause = ", ".join(f"{k} = ${i+2}" for i, k in enumerate(updates.keys()))
    values = list(updates.values()) + [session_id]

    async with pool.acquire() as conn:
        result = await conn.fetchrow(
            f"""
            UPDATE sessions
            SET {set_clause}, updated_at = now()
            WHERE id = ${len(values)}
            RETURNING id, model, title, created_at, updated_at
            """,
            *values,
        )
        return dict(result)


async def delete_session(session_id: UUID) -> None:
    """Delete a session and its related data."""
    pool = await _get_pool()
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM sessions WHERE id = $1", session_id)


# --- Messages ---

async def create_message(
    session_id: UUID, role: str, content: str, metadata: dict | None = None
) -> dict[str, Any]:
    """Create a new message."""
    pool = await _get_pool()
    async with pool.acquire() as conn:
        result = await conn.fetchrow(
            """
            INSERT INTO messages (session_id, role, content, metadata)
            VALUES ($1, $2, $3, $4)
            RETURNING id, session_id, role, content, metadata, created_at
            """,
            session_id,
            role,
            content,
            json.dumps(metadata or {}),
        )
        row = dict(result)
        row["metadata"] = _coerce_json_object(row.get("metadata"))
        return row


async def get_messages(session_id: UUID) -> list[dict[str, Any]]:
    """Get all messages for a session."""
    pool = await _get_pool()
    async with pool.acquire() as conn:
        results = await conn.fetch(
            """
            SELECT id, session_id, role, content, metadata, created_at
            FROM messages
            WHERE session_id = $1
            ORDER BY created_at
            """,
            session_id,
        )
        rows = []
        for r in results:
            row = dict(r)
            row["metadata"] = _coerce_json_object(row.get("metadata"))
            rows.append(row)
        return rows


# --- Research Steps ---

async def create_research_step(
    session_id: UUID, step_type: str, data: dict | None = None
) -> dict[str, Any]:
    """Create a new research step."""
    pool = await _get_pool()
    async with pool.acquire() as conn:
        result = await conn.fetchrow(
            """
            INSERT INTO research_steps (session_id, step_type, data, status)
            VALUES ($1, $2, $3, 'pending')
            RETURNING id, session_id, step_type, status, data, created_at
            """,
            session_id,
            step_type,
            json.dumps(data or {}),
        )
        row = dict(result)
        row["data"] = _coerce_json_object(row.get("data"))
        return row


async def update_research_step(step_id: UUID, status: str, data: dict | None = None) -> None:
    """Update a research step."""
    pool = await _get_pool()
    async with pool.acquire() as conn:
        if data is not None:
            await conn.execute(
                """
                UPDATE research_steps
                SET status = $1, data = $2
                WHERE id = $3
                """,
                status,
                json.dumps(data),
                step_id,
            )
        else:
            await conn.execute(
                """
                UPDATE research_steps
                SET status = $1
                WHERE id = $2
                """,
                status,
                step_id,
            )


async def get_research_steps(session_id: UUID) -> list[dict[str, Any]]:
    """Get all research steps for a session."""
    pool = await _get_pool()
    async with pool.acquire() as conn:
        results = await conn.fetch(
            """
            SELECT id, session_id, step_type, status, data, created_at
            FROM research_steps
            WHERE session_id = $1
            ORDER BY created_at
            """,
            session_id,
        )
        rows = []
        for r in results:
            row = dict(r)
            row["data"] = _coerce_json_object(row.get("data"))
            rows.append(row)
        return rows


async def save_research_plan(session_id: UUID, plan: dict) -> dict[str, Any]:
    """Save or update the research plan for a session."""
    pool = await _get_pool()
    async with pool.acquire() as conn:
        # Check if plan already exists
        existing = await conn.fetchrow(
            """
            SELECT id FROM research_steps
            WHERE session_id = $1 AND step_type = 'plan'
            """,
            session_id,
        )
        if existing:
            # Update existing plan
            result = await conn.fetchrow(
                """
                UPDATE research_steps
                SET data = $1, status = 'completed'
                WHERE session_id = $2 AND step_type = 'plan'
                RETURNING id, session_id, step_type, status, data, created_at
                """,
                json.dumps(plan),
                session_id,
            )
        else:
            # Create new plan
            result = await conn.fetchrow(
                """
                INSERT INTO research_steps (session_id, step_type, data, status)
                VALUES ($1, 'plan', $2, 'completed')
                RETURNING id, session_id, step_type, status, data, created_at
                """,
                session_id,
                json.dumps(plan),
            )
        row = dict(result)
        row["data"] = _coerce_json_object(row.get("data"))
        return row


async def get_research_plan(session_id: UUID) -> dict | None:
    """Get the research plan for a session."""
    pool = await _get_pool()
    async with pool.acquire() as conn:
        result = await conn.fetchrow(
            """
            SELECT id, session_id, step_type, status, data, created_at
            FROM research_steps
            WHERE session_id = $1 AND step_type = 'plan'
            """,
            session_id,
        )
        if not result:
            return None
        row = dict(result)
        row["data"] = _coerce_json_object(row.get("data"))
        return row


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
    """Log an LLM call."""
    pool = await _get_pool()
    async with pool.acquire() as conn:
        result = await conn.fetchrow(
            """
            INSERT INTO llm_calls (model, caller, input_tokens, output_tokens, total_tokens, duration_ms, session_id, status, error, metadata)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            RETURNING id, model, caller, input_tokens, output_tokens, total_tokens, duration_ms, session_id, status, error, metadata, created_at
            """,
            model,
            caller,
            input_tokens,
            output_tokens,
            input_tokens + output_tokens,
            duration_ms,
            session_id,
            status,
            error,
            json.dumps(metadata or {}),
        )
        row = dict(result)
        row["metadata"] = _coerce_json_object(row.get("metadata"))
        return row


# For backwards compatibility - create a simple alias
class DatabaseService:
    """Database service class for compatibility."""
    pass
