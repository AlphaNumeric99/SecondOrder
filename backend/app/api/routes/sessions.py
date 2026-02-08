from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, HTTPException

from app.models.schemas import MessageResponse, SessionDetailResponse, SessionResponse
from app.services import supabase as db

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.get("", response_model=list[SessionResponse])
async def list_sessions():
    """List all research sessions, most recent first."""
    sessions = await db.get_sessions()
    return sessions


@router.get("/{session_id}", response_model=SessionDetailResponse)
async def get_session(session_id: UUID):
    """Get a session with all its messages."""
    session = await db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = await db.get_messages(session_id)

    return SessionDetailResponse(
        session=SessionResponse(**session),
        messages=[MessageResponse(**m) for m in messages],
    )


@router.delete("/{session_id}")
async def delete_session(session_id: UUID):
    """Delete a research session and all its data."""
    session = await db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    await db.delete_session(session_id)
    return {"status": "deleted"}
