from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from app.agents.orchestrator import ResearchOrchestrator
from app.models.schemas import ResearchRequest, ResearchStartResponse
from app.services import supabase as db

router = APIRouter(prefix="/api/research", tags=["research"])


@router.post("", response_model=ResearchStartResponse)
async def start_research(request: ResearchRequest):
    """Start a new research session. Returns session_id to use for streaming."""
    # Create session
    session = await db.create_session(
        model=request.model,
        title=request.query[:100],
    )
    session_id = session["id"]

    # Save user message
    await db.create_message(
        session_id=UUID(session_id),
        role="user",
        content=request.query,
    )

    return ResearchStartResponse(session_id=UUID(session_id))


@router.get("/{session_id}/stream")
async def stream_research(session_id: UUID):
    """SSE endpoint that streams research progress events."""
    session = await db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Get the user's query from messages
    messages = await db.get_messages(session_id)
    user_messages = [m for m in messages if m["role"] == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No research query found")

    query = user_messages[-1]["content"]
    model = session.get("model", "claude-sonnet-4-5-20250929")

    async def event_generator():
        import json as _json

        orchestrator = ResearchOrchestrator(model=model, session_id=str(session_id))
        final_report = ""
        step_ids: dict[str, str] = {}

        async for event in orchestrator.research(query):
            etype = event.event.value

            if etype == "research_complete":
                final_report = event.data.get("report", "")

            # Persist research steps to Supabase
            try:
                if etype == "plan_created":
                    step = await db.create_research_step(
                        session_id, "plan", {"steps": event.data.get("steps", [])}
                    )
                    step_ids["plan"] = step["id"]
                    await db.update_research_step(UUID(step["id"]), "completed")
                elif etype == "agent_started":
                    agent = event.data.get("agent", "unknown")
                    step_key = f"{agent}_{event.data.get('step', 0)}"
                    step = await db.create_research_step(
                        session_id, agent, event.data
                    )
                    step_ids[step_key] = step["id"]
                    await db.update_research_step(UUID(step["id"]), "running")
                elif etype == "agent_completed":
                    agent = event.data.get("agent", "unknown")
                    step_key = f"{agent}_{event.data.get('step', 0)}"
                    sid = step_ids.get(step_key)
                    if sid:
                        await db.update_research_step(UUID(sid), "completed", event.data)
                elif etype == "synthesis_started":
                    step = await db.create_research_step(
                        session_id, "synthesis", event.data
                    )
                    step_ids["synthesis"] = step["id"]
                    await db.update_research_step(UUID(step["id"]), "running")
                elif etype == "research_complete":
                    sid = step_ids.get("synthesis")
                    if sid:
                        await db.update_research_step(UUID(sid), "completed")
                elif etype == "error":
                    await db.create_research_step(
                        session_id, "error", event.data
                    )
            except Exception:
                pass  # Don't let DB errors break the SSE stream

            yield {
                "event": etype,
                "data": _json.dumps(event.data),
            }

        # Save assistant response to DB
        if final_report:
            await db.create_message(
                session_id=session_id,
                role="assistant",
                content=final_report,
            )

    return EventSourceResponse(event_generator())
