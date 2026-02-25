from __future__ import annotations

import hashlib
import json as _json
from uuid import UUID

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from app.agents.orchestrator import ResearchOrchestrator
from app.llm_client import get_model
from app.models.schemas import ResearchRequest, ResearchStartResponse
from app.services import logger as log_service
from app.services import streaming
from app.services import database as db

router = APIRouter(prefix="/api/research", tags=["research"])


def _event_step_key(event_type: str, data: dict) -> str | None:
    """Build deterministic persistence key for step lifecycle events."""
    if event_type in ("verification_started", "verification_completed"):
        task_id = data.get("task_id")
        if isinstance(task_id, str) and task_id:
            return f"verification_{task_id}"
        return "verification_single"

    if event_type in ("mesh_stage_started", "mesh_stage_completed"):
        stage = data.get("stage")
        if isinstance(stage, str) and stage:
            return f"mesh_{stage}"
        return "mesh_unknown"

    if event_type not in ("agent_started", "agent_completed"):
        return None

    agent = data.get("agent", "unknown")
    if agent == "scraper":
        url = data.get("url")
        if url:
            digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
            return f"{agent}_{digest}"
        return f"{agent}_single"

    step = data.get("step")
    if step is not None:
        return f"{agent}_{step}"
    return f"{agent}_single"


@router.post("", response_model=ResearchStartResponse)
async def start_research(request: ResearchRequest):
    """Start a new research session. Returns session_id to use for streaming."""
    model = request.model or get_model()
    # Create session
    session = await db.create_session(
        model=model,
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
    model = session.get("model") or get_model()

    async def event_generator():
        log_service.log_event(
            event_type="research_started",
            message="Research started",
            session_id=str(session_id),
            model=model,
            query=query[:100],
        )

        orchestrator = ResearchOrchestrator(model=model, session_id=str(session_id))
        final_report = ""
        step_ids: dict[str, str] = {}
        memory_summary: dict[str, int] = {
            "inserted_chunks": 0,
            "deduplicated_chunks": 0,
            "documents_processed": 0,
        }
        verification_counts: dict[str, int] = {
            "supported": 0,
            "partially_supported": 0,
            "unsupported": 0,
            "total": 0,
        }

        try:
            async for event in orchestrator.research(query):
                etype = event.event.value

                if etype == "research_complete":
                    final_report = event.data.get("report", "")

                # Persist research steps to Supabase
                try:
                    if etype == "plan_created":
                        plan_data = event.data if isinstance(event.data, dict) else {}
                        if "steps" not in plan_data:
                            plan_data = {
                                "steps": (
                                    plan_data.get("steps", [])
                                    if isinstance(plan_data, dict)
                                    else []
                                )
                            }
                        step = await db.create_research_step(
                            session_id, "plan", plan_data
                        )
                        step_ids["plan"] = step["id"]
                        await db.update_research_step(UUID(step["id"]), "completed")
                    elif etype == "execution_compiled":
                        step = await db.create_research_step(
                            session_id, "execution_graph", event.data
                        )
                        step_ids["execution_graph"] = step["id"]
                        await db.update_research_step(UUID(step["id"]), "completed")
                    elif etype == "mesh_stage_started":
                        stage = event.data.get("stage", "unknown")
                        step_key = _event_step_key(etype, event.data)
                        step = await db.create_research_step(
                            session_id, f"mesh_{stage}", event.data
                        )
                        if step_key:
                            step_ids[step_key] = step["id"]
                        await db.update_research_step(UUID(step["id"]), "running")
                    elif etype == "mesh_stage_completed":
                        step_key = _event_step_key(etype, event.data)
                        sid = step_ids.get(step_key or "")
                        if sid:
                            await db.update_research_step(UUID(sid), "completed", event.data)
                        stage = event.data.get("stage")
                        if stage == "verify":
                            summary_step = await db.create_research_step(
                                session_id,
                                "verification",
                                {
                                    **verification_counts,
                                    **(event.data if isinstance(event.data, dict) else {}),
                                },
                            )
                            await db.update_research_step(UUID(summary_step["id"]), "completed")
                    elif etype == "memory_upserted":
                        inserted = int(event.data.get("inserted_chunks", 0) or 0)
                        deduped = int(event.data.get("deduplicated_chunks", 0) or 0)
                        docs = int(event.data.get("documents_processed", 0) or 0)
                        memory_summary["inserted_chunks"] += inserted
                        memory_summary["deduplicated_chunks"] += deduped
                        memory_summary["documents_processed"] += docs
                        step = await db.create_research_step(
                            session_id,
                            "memory",
                            {
                                **memory_summary,
                                **event.data,
                            },
                        )
                        await db.update_research_step(UUID(step["id"]), "completed")
                    elif etype == "verification_started":
                        step_key = _event_step_key(etype, event.data)
                        step = await db.create_research_step(
                            session_id, "verification_task", event.data
                        )
                        if step_key:
                            step_ids[step_key] = step["id"]
                        await db.update_research_step(UUID(step["id"]), "running")
                    elif etype == "verification_completed":
                        step_key = _event_step_key(etype, event.data)
                        sid = step_ids.get(step_key or "")
                        if sid:
                            await db.update_research_step(UUID(sid), "completed", event.data)
                        status = event.data.get("status")
                        if isinstance(status, str):
                            verification_counts[status] = verification_counts.get(status, 0) + 1
                            verification_counts["total"] += 1
                    elif etype == "agent_started":
                        agent = event.data.get("agent", "unknown")
                        step_key = _event_step_key(etype, event.data)
                        step = await db.create_research_step(
                            session_id, agent, event.data
                        )
                        if step_key:
                            step_ids[step_key] = step["id"]
                        await db.update_research_step(UUID(step["id"]), "running")
                    elif etype == "agent_completed":
                        step_key = _event_step_key(etype, event.data)
                        sid = step_ids.get(step_key or "")
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
                            await db.update_research_step(
                                UUID(sid),
                                "completed",
                                {
                                    "tokens_used": event.data.get("tokens_used", 0),
                                    "sources_count": len(event.data.get("sources", [])),
                                    "runtime_ms": event.data.get("runtime_ms"),
                                },
                            )
                        await db.create_research_step(
                            session_id,
                            "result",
                            {
                                "sources": event.data.get("sources", []),
                                "tokens_used": event.data.get("tokens_used", 0),
                                "runtime_ms": event.data.get("runtime_ms"),
                            },
                        )
                    elif etype == "error":
                        await db.create_research_step(
                            session_id, "error", event.data
                        )
                except Exception as e:
                    log_service.log_event(
                        event_type="db_error",
                        message=f"Failed to persist research step: {etype}",
                        error=str(e),
                        session_id=str(session_id),
                    )

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
        except Exception as e:
            log_service.log_event(
                event_type="stream_error",
                message="Unhandled error in research stream",
                error=str(e),
                session_id=str(session_id),
            )
            error_event = streaming.error("Research stream failed unexpectedly.")
            yield {
                "event": error_event.event.value,
                "data": _json.dumps(error_event.data),
            }

    return EventSourceResponse(event_generator())


@router.post("/{session_id}/revise")
async def revise_plan(session_id: UUID, request: dict):
    """
    Manually revise the research plan mid-research.

    Request body:
    {
        "reason": "Why the plan needs revision",
        "feedback": "Optional user feedback or specific instructions"
    }
    """
    from app.models.research_plan import ResearchPlan

    session = await db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Get the user's query from messages
    messages = await db.get_messages(session_id)
    user_messages = [m for m in messages if m["role"] == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No research query found")

    query = user_messages[-1]["content"]

    # Get current research steps to build context
    research_steps = await db.get_research_steps(session_id)

    # Build evidence summary from completed steps
    evidence_summary = ""
    for step in research_steps:
        step_data = step.get("data", {})
        if step.get("status") == "completed":
            if step.get("step_type") == "search":
                evidence_summary += f"\n- Search results from: {step_data.get('query', 'unknown')}"
            elif step.get("step_type") == "scraper":
                evidence_summary += f"\n- Scraped content"

    revision_context = {
        "evidence_summary": evidence_summary,
        "failed_steps": [],
        "low_quality_steps": [],
        "user_feedback": request.get("feedback", ""),
        "reason": request.get("reason", "Manual revision requested"),
    }

    model = session.get("model") or get_model()
    orchestrator = ResearchOrchestrator(model=model, session_id=str(session_id))

    # For now, return a placeholder - full implementation would load current plan
    # and call orchestrator._revise_plan()
    return {
        "status": "revision_requested",
        "session_id": str(session_id),
        "message": "Plan revision initiated. This will be applied in the next research iteration.",
    }
