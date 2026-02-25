from __future__ import annotations

from typing import Any

from app.models.events import EventType, SSEEvent
from app.models.research_plan import ResearchPlan


def plan_created(research_plan: ResearchPlan) -> SSEEvent:
    """Emit plan created event with structured research plan."""
    return SSEEvent(
        event=EventType.PLAN_CREATED,
        data={
            "id": research_plan.id,
            "original_query": research_plan.original_query,
            "steps": [
                {
                    "id": step.id,
                    "query": step.query,
                    "purpose": step.purpose,
                    "dependencies": step.dependencies,
                    "status": step.status.value,
                }
                for step in research_plan.steps
            ],
            "version": research_plan.version,
            "created_at": research_plan.created_at,
        },
    )


def plan_revised(old_step_id: str, new_steps: list[dict[str, Any]]) -> SSEEvent:
    """Emit plan revised event when a step is revised during research."""
    return SSEEvent(
        event=EventType.PLAN_REVISED,
        data={"old_step_id": old_step_id, "new_steps": new_steps},
    )


def execution_compiled(summary: dict[str, Any]) -> SSEEvent:
    return SSEEvent(event=EventType.EXECUTION_COMPILED, data=summary)


def mesh_stage_started(stage: str, **kwargs: Any) -> SSEEvent:
    return SSEEvent(event=EventType.MESH_STAGE_STARTED, data={"stage": stage, **kwargs})


def mesh_stage_completed(stage: str, **kwargs: Any) -> SSEEvent:
    return SSEEvent(event=EventType.MESH_STAGE_COMPLETED, data={"stage": stage, **kwargs})


def memory_upserted(**kwargs: Any) -> SSEEvent:
    return SSEEvent(event=EventType.MEMORY_UPSERTED, data=kwargs)


def verification_started(**kwargs: Any) -> SSEEvent:
    return SSEEvent(event=EventType.VERIFICATION_STARTED, data=kwargs)


def verification_completed(**kwargs: Any) -> SSEEvent:
    return SSEEvent(event=EventType.VERIFICATION_COMPLETED, data=kwargs)


def agent_started(agent: str, step: int | None = None, **kwargs: Any) -> SSEEvent:
    data: dict[str, Any] = {"agent": agent}
    if step is not None:
        data["step"] = step
    data.update(kwargs)
    return SSEEvent(event=EventType.AGENT_STARTED, data=data)


def agent_progress(agent: str, **kwargs: Any) -> SSEEvent:
    return SSEEvent(event=EventType.AGENT_PROGRESS, data={"agent": agent, **kwargs})


def agent_completed(agent: str, **kwargs: Any) -> SSEEvent:
    return SSEEvent(event=EventType.AGENT_COMPLETED, data={"agent": agent, **kwargs})


def search_result(
    step: int,
    results: list[dict],
    *,
    provider: str | None = None,
    fallback_from: str | None = None,
    fallback_reason: str | None = None,
) -> SSEEvent:
    data: dict[str, Any] = {"step": step, "results": results}
    if provider:
        data["provider"] = provider
    if fallback_from:
        data["fallback_from"] = fallback_from
    if fallback_reason:
        data["fallback_reason"] = fallback_reason
    return SSEEvent(event=EventType.SEARCH_RESULT, data=data)


def scrape_result(url: str, content_preview: str) -> SSEEvent:
    return SSEEvent(
        event=EventType.SCRAPE_RESULT,
        data={"url": url, "content_preview": content_preview},
    )


def synthesis_started(sources_count: int) -> SSEEvent:
    return SSEEvent(event=EventType.SYNTHESIS_STARTED, data={"sources_count": sources_count})


def synthesis_progress(chunk: str) -> SSEEvent:
    return SSEEvent(event=EventType.SYNTHESIS_PROGRESS, data={"chunk": chunk})


def research_complete(
    report: str,
    sources: list[dict],
    tokens_used: int = 0,
    runtime_ms: int | None = None,
) -> SSEEvent:
    data: dict[str, Any] = {
        "report": report,
        "sources": sources,
        "tokens_used": tokens_used,
    }
    if runtime_ms is not None:
        data["runtime_ms"] = runtime_ms
    return SSEEvent(event=EventType.RESEARCH_COMPLETE, data=data)


def error(message: str, agent: str | None = None) -> SSEEvent:
    data: dict[str, Any] = {"message": message}
    if agent:
        data["agent"] = agent
    return SSEEvent(event=EventType.ERROR, data=data)
