from __future__ import annotations

from typing import Any

from app.models.events import EventType, SSEEvent


def plan_created(steps: list[str]) -> SSEEvent:
    return SSEEvent(event=EventType.PLAN_CREATED, data={"steps": steps})


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


def search_result(step: int, results: list[dict]) -> SSEEvent:
    return SSEEvent(event=EventType.SEARCH_RESULT, data={"step": step, "results": results})


def scrape_result(url: str, content_preview: str) -> SSEEvent:
    return SSEEvent(event=EventType.SCRAPE_RESULT, data={"url": url, "content_preview": content_preview})


def synthesis_started(sources_count: int) -> SSEEvent:
    return SSEEvent(event=EventType.SYNTHESIS_STARTED, data={"sources_count": sources_count})


def synthesis_progress(chunk: str) -> SSEEvent:
    return SSEEvent(event=EventType.SYNTHESIS_PROGRESS, data={"chunk": chunk})


def research_complete(report: str, sources: list[dict], tokens_used: int = 0) -> SSEEvent:
    return SSEEvent(
        event=EventType.RESEARCH_COMPLETE,
        data={"report": report, "sources": sources, "tokens_used": tokens_used},
    )


def error(message: str, agent: str | None = None) -> SSEEvent:
    data: dict[str, Any] = {"message": message}
    if agent:
        data["agent"] = agent
    return SSEEvent(event=EventType.ERROR, data=data)
