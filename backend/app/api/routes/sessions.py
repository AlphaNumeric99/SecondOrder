from __future__ import annotations

import json
import re
from urllib.parse import urlparse
from uuid import UUID

from fastapi import APIRouter, HTTPException

from app.models.schemas import MessageResponse, SessionDetailResponse, SessionResponse
from app.services import supabase as db

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


def _parse_step_data(raw_data: object) -> dict:
    if isinstance(raw_data, dict):
        return raw_data
    if isinstance(raw_data, str):
        try:
            parsed = json.loads(raw_data)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return {}
    return {}


def _status_for_snapshot(
    *,
    has_plan: bool,
    has_search: bool,
    has_scrape: bool,
    has_running_search: bool,
    has_running_scrape: bool,
    has_report: bool,
    has_error: bool,
) -> str:
    if has_error:
        return "error"
    if has_report:
        return "complete"
    if has_running_scrape:
        return "scraping"
    if has_running_search:
        return "searching"
    if has_scrape:
        return "scraping"
    if has_search:
        return "searching"
    if has_plan:
        return "planning"
    return "idle"


def _extract_sources_from_report(report: str) -> list[dict[str, str]]:
    if not report:
        return []
    # Markdown links: [Title](https://example.com)
    link_pattern = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")
    sources: list[dict[str, str]] = []
    seen: set[str] = set()
    for title, url in link_pattern.findall(report):
        if url in seen:
            continue
        seen.add(url)
        sources.append(
            {
                "title": title.strip(),
                "url": url.strip(),
                "domain": urlparse(url).netloc or url,
            }
        )
    return sources


def _build_research_snapshot(messages: list[dict], research_steps: list[dict]) -> dict:
    plan: list[str] = []
    notes: list[str] = []
    ui_steps: list[dict] = []
    sources: list[dict] = []
    source_urls: set[str] = set()
    error_message: str | None = None

    has_plan = False
    has_search = False
    has_scrape = False
    has_running_search = False
    has_running_scrape = False
    has_error = False

    for row in research_steps:
        step_type = row.get("step_type", "")
        status = row.get("status", "pending")
        data = _parse_step_data(row.get("data"))

        if step_type == "plan":
            has_plan = True
            raw_steps = data.get("steps", [])
            if isinstance(raw_steps, list):
                plan = [s for s in raw_steps if isinstance(s, str)]
            continue

        if step_type == "search":
            has_search = True
            if status in ("pending", "running"):
                has_running_search = True
            label = data.get("query") if isinstance(data.get("query"), str) else "search agent"
            results = data.get("results") if isinstance(data.get("results"), list) else []
            if results:
                detail = f"Found {len(results)} results"
            elif isinstance(data.get("status"), str):
                detail = data["status"]
            else:
                detail = None
            ui_steps.append(
                {
                    "id": str(row.get("id")),
                    "agent": "search",
                    "status": "error" if status == "error" else "completed" if status == "completed" else "running",
                    "label": label,
                    "detail": detail,
                    "results": results,
                }
            )
            continue

        if step_type == "notes":
            raw_notes = data.get("highlights", [])
            if isinstance(raw_notes, list):
                for item in raw_notes:
                    if isinstance(item, str):
                        note = " ".join(item.split()).strip()
                        if note and note not in notes:
                            notes.append(note)
            continue

        if step_type == "scraper":
            has_scrape = True
            if status in ("pending", "running"):
                has_running_scrape = True
            label = data.get("url") if isinstance(data.get("url"), str) else "scraper agent"
            preview = data.get("content_preview") if isinstance(data.get("content_preview"), str) else ""
            if preview:
                detail = f"Scraped: {preview[:140]}..."
            elif isinstance(data.get("status"), str):
                detail = data["status"]
            else:
                detail = None
            ui_steps.append(
                {
                    "id": str(row.get("id")),
                    "agent": "scraper",
                    "status": "error" if status == "error" else "completed" if status == "completed" else "running",
                    "label": label,
                    "detail": detail,
                }
            )
            continue

        if step_type == "error" or status == "error":
            has_error = True
            if isinstance(data.get("message"), str):
                error_message = data["message"]
            continue

        # Optional persisted final payload (newer sessions)
        if step_type == "result":
            raw_sources = data.get("sources", [])
            if isinstance(raw_sources, list):
                for src in raw_sources:
                    if not isinstance(src, dict):
                        continue
                    url = src.get("url")
                    if not isinstance(url, str) or not url or url in source_urls:
                        continue
                    source_urls.add(url)
                    title = src.get("title") if isinstance(src.get("title"), str) else ""
                    domain = src.get("domain") if isinstance(src.get("domain"), str) else (urlparse(url).netloc or url)
                    sources.append({"title": title or domain, "url": url, "domain": domain})

    report = ""
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and isinstance(msg.get("content"), str):
            report = msg["content"]
            break

    if not sources:
        sources = _extract_sources_from_report(report)

    status = _status_for_snapshot(
        has_plan=has_plan,
        has_search=has_search,
        has_scrape=has_scrape,
        has_running_search=has_running_search,
        has_running_scrape=has_running_scrape,
        has_report=bool(report),
        has_error=has_error,
    )

    return {
        "status": status,
        "plan": plan,
        "notes": notes,
        "steps": ui_steps,
        "sources": sources,
        "report": report,
        "error": error_message,
    }


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
    research_steps = await db.get_research_steps(session_id)
    research_snapshot = _build_research_snapshot(messages, research_steps)

    return SessionDetailResponse(
        session=SessionResponse(**session),
        messages=[MessageResponse(**m) for m in messages],
        research=research_snapshot,
    )


@router.delete("/{session_id}")
async def delete_session(session_id: UUID):
    """Delete a research session and all its data."""
    session = await db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    await db.delete_session(session_id)
    return {"status": "deleted"}
