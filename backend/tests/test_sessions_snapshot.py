"""Tests for session history research snapshot reconstruction."""

from app.api.routes.sessions import _build_research_snapshot


def test_build_research_snapshot_hydrates_plan_steps_and_sources():
    messages = [
        {
            "role": "assistant",
            "content": "Final report with source [Alpha](https://example.com/a).",
        }
    ]
    research_steps = [
        {
            "id": "plan-1",
            "step_type": "plan",
            "status": "completed",
            "data": {"steps": ["Resolve entities", "Collect evidence"]},
        },
        {
            "id": "search-1",
            "step_type": "search",
            "status": "completed",
            "data": {
                "query": "Resolve entities",
                "results": [{"title": "Alpha", "url": "https://example.com/a", "content": "x"}],
            },
        },
        {
            "id": "scrape-1",
            "step_type": "scraper",
            "status": "completed",
            "data": {"url": "https://example.com/a", "content_preview": "captured"},
        },
    ]

    snapshot = _build_research_snapshot(messages, research_steps)

    assert snapshot["status"] == "complete"
    assert snapshot["plan"] == ["Resolve entities", "Collect evidence"]
    assert len(snapshot["steps"]) == 2
    assert snapshot["steps"][0]["agent"] == "search"
    assert snapshot["steps"][1]["agent"] == "scraper"
    assert snapshot["sources"][0]["url"] == "https://example.com/a"


def test_build_research_snapshot_marks_error_state():
    messages = []
    research_steps = [
        {
            "id": "err-1",
            "step_type": "error",
            "status": "completed",
            "data": {"message": "Search provider unavailable"},
        }
    ]

    snapshot = _build_research_snapshot(messages, research_steps)

    assert snapshot["status"] == "error"
    assert snapshot["error"] == "Search provider unavailable"
