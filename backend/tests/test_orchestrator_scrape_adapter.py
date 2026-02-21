from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.agents.orchestrator import ResearchOrchestrator


@pytest.mark.asyncio
async def test_scrape_adapter_uses_primary_pipeline(monkeypatch):
    orchestrator = ResearchOrchestrator(model="openai/gpt-4o-mini")
    orchestrator._run_parallel_scrapes_primary = AsyncMock(  # type: ignore[method-assign]
        return_value=({"https://example.com": "core content"}, [])
    )

    content_map, events = await orchestrator._run_parallel_scrapes_bounded(
        ["https://example.com"],
        max_parallel=2,
    )

    orchestrator._run_parallel_scrapes_primary.assert_awaited_once_with(
        ["https://example.com"], max_parallel=2
    )
    assert content_map["https://example.com"] == "core content"
    assert events == []


@pytest.mark.asyncio
async def test_scrape_bounded_passes_configured_parallel_cap(monkeypatch):
    orchestrator = ResearchOrchestrator(model="openai/gpt-4o-mini")
    orchestrator.scrape_pipeline_max_parallel = 1
    observed: dict[str, int] = {}

    async def fake_primary(urls: list[str], *, max_parallel: int):
        observed["max_parallel"] = max_parallel
        return ({u: "ok" for u in urls}, [])

    monkeypatch.setattr(
        orchestrator,
        "_run_parallel_scrapes_primary",
        fake_primary,  # type: ignore[method-assign]
    )

    content_map, events = await orchestrator._run_parallel_scrapes_bounded(
        ["https://example.com", "https://example.org"],
        max_parallel=4,
    )

    assert observed["max_parallel"] == 4
    assert content_map["https://example.com"] == "ok"
    assert content_map["https://example.org"] == "ok"
    assert events == []
