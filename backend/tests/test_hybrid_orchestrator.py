from __future__ import annotations

from types import MethodType
from unittest.mock import AsyncMock

import pytest

from app.agents.orchestrator import ResearchNotes, ResearchOrchestrator, SynthesisReview
from app.models.execution import VerificationResult
from app.services import streaming


@pytest.mark.asyncio
async def test_orchestrator_emits_mesh_and_verification_events(monkeypatch):
    """Test that the default execution path emits all expected events."""
    monkeypatch.setattr("app.agents.orchestrator.settings.shadow_mode", False)

    orchestrator = ResearchOrchestrator(model="openai/gpt-4o-mini", session_id="s1")

    orchestrator._generate_plan = AsyncMock(return_value=["step one"])  # type: ignore[method-assign]
    orchestrator._run_parallel_searches_bounded = AsyncMock(  # type: ignore[method-assign]
        return_value=(
            [{"title": "A", "url": "https://example.com", "content": "x", "score": 1.0}],
            [],
        )
    )
    orchestrator._select_top_urls = AsyncMock(return_value=["https://example.com"])  # type: ignore[method-assign]
    orchestrator._run_parallel_scrapes_bounded = AsyncMock(  # type: ignore[method-assign]
        return_value=({"https://example.com": "evidence text"}, [])
    )
    orchestrator._upsert_memory_chunks = AsyncMock(  # type: ignore[method-assign]
        return_value=(
            {"inserted_chunks": 1, "deduplicated_chunks": 0, "documents_processed": 1},
            [
                streaming.memory_upserted(
                    inserted_chunks=1,
                    deduplicated_chunks=0,
                    documents_processed=1,
                )
            ],
        )
    )
    orchestrator._run_verification_stage = AsyncMock(  # type: ignore[method-assign]
        return_value=(
            [
                VerificationResult(
                    task_id="verify_task_0",
                    status="supported",
                    score=0.9,
                    reason="ok",
                    citations=[{"url": "https://example.com"}],
                )
            ],
            [
                streaming.verification_started(task_id="verify_task_0", claim="step one"),
                streaming.verification_completed(
                    task_id="verify_task_0",
                    status="supported",
                    score=0.9,
                    reason="ok",
                    citations=[],
                ),
            ],
        )
    )
    orchestrator._capture_research_notes = AsyncMock(return_value=ResearchNotes(highlights=["h1"]))  # type: ignore[method-assign]
    orchestrator._persist_research_notes = AsyncMock()  # type: ignore[method-assign]
    orchestrator._generate_draft_report = AsyncMock(return_value="draft")  # type: ignore[method-assign]
    orchestrator._review_report_sufficiency = AsyncMock(  # type: ignore[method-assign]
        return_value=SynthesisReview(needs_more_research=False)
    )
    orchestrator._persist_synthesis_review = AsyncMock()  # type: ignore[method-assign]

    async def fake_synthesize(
        self,
        query,
        search_results,
        scraped_content,
        notes=None,
        pipeline_started_at=None,
    ):
        yield streaming.research_complete(
            report="final",
            sources=[{"title": "A", "url": "https://example.com", "domain": "example.com"}],
            tokens_used=42,
        )

    orchestrator._synthesize = MethodType(fake_synthesize, orchestrator)  # type: ignore[method-assign]

    events = [event async for event in orchestrator.research("test query")]
    event_names = [event.event.value for event in events]

    assert "execution_compiled" in event_names
    assert "mesh_stage_started" in event_names
    assert "memory_upserted" in event_names
    assert "verification_started" in event_names
    assert "verification_completed" in event_names
    assert event_names[-1] == "research_complete"


@pytest.mark.asyncio
async def test_orchestrator_uses_default_execution_path(monkeypatch):
    """Test that research() always uses the default execution path (no legacy dispatch)."""
    monkeypatch.setattr("app.agents.orchestrator.settings.shadow_mode", True)

    orchestrator = ResearchOrchestrator(model="openai/gpt-4o-mini")

    # Mock the _execute_research method to track which path is taken
    async def fake_execute_research(_query: str):
        yield streaming.plan_created(["default"])
        yield streaming.research_complete(report="default", sources=[], tokens_used=1)

    orchestrator._execute_research = fake_execute_research  # type: ignore[method-assign]

    events = [event async for event in orchestrator.research("test query")]
    assert [event.event.value for event in events] == ["plan_created", "research_complete"]
