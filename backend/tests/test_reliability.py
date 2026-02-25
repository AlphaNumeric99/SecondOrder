from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from app.api.routes.research import _event_step_key, start_research
from app.models.schemas import ResearchRequest
from app.services import database as db


def test_event_step_key_is_stable_for_scraper_url():
    url = "https://example.com/a"
    start_key = _event_step_key("agent_started", {"agent": "scraper", "url": url})
    done_key = _event_step_key("agent_completed", {"agent": "scraper", "url": url})
    other_key = _event_step_key("agent_started", {"agent": "scraper", "url": "https://example.com/b"})

    assert start_key is not None
    assert start_key == done_key
    assert start_key != other_key


def test_event_step_key_supports_mesh_and_verification_keys():
    mesh_start = _event_step_key("mesh_stage_started", {"stage": "verify"})
    mesh_done = _event_step_key("mesh_stage_completed", {"stage": "verify"})
    verify_start = _event_step_key("verification_started", {"task_id": "t1"})
    verify_done = _event_step_key("verification_completed", {"task_id": "t1"})

    assert mesh_start == "mesh_verify"
    assert mesh_start == mesh_done
    assert verify_start == "verification_t1"
    assert verify_start == verify_done


@pytest.mark.asyncio
async def test_start_research_uses_backend_model_when_missing_from_request():
    session_id = "12345678-1234-5678-1234-567812345678"
    with (
        patch("app.api.routes.research.get_model", return_value="openai/gpt-4.1"),
        patch.object(db, "create_session", new=AsyncMock(return_value={"id": session_id})) as create_session,
        patch.object(db, "create_message", new=AsyncMock()) as create_message,
    ):
        response = await start_research(ResearchRequest(query="test query", model=None))

    assert str(response.session_id) == session_id
    create_session.assert_awaited_once_with(model="openai/gpt-4.1", title="test query")
    create_message.assert_awaited()


@pytest.mark.asyncio
async def test_get_sessions_uses_to_thread_for_blocking_execute():
    from app.services import supabase

    expected = [{"id": "s1"}]
    fake_result = SimpleNamespace(data=expected)
    fake_query = MagicMock()
    fake_query.execute = MagicMock(return_value=fake_result)
    fake_table = MagicMock()
    fake_table.select.return_value.order.return_value = fake_query
    fake_client = MagicMock()
    fake_client.table.return_value = fake_table

    with (
        patch("app.services.supabase.client", return_value=fake_client),
        patch("app.services.supabase.asyncio.to_thread", new=AsyncMock(return_value=fake_result)) as to_thread,
    ):
        result = await supabase.get_sessions()

    assert result == expected
    to_thread.assert_awaited_once_with(fake_query.execute)


@pytest.mark.asyncio
async def test_get_messages_normalizes_legacy_string_metadata():
    from app.services import supabase

    session_id = UUID("12345678-1234-5678-1234-567812345678")
    raw_rows = [
        {
            "id": "m1",
            "session_id": str(session_id),
            "role": "assistant",
            "content": "hello",
            "metadata": "{}",
            "created_at": "2026-02-13T10:00:00+00:00",
        }
    ]
    fake_result = SimpleNamespace(data=raw_rows)
    fake_query = MagicMock()
    fake_query.execute = MagicMock(return_value=fake_result)

    fake_table = MagicMock()
    fake_table.select.return_value.eq.return_value.order.return_value = fake_query
    fake_client = MagicMock()
    fake_client.table.return_value = fake_table

    with (
        patch("app.services.supabase.client", return_value=fake_client),
        patch("app.services.supabase.asyncio.to_thread", new=AsyncMock(return_value=fake_result)),
    ):
        rows = await supabase.get_messages(session_id)

    assert rows[0]["metadata"] == {}


@pytest.mark.asyncio
async def test_create_message_writes_metadata_as_object():
    from app.services import supabase

    session_id = UUID("12345678-1234-5678-1234-567812345678")
    fake_result = SimpleNamespace(data=[{"id": "m1"}])
    fake_query = MagicMock()
    fake_query.execute = MagicMock(return_value=fake_result)
    insert_mock = MagicMock(return_value=fake_query)

    fake_table = MagicMock()
    fake_table.insert = insert_mock
    fake_client = MagicMock()
    fake_client.table.return_value = fake_table

    with (
        patch("app.services.supabase.client", return_value=fake_client),
        patch("app.services.supabase.asyncio.to_thread", new=AsyncMock(return_value=fake_result)),
    ):
        await supabase.create_message(
            session_id=session_id,
            role="assistant",
            content="hello",
            metadata={"source": "test"},
        )

    payload = insert_mock.call_args.args[0]
    assert payload["metadata"] == {"source": "test"}


@pytest.mark.asyncio
async def test_run_parallel_scrapes_emits_completion_per_url(tmp_path):
    from app.agents.orchestrator import ResearchOrchestrator
    from app.research_core.models.interfaces import ExtractionResult, ScrapeArtifact

    urls = ["https://example.com/a", "https://example.com/b"]

    class FakeScrapeService:
        async def scrape(self, request):
            html_path = tmp_path / f"{request.url.split('/')[-1]}.html"
            html_path.write_text("<html>ok</html>", encoding="utf-8")
            return ScrapeArtifact(
                url=request.url,
                final_url=request.url,
                status_code=200,
                rendered_html_path=str(html_path),
                screenshot_path=None,
                timing_ms=1,
                attempts=1,
                policy_applied="default",
            )

    class FakeExtractService:
        def extract(self, *, url: str, raw_html: str, quality_threshold: float):
            text = "content a" if url.endswith("/a") else ""
            return ExtractionResult(
                url=url,
                method="trafilatura",
                quality_score=0.9 if text else 0.1,
                quality_flags=[],
                content_text=text,
                content_hash="hash-a" if text else "hash-b",
            )

    class FakeEvidenceRepo:
        def persist_artifact(self, artifact):
            return None

        def persist_extraction(self, extraction):
            return None

        def build_records(self, *, url, extraction, metadata=None):
            return []

        def persist_records(self, *, url, records):
            return None

    orchestrator = ResearchOrchestrator(model="openai/gpt-4o-mini")
    orchestrator._get_scrape_services = lambda: (  # type: ignore[method-assign]
        FakeScrapeService(),
        FakeExtractService(),
        FakeEvidenceRepo(),
    )
    _, events = await orchestrator._run_parallel_scrapes(urls)

    completed = [
        e for e in events
        if e.event.value == "agent_completed" and e.data.get("agent") == "scraper"
    ]
    assert len(completed) == 2
    by_url = {e.data["url"]: e.data for e in completed}
    assert by_url["https://example.com/a"]["success"] is True
    assert by_url["https://example.com/b"]["success"] is False


@pytest.mark.asyncio
async def test_synthesis_uses_final_message_usage_for_tokens():
    from app.agents.orchestrator import ResearchOrchestrator

    class FakeStream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        @property
        def text_stream(self):
            async def gen():
                yield "hello world"
            return gen()

        async def get_final_message(self):
            return SimpleNamespace(
                usage=SimpleNamespace(input_tokens=13, output_tokens=8),
                content=[],
            )

    class FakeMessages:
        def stream(self, **kwargs):
            return FakeStream()

    class FakeClient:
        messages = FakeMessages()

    orchestrator = ResearchOrchestrator(model="openai/gpt-4o-mini")
    orchestrator.client = FakeClient()
    orchestrator._log_call = AsyncMock()

    events = [
        e
        async for e in orchestrator._synthesize(
            "query",
            [{"title": "t", "url": "https://example.com", "content": "c"}],
            {},
        )
    ]
    final_event = next(e for e in events if e.event.value == "research_complete")
    assert final_event.data["tokens_used"] == 21


def test_stream_research_emits_error_event_on_unhandled_exception():
    from app.main import app

    session_id = UUID("12345678-1234-5678-1234-567812345678")

    class BoomOrchestrator:
        def __init__(self, *args, **kwargs):
            pass

        async def research(self, query: str):
            raise RuntimeError("boom")
            if False:
                yield None

    with (
        patch("app.api.routes.research.ResearchOrchestrator", BoomOrchestrator),
        patch("app.api.routes.research.db.get_session", new=AsyncMock(return_value={"id": str(session_id), "model": "openai/gpt-4o-mini"})),
        patch("app.api.routes.research.db.get_messages", new=AsyncMock(return_value=[{"role": "user", "content": "test"}])),
        patch("app.api.routes.research.db.create_message", new=AsyncMock()),
    ):
        client = TestClient(app)
        with client.stream("GET", f"/api/research/{session_id}/stream") as response:
            body = "\n".join([line for line in response.iter_lines() if line])

    assert response.status_code == 200
    assert "event: error" in body
    assert "Research stream failed unexpectedly." in body


def test_strip_deferred_sections_removes_future_work_heading():
    from app.agents.orchestrator import ResearchOrchestrator

    orchestrator = ResearchOrchestrator(model="openai/gpt-4o-mini")
    report = (
        "## Findings\n"
        "Answer text.\n\n"
        "## Areas for Further Investigation\n"
        "- More work later.\n\n"
        "## Sources\n"
        "- [A](https://example.com)"
    )

    cleaned = orchestrator._strip_deferred_sections(report)

    assert "Areas for Further Investigation" not in cleaned
    assert "More work later" not in cleaned
    assert "## Sources" in cleaned


def test_should_run_synthesis_review_conditional_thresholds():
    from app.agents.orchestrator import ResearchNotes, ResearchOrchestrator
    from app.models.execution import VerificationResult

    orchestrator = ResearchOrchestrator(model="openai/gpt-4o-mini")
    orchestrator.review_mode = "conditional"
    orchestrator.review_min_supported_ratio = 0.7
    orchestrator.review_min_citations = 2

    should_skip = orchestrator._should_run_synthesis_review(
        notes=ResearchNotes(unresolved_points=[]),
        search_results=[
            {"url": "https://example.com/1"},
            {"url": "https://example.com/2"},
        ],
        verification_results=[
            VerificationResult(task_id="a", status="supported", score=0.9, reason="ok"),
            VerificationResult(task_id="b", status="supported", score=0.8, reason="ok"),
        ],
    )
    assert should_skip is False

    should_review = orchestrator._should_run_synthesis_review(
        notes=ResearchNotes(unresolved_points=["Missing key evidence"]),
        search_results=[{"url": "https://example.com/1"}],
        verification_results=[
            VerificationResult(task_id="a", status="supported", score=0.9, reason="ok"),
            VerificationResult(task_id="b", status="unsupported", score=0.2, reason="no"),
        ],
    )
    assert should_review is True


@pytest.mark.asyncio
async def test_run_parallel_searches_applies_step_offset():
    from app.agents.orchestrator import ResearchOrchestrator
    from app.models.events import SSEEvent, EventType

    class FakeSearchAgent:
        def __init__(self, model: str | None, step_index: int, session_id: str | None = None):
            self.step_index = step_index
            self.all_results = [{"title": f"r{step_index}", "url": f"https://example.com/{step_index}", "content": "c", "score": 1.0}]

        async def run(self, query: str, context: str = ""):
            yield SSEEvent(event=EventType.AGENT_PROGRESS, data={"agent": "search", "step": self.step_index})

    with patch("app.agents.orchestrator.SearchAgent", FakeSearchAgent):
        orchestrator = ResearchOrchestrator(model="openai/gpt-4o-mini")
        orchestrator.search_executor_mode = "agent_loop"
        _, events = await orchestrator._run_parallel_searches(
            ["q1", "q2"], step_offset=5
        )

    started_steps = [e.data.get("step") for e in events if e.event.value == "agent_started"]
    completed_steps = [e.data.get("step") for e in events if e.event.value == "agent_completed"]
    assert started_steps == [5, 6]
    assert completed_steps == [5, 6]


@pytest.mark.asyncio
async def test_capture_research_notes_falls_back_when_json_invalid():
    from app.agents.orchestrator import ResearchOrchestrator

    class FakeMessages:
        async def create(self, **kwargs):
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text="not json")],
                usage=SimpleNamespace(input_tokens=1, output_tokens=1),
            )

    class FakeClient:
        messages = FakeMessages()

    orchestrator = ResearchOrchestrator(model="openai/gpt-4o-mini")
    orchestrator.client = FakeClient()
    orchestrator._log_call = AsyncMock()

    notes = await orchestrator._capture_research_notes(
        "query text",
        [{"title": "A", "url": "https://example.com", "content": "snippet"}],
        {},
    )

    assert notes.highlights
    assert isinstance(notes.unresolved_points, list)


@pytest.mark.asyncio
async def test_review_report_sufficiency_parses_json():
    from app.agents.orchestrator import ResearchNotes, ResearchOrchestrator

    class FakeMessages:
        async def create(self, **kwargs):
            return SimpleNamespace(
                content=[
                    SimpleNamespace(
                        type="text",
                        text=(
                            '{"needs_more_research": true, '
                            '"reason": "Missing publication-year confirmation.", '
                            '"missing_points": ["Confirm release years."], '
                            '"follow_up_queries": ["artist song release year source", "song wikipedia release date"]}'
                        ),
                    )
                ],
                usage=SimpleNamespace(input_tokens=2, output_tokens=2),
            )

    class FakeClient:
        messages = FakeMessages()

    orchestrator = ResearchOrchestrator(model="openai/gpt-4o-mini")
    orchestrator.client = FakeClient()
    orchestrator._log_call = AsyncMock()

    review = await orchestrator._review_report_sufficiency(
        "query",
        "draft",
        ResearchNotes(),
    )

    assert review.needs_more_research is True
    assert review.reason.startswith("Missing publication-year")
    assert review.missing_points == ["Confirm release years."]
    assert len(review.follow_up_queries) == 2


@pytest.mark.asyncio
async def test_review_report_sufficiency_fallback_uses_uncertainty_markers():
    from app.agents.orchestrator import ResearchNotes, ResearchOrchestrator

    class FakeMessages:
        async def create(self, **kwargs):
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text="not-json")],
                usage=SimpleNamespace(input_tokens=1, output_tokens=1),
            )

    class FakeClient:
        messages = FakeMessages()

    orchestrator = ResearchOrchestrator(model="openai/gpt-4o-mini")
    orchestrator.client = FakeClient()
    orchestrator._log_call = AsyncMock()

    review = await orchestrator._review_report_sufficiency(
        "query",
        "The available sources are insufficient evidence and unclear from available sources.",
        ResearchNotes(follow_up_queries=["query corroborating source"]),
    )

    assert review.needs_more_research is True
    assert review.follow_up_queries == ["query corroborating source"]
