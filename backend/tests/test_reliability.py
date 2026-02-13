from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from app.api.routes.research import _event_step_key, start_research
from app.models.schemas import ResearchRequest
from app.services import supabase as db


def test_event_step_key_is_stable_for_scraper_url():
    url = "https://example.com/a"
    start_key = _event_step_key("agent_started", {"agent": "scraper", "url": url})
    done_key = _event_step_key("agent_completed", {"agent": "scraper", "url": url})
    other_key = _event_step_key("agent_started", {"agent": "scraper", "url": "https://example.com/b"})

    assert start_key is not None
    assert start_key == done_key
    assert start_key != other_key


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
async def test_run_parallel_scrapes_emits_completion_per_url():
    from app.agents.orchestrator import ResearchOrchestrator

    urls = ["https://example.com/a", "https://example.com/b"]

    class FakeScraperAgent:
        def __init__(self, *args, **kwargs):
            self.scraped_content = {"https://example.com/a": "content a"}

        async def run(self, prompt: str):
            if False:
                yield None

    with patch("app.agents.orchestrator.ScraperAgent", FakeScraperAgent):
        orchestrator = ResearchOrchestrator(model="openai/gpt-4o-mini")
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
