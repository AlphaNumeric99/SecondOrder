from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.tools.search_provider import SearchResponse
from app.tools.tavily_search import SearchResult


@pytest.mark.asyncio
async def test_search_provider_uses_brave_when_configured():
    with (
        patch("app.tools.search_provider.settings") as mock_settings,
        patch("app.tools.search_provider.brave_search.search", new=AsyncMock(return_value=[
            SearchResult(title="t", url="https://a.com", content="c", score=1.0)
        ])) as brave_search,
        patch("app.tools.search_provider.tavily_search.search", new=AsyncMock()) as tavily_search,
    ):
        mock_settings.search_provider = "brave"
        mock_settings.search_fallback_to_tavily = True

        from app.tools import search_provider

        result = await search_provider.search("query", max_results=3)

    assert result.provider == "brave"
    assert len(result.results) == 1
    brave_search.assert_awaited_once()
    tavily_search.assert_not_awaited()


@pytest.mark.asyncio
async def test_search_provider_falls_back_to_tavily_on_brave_error():
    with (
        patch("app.tools.search_provider.settings") as mock_settings,
        patch("app.tools.search_provider.brave_search.search", new=AsyncMock(side_effect=RuntimeError("brave down"))),
        patch("app.tools.search_provider.tavily_search.search", new=AsyncMock(return_value=[
            SearchResult(title="t2", url="https://b.com", content="c2", score=0.9)
        ])) as tavily_search,
    ):
        mock_settings.search_provider = "brave"
        mock_settings.search_fallback_to_tavily = True

        from app.tools import search_provider

        result = await search_provider.search("query", max_results=5)

    assert result.provider == "tavily"
    assert result.fallback_from == "brave"
    assert "brave down" in (result.fallback_reason or "")
    tavily_search.assert_awaited_once()


@pytest.mark.asyncio
async def test_search_provider_raises_when_provider_unsupported():
    with patch("app.tools.search_provider.settings") as mock_settings:
        mock_settings.search_provider = "unknown-provider"
        mock_settings.search_fallback_to_tavily = True

        from app.tools import search_provider

        with pytest.raises(ValueError):
            await search_provider.search("query")


@pytest.mark.asyncio
async def test_brave_search_maps_response_shape():
    from app.tools import brave_search

    payload = {
        "web": {
            "results": [
                {
                    "title": "Result 1",
                    "url": "https://example.com/1",
                    "description": "Desc 1",
                },
                {
                    "title": "Result 2",
                    "url": "https://example.com/2",
                    "extra_snippets": ["Snippet 2a", "Snippet 2b"],
                },
            ]
        }
    }

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return payload

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, *args, **kwargs):
            return FakeResponse()

    with (
        patch("app.tools.brave_search.settings") as mock_settings,
        patch("app.tools.brave_search.httpx.AsyncClient", return_value=FakeClient()),
    ):
        mock_settings.brave_api_key = "brave-key"
        results = await brave_search.search("query", max_results=2)

    assert len(results) == 2
    assert results[0].title == "Result 1"
    assert results[0].content == "Desc 1"
    assert results[1].content == "Snippet 2a Snippet 2b"


@pytest.mark.asyncio
async def test_search_agent_includes_provider_metadata_in_event():
    from app.agents.search_agent import SearchAgent

    with patch("app.agents.search_agent.search_provider.search", new=AsyncMock(return_value=SearchResponse(
        results=[SearchResult(title="T", url="https://x.com", content="C", score=1.0)],
        provider="brave",
    ))):
        agent = SearchAgent(model="openai/gpt-4o-mini", step_index=2)
        _, events = await agent.handle_tool_call("web_search", {"query": "test"})

    search_event = next(e for e in events if e.event.value == "search_result")
    assert search_event.data["provider"] == "brave"
    assert search_event.data["step"] == 2
