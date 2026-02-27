from __future__ import annotations

from unittest.mock import patch

import pytest

from app.tools.search_provider import SearchResponse
from app.tools.jina_search import SearchResult


@pytest.mark.asyncio
async def test_search_provider_uses_jina_when_configured():
    from app.tools import search_provider

    with patch("app.tools.search_provider.settings") as mock_settings:
        mock_settings.search_provider = "jina"

        result = await search_provider.search("query", max_results=3)

    assert result.provider == "jina"


@pytest.mark.asyncio
async def test_search_provider_raises_when_provider_unsupported():
    with patch("app.tools.search_provider.settings") as mock_settings:
        mock_settings.search_provider = "unknown-provider"

        from app.tools import search_provider

        with pytest.raises(ValueError):
            await search_provider.search("query")
