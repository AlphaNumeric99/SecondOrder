from __future__ import annotations

from dataclasses import dataclass

from app.config import settings
from app.services.env_safety import sanitize_ssl_keylogfile
from app.tools import jina_search
from app.tools.jina_search import SearchResult


@dataclass
class SearchResponse:
    results: list[SearchResult]
    provider: str
    fallback_from: str | None = None
    fallback_reason: str | None = None


async def search(
    query: str,
    *,
    search_depth: str = "advanced",
    max_results: int = 10,
    time_range: str | None = None,
) -> SearchResponse:
    sanitize_ssl_keylogfile()
    provider = settings.search_provider.lower().strip()

    if provider == "jina":
        results = await jina_search.search(
            query=query,
            max_results=max_results,
            time_range=time_range,
        )
        return SearchResponse(results=results, provider="jina")

    raise ValueError(f"Unsupported SEARCH_PROVIDER: {settings.search_provider}")


def results_to_dicts(results: list[SearchResult]) -> list[dict]:
    return jina_search.results_to_dicts(results)
