from __future__ import annotations

from dataclasses import dataclass

from app.config import settings
from app.services.env_safety import sanitize_ssl_keylogfile
from app.tools import brave_search, tavily_search
from app.tools.tavily_search import SearchResult


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
    use_fallback = settings.search_fallback_to_tavily

    if provider == "tavily":
        results = await tavily_search.search(
            query=query,
            search_depth=search_depth,
            max_results=max_results,
            time_range=time_range,
        )
        return SearchResponse(results=results, provider="tavily")

    if provider == "brave":
        try:
            results = await brave_search.search(
                query=query,
                max_results=max_results,
                time_range=time_range,
            )
            if results or not use_fallback:
                return SearchResponse(results=results, provider="brave")

            fallback_results = await tavily_search.search(
                query=query,
                search_depth=search_depth,
                max_results=max_results,
                time_range=time_range,
            )
            return SearchResponse(
                results=fallback_results,
                provider="tavily",
                fallback_from="brave",
                fallback_reason="brave returned zero results",
            )
        except Exception as e:
            if not use_fallback:
                raise
            fallback_results = await tavily_search.search(
                query=query,
                search_depth=search_depth,
                max_results=max_results,
                time_range=time_range,
            )
            return SearchResponse(
                results=fallback_results,
                provider="tavily",
                fallback_from="brave",
                fallback_reason=str(e),
            )

    raise ValueError(f"Unsupported SEARCH_PROVIDER: {settings.search_provider}")


def results_to_dicts(results: list[SearchResult]) -> list[dict]:
    return tavily_search.results_to_dicts(results)
