from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tavily import AsyncTavilyClient

from app.config import settings


@dataclass
class SearchResult:
    title: str
    url: str
    content: str
    score: float


async def search(
    query: str,
    *,
    search_depth: str = "advanced",
    max_results: int = 10,
    topic: str = "general",
    include_raw_content: bool = False,
    time_range: str | None = None,
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
) -> list[SearchResult]:
    """Execute a Tavily web search and return structured results."""
    client = AsyncTavilyClient(api_key=settings.tavily_api_key)

    kwargs: dict[str, Any] = {
        "query": query,
        "search_depth": search_depth,
        "max_results": max_results,
        "topic": topic,
        "include_raw_content": include_raw_content,
    }
    if time_range:
        kwargs["time_range"] = time_range
    if include_domains:
        kwargs["include_domains"] = include_domains
    if exclude_domains:
        kwargs["exclude_domains"] = exclude_domains

    response = await client.search(**kwargs)

    return [
        SearchResult(
            title=r.get("title", ""),
            url=r.get("url", ""),
            content=r.get("content", ""),
            score=r.get("score", 0.0),
        )
        for r in response.get("results", [])
    ]


def results_to_dicts(results: list[SearchResult]) -> list[dict[str, Any]]:
    """Convert SearchResult list to JSON-serializable dicts."""
    return [
        {"title": r.title, "url": r.url, "content": r.content, "score": r.score}
        for r in results
    ]
