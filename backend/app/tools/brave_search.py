from __future__ import annotations

from typing import Any

import httpx

from app.config import settings
from app.tools.tavily_search import SearchResult

BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"

FRESHNESS_MAP = {
    "day": "pd",
    "week": "pw",
    "month": "pm",
    "year": "py",
}


async def search(
    query: str,
    *,
    max_results: int = 10,
    time_range: str | None = None,
) -> list[SearchResult]:
    """Execute a Brave web search and normalize results."""
    if not settings.brave_api_key:
        raise RuntimeError("BRAVE_API_KEY is not configured")

    params: dict[str, Any] = {
        "q": query,
        "count": max_results,
    }
    if time_range and time_range in FRESHNESS_MAP:
        params["freshness"] = FRESHNESS_MAP[time_range]

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(
            BRAVE_SEARCH_URL,
            params=params,
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": settings.brave_api_key,
            },
        )
        response.raise_for_status()
        payload = response.json()

    raw_results = payload.get("web", {}).get("results", [])
    total = max(len(raw_results), 1)
    mapped: list[SearchResult] = []
    for idx, item in enumerate(raw_results):
        snippets = item.get("extra_snippets", []) or []
        description = item.get("description", "") or ""
        content = description.strip() or " ".join(snippets).strip()
        # Brave does not expose a direct relevance score in this response shape.
        score = max(0.0, 1.0 - (idx / total))
        mapped.append(
            SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                content=content,
                score=score,
            )
        )
    return mapped
