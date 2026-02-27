from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import quote

import httpx

from app.config import settings


@dataclass
class SearchResult:
    """Normalized search result from Jina AI search."""
    title: str
    url: str
    content: str
    score: float = 0.0


def _parse_jina_search_response(text: str, max_results: int = 10) -> list[SearchResult]:
    """Parse Jina search plain text response into SearchResult objects.

    Format:
    [1] Title: ...
    [1] URL Source: ...
    [1] Description: ...

    [2] Title: ...
    ...
    """
    results: list[SearchResult] = []

    # Split by result blocks (e.g., [1], [2], etc.)
    # Pattern matches [N] followed by field
    result_pattern = re.compile(r'\[(\d+)\]\s+(Title|URL Source|Description):\s*(.*?)(?=\[\d+\]|$)', re.DOTALL)

    matches = result_pattern.findall(text)

    current_result: dict = {}
    current_index = 0

    for index_str, field, value in matches:
        index = int(index_str)
        value = value.strip()

        if index != current_index:
            # New result block
            if current_result and current_index > 0:
                results.append(SearchResult(
                    title=current_result.get("Title", ""),
                    url=current_result.get("URL Source", ""),
                    content=current_result.get("Description", ""),
                    score=0.0,
                ))
                if len(results) >= max_results:
                    break
            current_result = {}
            current_index = index

        current_result[field] = value

    # Don't forget the last result
    if current_result and current_index > 0:
        results.append(SearchResult(
            title=current_result.get("Title", ""),
            url=current_result.get("URL Source", ""),
            content=current_result.get("Description", ""),
            score=0.0,
        ))

    return results[:max_results]


async def search(
    query: str,
    *,
    max_results: int = 10,
    time_range: str | None = None,
) -> list[SearchResult]:
    """Execute a web search using Jina AI search API.

    API: GET https://s.jina.ai/?q=<query>
    Headers:
        - Authorization: Bearer <api_key>
        - X-Respond-With: no-content

    Response is plain text format:
        [N] Title: ...
        [N] URL Source: ...
        [N] Description: ...
    """
    api_key = settings.jina_api_key
    if not api_key:
        raise ValueError("JINA_API_KEY not configured")

    # Properly encode the query for URL using percent-encoding
    # Use quote() to handle special characters including quotes
    encoded_query = quote(query, safe="")
    url = f"https://s.jina.ai/?q={encoded_query}"

    async with httpx.AsyncClient() as client:
        response = await client.get(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "X-Respond-With": "no-content",
            },
            timeout=30.0,
        )
        response.raise_for_status()

        # Jina search returns plain text, not JSON
        text = response.text

        results = _parse_jina_search_response(text, max_results)

        return results


def results_to_dicts(results: list[SearchResult]) -> list[dict]:
    """Convert SearchResult list to list of dicts for compatibility."""
    return [
        {
            "title": r.title,
            "url": r.url,
            "content": r.content,
            "score": r.score,
        }
        for r in results
    ]
