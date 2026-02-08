from __future__ import annotations

from dataclasses import dataclass

import httpx

from app.config import settings

HASDATA_BASE_URL = "https://api.hasdata.com/scrape/web"


@dataclass
class ScrapeResult:
    url: str
    content: str
    status_code: int


async def scrape(url: str, *, render_js: bool = False) -> ScrapeResult:
    """Scrape a URL using Hasdata API and return cleaned content."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(
            HASDATA_BASE_URL,
            params={"url": url, "js_rendering": str(render_js).lower()},
            headers={
                "x-api-key": settings.hasdata_api_key,
                "Content-Type": "application/json",
            },
        )
        response.raise_for_status()
        data = response.json()

    return ScrapeResult(
        url=url,
        content=data.get("content", data.get("text", "")),
        status_code=data.get("status_code", 200),
    )


async def scrape_multiple(urls: list[str], *, render_js: bool = False) -> list[ScrapeResult]:
    """Scrape multiple URLs concurrently."""
    import asyncio

    tasks = [scrape(url, render_js=render_js) for url in urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    scraped: list[ScrapeResult] = []
    for r in results:
        if isinstance(r, ScrapeResult):
            scraped.append(r)
        # Silently skip failed scrapes — the orchestrator will note missing sources
    return scraped
