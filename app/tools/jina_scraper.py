from __future__ import annotations

from dataclasses import dataclass

import httpx

from app.config import settings


@dataclass
class ScrapeResult:
    """Result from Jina AI web scraping."""
    url: str
    content: str
    title: str = ""
    markdown: str = ""


async def scrape(
    url: str,
    *,
    render_js: bool = False,
    output_format_override: str | None = None,
) -> ScrapeResult:
    """Scrape a URL using Jina AI Reader API.

    API: GET https://r.jina.ai/<url>
    Headers:
        - Authorization: Bearer <api_key>
        - X-Engine: cf-browser-rendering (for JS rendering)
        - X-Return-Format: markdown
    """
    api_key = settings.jina_api_key
    if not api_key:
        raise ValueError("JINA_API_KEY not configured")

    # Build headers
    headers = {
        "Authorization": f"Bearer {api_key}",
    }

    if render_js:
        headers["X-Engine"] = "cf-browser-rendering"

    # Use markdown format (default and recommended)
    headers["X-Return-Format"] = "markdown"

    scrape_url = f"https://r.jina.ai/{url}"

    async with httpx.AsyncClient() as client:
        response = await client.get(
            scrape_url,
            headers=headers,
            timeout=60.0,
        )
        response.raise_for_status()

        # Jina Reader returns the scraped content directly as text/markdown
        content = response.text

        # Extract title from first line if present
        title = ""
        markdown = content

        return ScrapeResult(
            url=url,
            content=content,
            title=title,
            markdown=markdown,
        )
