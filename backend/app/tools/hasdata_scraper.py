from __future__ import annotations

from dataclasses import dataclass

import httpx

from app.config import settings
from app.tools import scrape_cache

HASDATA_BASE_URL = "https://api.hasdata.com/scrape/web"
SUPPORTED_OUTPUT_FORMATS = {"html", "text", "markdown"}


@dataclass
class ScrapeResult:
    url: str
    content: str
    status_code: int
    output_format: str
    from_cache: bool = False


def _resolve_output_format(output_format_override: str | None = None) -> str:
    fmt = (output_format_override or settings.scrape_output_format).lower().strip()
    if fmt not in SUPPORTED_OUTPUT_FORMATS:
        return "html"
    return fmt


def _extract_content(payload: dict, output_format: str) -> str:
    if output_format == "markdown":
        return str(payload.get("markdown") or payload.get("text") or payload.get("content") or "")
    if output_format == "text":
        return str(payload.get("text") or payload.get("markdown") or payload.get("content") or "")
    return str(payload.get("content") or payload.get("html") or payload.get("text") or "")


async def scrape(
    url: str,
    *,
    render_js: bool = False,
    output_format_override: str | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> ScrapeResult:
    """Scrape a URL using Hasdata API and return cleaned content."""
    output_format = _resolve_output_format(output_format_override)

    cached = scrape_cache.load(url, render_js=render_js, output_format=output_format)
    if cached is not None:
        return ScrapeResult(
            url=url,
            content=str(cached["content"]),
            status_code=int(cached["status_code"]),
            output_format=output_format,
            from_cache=True,
        )

    request_body = {
        "url": url,
        "outputFormat": ["json", output_format],
    }
    if render_js:
        request_body["jsRendering"] = True

    async def _do_request(client: httpx.AsyncClient) -> dict:
        response = await client.post(
            HASDATA_BASE_URL,
            json=request_body,
            headers={
                "x-api-key": settings.hasdata_api_key,
                "Content-Type": "application/json",
            },
        )
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, dict) else {}

    if http_client is None:
        async with httpx.AsyncClient(timeout=30.0) as client:
            data = await _do_request(client)
    else:
        data = await _do_request(http_client)

    content = _extract_content(data if isinstance(data, dict) else {}, output_format)
    status_code = 200
    if isinstance(data, dict):
        status_code = int(data.get("statusCode") or data.get("status_code") or 200)

    scrape_cache.save(
        url,
        render_js=render_js,
        output_format=output_format,
        content=content,
        status_code=status_code,
    )

    return ScrapeResult(
        url=url,
        content=content,
        status_code=status_code,
        output_format=output_format,
        from_cache=False,
    )


async def scrape_multiple(
    urls: list[str],
    *,
    render_js: bool = False,
    output_format_override: str | None = None,
) -> list[ScrapeResult]:
    """Scrape multiple URLs concurrently."""
    import asyncio

    async with httpx.AsyncClient(timeout=30.0) as client:
        tasks = [
            scrape(
                url,
                render_js=render_js,
                output_format_override=output_format_override,
                http_client=client,
            )
            for url in urls
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    scraped: list[ScrapeResult] = []
    for r in results:
        if isinstance(r, ScrapeResult):
            scraped.append(r)
        # Silently skip failed scrapes — the orchestrator will note missing sources
    return scraped
