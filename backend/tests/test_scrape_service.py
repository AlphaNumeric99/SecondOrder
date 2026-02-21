from __future__ import annotations

from pathlib import Path

import pytest

from app.research_core.models.interfaces import ScrapeRequest
from app.research_core.scrape.service import ScrapeService, resolve_domain_policy


def test_resolve_domain_policy_applies_domain_overrides():
    policy = resolve_domain_policy(
        "https://x.com/langchain",
        timeout_profile="fast",
        retry_max=2,
    )
    assert policy.domain == "x.com"
    assert policy.wait_until == "networkidle"
    assert policy.max_attempts == 3
    assert policy.timeout_ms >= 10000


@pytest.mark.asyncio
async def test_scrape_service_retries_then_succeeds(tmp_path: Path):
    attempts: list[int] = []

    async def flaky_fetcher(request, _policy, attempt):
        attempts.append(attempt)
        if attempt == 1:
            raise RuntimeError("transient failure")
        return "<html><body>final content</body></html>", request.url, 200, None

    service = ScrapeService(
        artifacts_dir=str(tmp_path / "scrape"),
        retry_max=2,
        fetcher=flaky_fetcher,
    )
    artifact = await service.scrape(ScrapeRequest(url="https://example.com"))
    assert attempts == [1, 2]
    assert artifact.attempts == 2
    assert Path(artifact.rendered_html_path).exists()


@pytest.mark.asyncio
async def test_auto_provider_falls_through_chain(monkeypatch):
    service = ScrapeService(provider="auto")
    req = ScrapeRequest(url="https://example.com")
    policy = resolve_domain_policy(req.url)

    async def fail_playwright(*_args, **_kwargs):
        raise RuntimeError("playwright down")

    async def fail_firecrawl(*_args, **_kwargs):
        raise RuntimeError("firecrawl down")

    async def ok_jina(*_args, **_kwargs):
        return "markdown body", req.url, 200, None

    monkeypatch.setattr(service, "_fetch_with_playwright", fail_playwright)
    monkeypatch.setattr(service, "_fetch_with_firecrawl", fail_firecrawl)
    monkeypatch.setattr(service, "_fetch_with_jina_reader", ok_jina)

    html, final_url, status, screenshot = await service._fetch_default(req, policy, 1)
    assert html == "markdown body"
    assert final_url == req.url
    assert status == 200
    assert screenshot is None


@pytest.mark.asyncio
async def test_js_heavy_page_path_succeeds_with_headless_provider(monkeypatch, tmp_path: Path):
    service = ScrapeService(
        artifacts_dir=str(tmp_path / "scrape"),
        provider="playwright",
    )

    async def fake_headless(_request, _policy):
        return (
            "<html><body><div id='app'>Rendered dynamic content from JS</div></body></html>",
            "https://example.com/final",
            200,
            None,
        )

    async def fail_http(*_args, **_kwargs):
        raise RuntimeError("http fallback should not be used")

    monkeypatch.setattr(service, "_fetch_with_playwright", fake_headless)
    monkeypatch.setattr(service, "_fetch_with_httpx", fail_http)

    artifact = await service.scrape(ScrapeRequest(url="https://example.com/js"))
    rendered = Path(artifact.rendered_html_path).read_text(encoding="utf-8")
    assert "Rendered dynamic content from JS" in rendered
    assert artifact.final_url == "https://example.com/final"
