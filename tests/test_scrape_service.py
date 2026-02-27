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
        return "<html><body>final content</html>", request.url, 200, None

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
async def test_jina_provider_uses_jina_api(monkeypatch):
    """Test that Jina provider calls the Jina API."""
    service = ScrapeService(provider="jina", jina_api_key="test-key")
    req = ScrapeRequest(url="https://example.com")
    policy = resolve_domain_policy(req.url)

    async def mock_jina(_request, _policy):
        return "markdown content", req.url, 200, None

    monkeypatch.setattr(service, "_fetch_with_jina", mock_jina)

    html, final_url, status, screenshot = await service._fetch_default(req, policy, 1)
    assert html == "markdown content"
    assert final_url == req.url
    assert status == 200
