from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from app.config import settings
from app.tools import hasdata_scraper, scrape_cache


class _FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


@pytest.mark.asyncio
async def test_scrape_uses_local_cache_on_second_call(monkeypatch, tmp_path):
    monkeypatch.delenv("SSLKEYLOGFILE", raising=False)
    monkeypatch.setattr(settings, "hasdata_api_key", "test-key")
    monkeypatch.setattr(settings, "scrape_output_format", "markdown")
    monkeypatch.setattr(settings, "scrape_cache_enabled", True)
    monkeypatch.setattr(settings, "scrape_cache_dir", str(tmp_path))
    monkeypatch.setattr(settings, "scrape_cache_ttl_hours", 24)

    calls = 0
    captured_payloads: list[dict] = []

    async def fake_post(self, url: str, **kwargs):  # noqa: ARG001
        nonlocal calls
        calls += 1
        captured_payloads.append(kwargs.get("json", {}))
        return _FakeResponse({"statusCode": 200, "markdown": "# Page Content"})

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    first = await hasdata_scraper.scrape("https://example.com/path")
    second = await hasdata_scraper.scrape("https://example.com/path")

    assert calls == 1
    assert first.from_cache is False
    assert second.from_cache is True
    assert second.content == "# Page Content"
    assert captured_payloads[0]["outputFormat"] == ["json", "markdown"]


@pytest.mark.asyncio
async def test_scrape_bypasses_cache_when_disabled(monkeypatch, tmp_path):
    monkeypatch.delenv("SSLKEYLOGFILE", raising=False)
    monkeypatch.setattr(settings, "hasdata_api_key", "test-key")
    monkeypatch.setattr(settings, "scrape_output_format", "text")
    monkeypatch.setattr(settings, "scrape_cache_enabled", False)
    monkeypatch.setattr(settings, "scrape_cache_dir", str(tmp_path))
    monkeypatch.setattr(settings, "scrape_cache_ttl_hours", 24)

    calls = 0

    async def fake_post(self, url: str, **kwargs):  # noqa: ARG001
        nonlocal calls
        calls += 1
        return _FakeResponse({"statusCode": 200, "text": "hello"})

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    first = await hasdata_scraper.scrape("https://example.com/no-cache")
    second = await hasdata_scraper.scrape("https://example.com/no-cache")

    assert calls == 2
    assert first.from_cache is False
    assert second.from_cache is False


@pytest.mark.asyncio
async def test_expired_cache_entry_is_ignored(monkeypatch, tmp_path):
    monkeypatch.delenv("SSLKEYLOGFILE", raising=False)
    monkeypatch.setattr(settings, "hasdata_api_key", "test-key")
    monkeypatch.setattr(settings, "scrape_output_format", "markdown")
    monkeypatch.setattr(settings, "scrape_cache_enabled", True)
    monkeypatch.setattr(settings, "scrape_cache_dir", str(tmp_path))
    monkeypatch.setattr(settings, "scrape_cache_ttl_hours", 1)

    url = "https://example.com/ttl"
    cache_file = scrape_cache.cache_path(url, render_js=False, output_format="markdown")
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    stale_payload = {
        "version": 1,
        "url": url,
        "render_js": False,
        "output_format": "markdown",
        "status_code": 200,
        "fetched_at": (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat(),
        "content": "stale",
    }
    cache_file.write_text(json.dumps(stale_payload), encoding="utf-8")

    calls = 0

    async def fake_post(self, _url: str, **kwargs):  # noqa: ARG001
        nonlocal calls
        calls += 1
        return _FakeResponse({"statusCode": 200, "markdown": "fresh"})

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    result = await hasdata_scraper.scrape(url)

    assert calls == 1
    assert result.from_cache is False
    assert result.content == "fresh"


@pytest.mark.asyncio
async def test_scrape_respects_output_format_override(monkeypatch, tmp_path):
    monkeypatch.delenv("SSLKEYLOGFILE", raising=False)
    monkeypatch.setattr(settings, "hasdata_api_key", "test-key")
    monkeypatch.setattr(settings, "scrape_output_format", "markdown")
    monkeypatch.setattr(settings, "scrape_cache_enabled", True)
    monkeypatch.setattr(settings, "scrape_cache_dir", str(tmp_path))
    monkeypatch.setattr(settings, "scrape_cache_ttl_hours", 24)

    captured_payloads: list[dict] = []

    async def fake_post(self, _url: str, **kwargs):  # noqa: ARG001
        captured_payloads.append(kwargs.get("json", {}))
        return _FakeResponse({"statusCode": 200, "html": "<html><body>ok</body></html>"})

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    result = await hasdata_scraper.scrape(
        "https://example.com/html",
        output_format_override="html",
    )

    assert result.output_format == "html"
    assert captured_payloads[0]["outputFormat"] == ["json", "html"]
