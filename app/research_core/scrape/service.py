from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable
from urllib.parse import urlparse

import httpx

from app.services.env_safety import sanitize_ssl_keylogfile
from app.research_core.models.interfaces import ScrapeArtifact, ScrapeRequest

TIMEOUT_MS_BY_PROFILE = {
    "fast": 10000,
    "standard": 20000,
    "slow": 35000,
}

DOMAIN_POLICY_OVERRIDES = {
    "x.com": {"wait_until": "networkidle", "timeout_ms": 25000},
    "twitter.com": {"wait_until": "networkidle", "timeout_ms": 25000},
    "docs.langchain.com": {"wait_until": "domcontentloaded", "timeout_ms": 22000},
}


@dataclass(frozen=True, slots=True)
class DomainPolicy:
    policy_id: str
    domain: str
    timeout_ms: int
    wait_until: str
    max_attempts: int


FetchResult = tuple[str, str, int, str | None]
Fetcher = Callable[[ScrapeRequest, DomainPolicy, int], Awaitable[FetchResult]]


def _normalize_domain(url: str) -> str:
    host = (urlparse(url).hostname or "").lower().strip()
    if host.startswith("www."):
        host = host[4:]
    return host


def resolve_domain_policy(
    url: str,
    *,
    timeout_profile: str = "standard",
    policy_id: str = "default",
    retry_max: int = 1,
) -> DomainPolicy:
    domain = _normalize_domain(url)
    timeout_ms = TIMEOUT_MS_BY_PROFILE.get(timeout_profile, TIMEOUT_MS_BY_PROFILE["standard"])
    wait_until = "domcontentloaded"

    override = DOMAIN_POLICY_OVERRIDES.get(domain)
    if override:
        timeout_ms = int(override.get("timeout_ms", timeout_ms))
        wait_until = str(override.get("wait_until", wait_until))

    max_attempts = max(int(retry_max), 0) + 1
    return DomainPolicy(
        policy_id=policy_id,
        domain=domain,
        timeout_ms=max(timeout_ms, 1000),
        wait_until=wait_until,
        max_attempts=max_attempts,
    )


class ScrapeService:
    """Scrape service using Jina AI Reader API."""

    def __init__(
        self,
        *,
        artifacts_dir: str = ".cache/research/scrape",
        retry_max: int = 1,
        fetcher: Fetcher | None = None,
        capture_screenshot: bool = False,
        provider: str = "jina",
        jina_api_key: str = "",
    ):
        from app.config import settings

        self.artifacts_dir = Path(artifacts_dir)
        self.retry_max = max(int(retry_max), 0)
        self._fetcher = fetcher
        self.capture_screenshot = bool(capture_screenshot)
        self.provider = provider.lower().strip() or "jina"
        self.jina_api_key = jina_api_key or settings.jina_api_key
        self._html_dir = self.artifacts_dir / "html"
        self._meta_dir = self.artifacts_dir / "metadata"
        self._html_dir.mkdir(parents=True, exist_ok=True)
        self._meta_dir.mkdir(parents=True, exist_ok=True)

    async def scrape(self, request: ScrapeRequest) -> ScrapeArtifact:
        sanitize_ssl_keylogfile()
        policy = resolve_domain_policy(
            request.url,
            timeout_profile=request.timeout_profile,
            policy_id=request.domain_policy_id,
            retry_max=self.retry_max,
        )
        fetcher = self._fetcher or self._fetch_default
        started = time.monotonic()
        last_error: Exception | None = None

        for attempt in range(1, policy.max_attempts + 1):
            try:
                html, final_url, status_code, screenshot_path = await fetcher(
                    request,
                    policy,
                    attempt,
                )
                digest = hashlib.sha1(
                    f"{request.url}|{final_url}|{html}".encode("utf-8")
                ).hexdigest()
                html_path = self._html_dir / f"{digest}.html"
                html_path.write_text(html, encoding="utf-8")

                artifact = ScrapeArtifact(
                    url=request.url,
                    final_url=final_url,
                    status_code=int(status_code),
                    rendered_html_path=str(html_path),
                    screenshot_path=screenshot_path,
                    timing_ms=int((time.monotonic() - started) * 1000),
                    attempts=attempt,
                    policy_applied=policy.policy_id,
                )
                self._persist_metadata(digest, artifact)
                return artifact
            except Exception as exc:  # pragma: no cover - branch covered by retry tests
                last_error = exc
                if attempt < policy.max_attempts:
                    await asyncio.sleep(min(0.25 * attempt, 1.0))

        raise RuntimeError(f"Scrape failed for {request.url}: {last_error}")

    def _persist_metadata(self, artifact_id: str, artifact: ScrapeArtifact) -> None:
        payload = {
            "url": artifact.url,
            "final_url": artifact.final_url,
            "status_code": artifact.status_code,
            "rendered_html_path": artifact.rendered_html_path,
            "screenshot_path": artifact.screenshot_path,
            "timing_ms": artifact.timing_ms,
            "attempts": artifact.attempts,
            "policy_applied": artifact.policy_applied,
        }
        (self._meta_dir / f"{artifact_id}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    async def _fetch_default(
        self,
        request: ScrapeRequest,
        policy: DomainPolicy,
        _attempt: int,
    ) -> FetchResult:
        return await self._fetch_with_jina(request, policy)

    async def _fetch_with_jina(
        self,
        request: ScrapeRequest,
        policy: DomainPolicy,
    ) -> FetchResult:
        """Fetch using Jina AI Reader API (https://r.jina.ai/<url>)."""
        if not self.jina_api_key:
            raise RuntimeError("Jina API key not configured. Set JINA_API_KEY in environment.")

        # Build the Jina Reader URL
        target = f"https://r.jina.ai/{request.url}"

        headers = {
            "Authorization": f"Bearer {self.jina_api_key}",
            "X-Return-Format": "markdown",
        }

        # Enable JavaScript rendering if requested
        if request.render_mode != "http_only":
            headers["X-Engine"] = "cf-browser-rendering"

        timeout_seconds = max(policy.timeout_ms / 1000.0, 1.0)
        async with httpx.AsyncClient(timeout=timeout_seconds, follow_redirects=True) as client:
            response = await client.get(target, headers=headers)
            response.raise_for_status()

        text = response.text
        if not text:
            raise RuntimeError("Jina Reader returned empty body")

        # Jina returns markdown content - we store it as text
        return text, request.url, int(response.status_code), None
