from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from app.config import settings

CACHE_VERSION = 1


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _canonical_url(url: str) -> str:
    parsed = urlsplit(url.strip())
    scheme = (parsed.scheme or "https").lower()
    netloc = parsed.netloc.lower()
    path = parsed.path or "/"
    query = urlencode(sorted(parse_qsl(parsed.query, keep_blank_values=True)))
    return urlunsplit((scheme, netloc, path, query, ""))


def _cache_key(url: str, *, render_js: bool, output_format: str) -> str:
    material = (
        f"v{CACHE_VERSION}|{_canonical_url(url)}|"
        f"render_js={int(render_js)}|format={output_format.lower()}"
    )
    return sha256(material.encode("utf-8")).hexdigest()


def cache_path(url: str, *, render_js: bool, output_format: str) -> Path:
    key = _cache_key(url, render_js=render_js, output_format=output_format)
    return Path(settings.scrape_cache_dir) / f"{key}.json"


def load(url: str, *, render_js: bool, output_format: str) -> dict[str, Any] | None:
    if not settings.scrape_cache_enabled:
        return None

    path = cache_path(url, render_js=render_js, output_format=output_format)
    if not path.exists():
        return None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    fetched_at_raw = payload.get("fetched_at")
    if not isinstance(fetched_at_raw, str):
        return None

    try:
        fetched_at = datetime.fromisoformat(fetched_at_raw)
    except ValueError:
        return None

    if fetched_at.tzinfo is None:
        fetched_at = fetched_at.replace(tzinfo=timezone.utc)

    ttl = max(int(settings.scrape_cache_ttl_hours), 0)
    if ttl == 0:
        return None

    expires_at = fetched_at + timedelta(hours=ttl)
    if _utc_now() > expires_at:
        return None

    content = payload.get("content")
    if not isinstance(content, str) or not content.strip():
        return None

    status_code = payload.get("status_code", 200)
    if not isinstance(status_code, int):
        status_code = 200

    return {"content": content, "status_code": status_code}


def save(
    url: str,
    *,
    render_js: bool,
    output_format: str,
    content: str,
    status_code: int,
) -> None:
    if not settings.scrape_cache_enabled:
        return
    if not content.strip():
        return

    path = cache_path(url, render_js=render_js, output_format=output_format)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "version": CACHE_VERSION,
        "url": _canonical_url(url),
        "render_js": bool(render_js),
        "output_format": output_format.lower(),
        "status_code": int(status_code),
        "fetched_at": _utc_now().isoformat(),
        "content": content,
    }
    path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
