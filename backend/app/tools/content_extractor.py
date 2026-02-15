from __future__ import annotations

import re
import shutil
from functools import lru_cache
from pathlib import Path
from dataclasses import dataclass
from typing import Any

from bs4 import BeautifulSoup

from app.config import settings

NAV_MARKERS = (
    "main menu",
    "navigation",
    "contribute",
    "tools",
    "jump to content",
    "appearance",
)


@dataclass
class ExtractedContent:
    url: str
    title: str
    text: str
    method: str
    fallback_used: bool
    raw_length: int
    extracted_length: int


def _normalize_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def _extract_title(raw_content: str) -> str:
    soup = BeautifulSoup(raw_content, "html.parser")
    title = soup.title.string if soup.title and soup.title.string else ""
    return _normalize_text(title)


def _looks_low_quality(text: str) -> bool:
    normalized = text.lower()
    marker_hits = sum(normalized.count(marker) for marker in NAV_MARKERS)
    if len(text) < 500:
        return True
    if marker_hits >= 4 and len(text) < 2500:
        return True
    return False


def _extract_with_trafilatura(raw_html: str, url: str) -> str:
    import trafilatura

    extracted = trafilatura.extract(raw_html, output_format="txt")
    if not isinstance(extracted, str):
        return ""
    return _normalize_text(extracted)


def _parse_readabilipy_payload(payload: dict[str, Any]) -> tuple[str, str]:
    title = ""
    raw_title = payload.get("title")
    if isinstance(raw_title, str):
        title = _normalize_text(raw_title)

    plain_text = payload.get("plain_text")
    if isinstance(plain_text, list):
        chunks: list[str] = []
        for item in plain_text:
            if isinstance(item, str):
                chunks.append(item)
                continue
            if isinstance(item, dict):
                text_value = item.get("text")
                if isinstance(text_value, str):
                    chunks.append(text_value)
        normalized = _normalize_text("\n\n".join(chunks))
        if normalized:
            return title, normalized
    if isinstance(plain_text, str):
        normalized = _normalize_text(plain_text)
        if normalized:
            return title, normalized

    content = payload.get("content")
    if isinstance(content, str):
        soup = BeautifulSoup(content, "html.parser")
        return title, _normalize_text(soup.get_text("\n"))

    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
                continue
            if isinstance(item, dict):
                text_value = item.get("text")
                if isinstance(text_value, str):
                    chunks.append(text_value)
        return title, _normalize_text("\n\n".join(chunks))

    return title, ""


@lru_cache(maxsize=1)
def _readabilipy_js_ready() -> bool:
    try:
        import readabilipy
    except Exception:
        return False

    if shutil.which("node") is None:
        return False

    js_dir = Path(readabilipy.__file__).resolve().parent / "javascript"
    node_modules = js_dir / "node_modules"
    return node_modules.exists()


def _extract_with_readabilipy(raw_html: str, *, use_readability: bool) -> tuple[str, str]:
    from readabilipy import simple_json_from_html_string

    if use_readability and not _readabilipy_js_ready():
        use_readability = False

    payload = simple_json_from_html_string(raw_html, use_readability=use_readability)
    if not isinstance(payload, dict):
        return "", ""
    return _parse_readabilipy_payload(payload)


def extract_main_content(
    url: str,
    raw_content: str,
    *,
    max_chars: int | None = None,
) -> ExtractedContent:
    """Extract main article content from raw scraped payload."""
    target_chars = (
        max_chars
        if max_chars is not None
        else int(settings.extractor_max_page_chars)
    )

    fallback_mode = settings.extractor_fallback.lower().strip()
    title = _extract_title(raw_content)

    # If the payload is plain text/markdown, Trafilatura may not help. Use normalized text directly.
    seems_html = "<html" in raw_content.lower() or "<body" in raw_content.lower()
    primary_input = raw_content if seems_html else f"<html><body>{raw_content}</body></html>"

    primary_text = _extract_with_trafilatura(primary_input, url)
    if primary_text and not _looks_low_quality(primary_text):
        clipped = _truncate(primary_text, target_chars)
        return ExtractedContent(
            url=url,
            title=title,
            text=clipped,
            method="trafilatura",
            fallback_used=False,
            raw_length=len(raw_content),
            extracted_length=len(clipped),
        )

    if fallback_mode != "readabilipy":
        normalized = _truncate(_normalize_text(raw_content), target_chars)
        return ExtractedContent(
            url=url,
            title=title,
            text=normalized,
            method="raw",
            fallback_used=False,
            raw_length=len(raw_content),
            extracted_length=len(normalized),
        )

    for use_readability, method in ((False, "readabilipy_fast"), (True, "readabilipy_js")):
        try:
            rb_title, rb_text = _extract_with_readabilipy(primary_input, use_readability=use_readability)
        except Exception:
            continue
        if rb_text and not _looks_low_quality(rb_text):
            clipped = _truncate(rb_text, target_chars)
            chosen_title = rb_title or title
            return ExtractedContent(
                url=url,
                title=chosen_title,
                text=clipped,
                method=method,
                fallback_used=True,
                raw_length=len(raw_content),
                extracted_length=len(clipped),
            )

    normalized = _truncate(_normalize_text(raw_content), target_chars)
    return ExtractedContent(
        url=url,
        title=title,
        text=normalized,
        method="raw",
        fallback_used=True,
        raw_length=len(raw_content),
        extracted_length=len(normalized),
    )
