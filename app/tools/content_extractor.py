from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO

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


def _extract_with_trafilatura(raw_html: str) -> str:
    import trafilatura

    extracted = trafilatura.extract(raw_html, output_format="txt")
    if not isinstance(extracted, str):
        return ""
    return _normalize_text(extracted)


@lru_cache(maxsize=1)
def _markitdown_converter():
    from markitdown import MarkItDown

    return MarkItDown()


def _clean_markitdown_text(markdown_text: str) -> str:
    text = re.sub(r"<img\b[^>]*>", "", markdown_text, flags=re.IGNORECASE)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"!\[[^\]]*\]\[[^\]]*\]", "", text)
    text = re.sub(r"\[([^\]]+)\]\((?:https?:)?//[^)]+\)", r"\1", text)
    text = re.sub(r"<https?://[^>]+>", "", text)
    text = re.sub(r"<data:image/[^>]+>", "", text, flags=re.IGNORECASE)

    cleaned_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append("")
            continue
        if re.fullmatch(r"(?:https?://|data:image/\S+)\S*", stripped, flags=re.IGNORECASE):
            continue
        if re.fullmatch(
            r"\[[^\]]+\]:\s*(?:<)?(?:https?://|data:image/)\S+(?:>)?",
            stripped,
            flags=re.IGNORECASE,
        ):
            continue
        cleaned_lines.append(stripped)

    return _normalize_text("\n".join(cleaned_lines))


def _extract_with_markitdown(raw_html: str) -> str:
    converter = _markitdown_converter()
    result = converter.convert_stream(BytesIO(raw_html.encode("utf-8")), file_extension=".html")
    text_content = getattr(result, "text_content", "")
    if not isinstance(text_content, str):
        return ""
    return _clean_markitdown_text(text_content)


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

    primary_text = _extract_with_trafilatura(primary_input)
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

    if fallback_mode == "none":
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

    try:
        md_text = _extract_with_markitdown(primary_input)
    except Exception:
        md_text = ""
    if md_text and not _looks_low_quality(md_text):
        clipped = _truncate(md_text, target_chars)
        return ExtractedContent(
            url=url,
            title=title,
            text=clipped,
            method="markitdown",
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
