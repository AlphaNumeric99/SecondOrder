from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from io import BytesIO

from bs4 import BeautifulSoup

from app.research_core.models.interfaces import ExtractionResult

NAV_MARKERS = (
    "main menu",
    "navigation",
    "skip to",
    "cookie",
    "subscribe",
    "sign in",
)


def _normalize_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars]


def chunk_text(text: str, *, chunk_size: int = 1800, overlap: int = 200) -> list[str]:
    if not text.strip():
        return []
    chunks: list[str] = []
    step = max(chunk_size - overlap, 200)
    start = 0
    while start < len(text):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


@dataclass(slots=True)
class Candidate:
    method: str
    text: str
    score: float
    flags: list[str]


class ExtractService:
    """Extraction ensemble with quality scoring and fallback chain."""

    def __init__(self, *, max_chars: int = 120000):
        self.max_chars = max(int(max_chars), 1000)

    def extract(
        self,
        *,
        url: str,
        raw_html: str,
        quality_threshold: float = 0.55,
    ) -> ExtractionResult:
        methods = [
            ("trafilatura", self._extract_trafilatura),
            ("readability", self._extract_readability),
            ("markitdown", self._extract_markitdown),
            ("raw", self._extract_raw),
        ]
        candidates: list[Candidate] = []

        for method, fn in methods:
            extracted = _normalize_text(fn(raw_html))
            extracted = _truncate(extracted, self.max_chars)
            score, flags = self.score_quality(extracted)
            candidates.append(Candidate(method=method, text=extracted, score=score, flags=flags))
            if extracted and score >= quality_threshold and method != "raw":
                return self._to_result(url, method, extracted, score, flags)

        best = max(candidates, key=lambda item: item.score)
        return self._to_result(url, best.method, best.text, best.score, best.flags)

    def score_quality(self, text: str) -> tuple[float, list[str]]:
        flags: list[str] = []
        lowered = text.lower()
        text_len = len(text)
        if text_len < 450:
            flags.append("too_short")

        marker_hits = sum(lowered.count(marker) for marker in NAV_MARKERS)
        if marker_hits >= 4:
            flags.append("nav_heavy")

        unique_words = len(set(re.findall(r"[a-zA-Z]{3,}", lowered)))
        if unique_words < 80:
            flags.append("low_variety")

        length_score = min(text_len / 3000.0, 1.0)
        nav_penalty = min(marker_hits * 0.08, 0.5)
        variety_boost = min(unique_words / 500.0, 0.3)
        base = 0.2 + (0.6 * length_score) + variety_boost - nav_penalty
        score = max(0.0, min(base, 1.0))
        return score, flags

    def _to_result(
        self,
        url: str,
        method: str,
        text: str,
        score: float,
        flags: list[str],
    ) -> ExtractionResult:
        return ExtractionResult(
            url=url,
            method=method,  # type: ignore[arg-type]
            quality_score=round(score, 4),
            quality_flags=flags,
            content_text=text,
            content_hash=hashlib.sha1(text.encode("utf-8")).hexdigest(),
        )

    def _extract_trafilatura(self, raw_html: str) -> str:
        try:
            import trafilatura
        except Exception:
            return ""
        extracted = trafilatura.extract(raw_html, output_format="txt")
        return extracted if isinstance(extracted, str) else ""

    def _extract_readability(self, raw_html: str) -> str:
        try:
            from readability import Document
        except Exception:
            return ""

        try:
            doc = Document(raw_html)
            summary_html = doc.summary(html_partial=True)
        except Exception:
            return ""
        soup = BeautifulSoup(summary_html, "html.parser")
        return soup.get_text("\n")

    def _extract_markitdown(self, raw_html: str) -> str:
        try:
            from markitdown import MarkItDown
        except Exception:
            return ""
        try:
            result = MarkItDown().convert_stream(BytesIO(raw_html.encode("utf-8")), file_extension=".html")
        except Exception:
            return ""
        text_content = getattr(result, "text_content", "")
        if not isinstance(text_content, str):
            return ""
        text_content = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text_content)
        text_content = re.sub(r"\[([^\]]+)\]\((?:https?:)?//[^)]+\)", r"\1", text_content)
        return text_content

    def _extract_raw(self, raw_html: str) -> str:
        soup = BeautifulSoup(raw_html, "html.parser")
        return soup.get_text("\n")
