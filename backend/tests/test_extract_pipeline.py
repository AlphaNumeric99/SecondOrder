from __future__ import annotations

from app.research_core.extract.service import ExtractService


def test_quality_scoring_flags_short_and_nav_noise():
    service = ExtractService()
    score, flags = service.score_quality("Main menu\nNavigation\nSign in")
    assert score < 0.55
    assert "too_short" in flags


def test_extract_falls_back_to_markitdown_when_primary_methods_empty(monkeypatch):
    service = ExtractService(max_chars=5000)
    monkeypatch.setattr(service, "_extract_trafilatura", lambda *_: "")
    monkeypatch.setattr(service, "_extract_readability", lambda *_: "")
    monkeypatch.setattr(
        service,
        "_extract_markitdown",
        lambda *_: " ".join(["Recovered markdown content for evidence extraction."] * 80),
    )
    result = service.extract(
        url="https://example.com/article",
        raw_html="<html><body>placeholder</body></html>",
        quality_threshold=0.55,
    )
    assert result.method == "markitdown"
    assert result.quality_score >= 0.55
    assert "Recovered markdown content" in result.content_text


def test_extract_uses_raw_when_all_methods_low_quality(monkeypatch):
    service = ExtractService(max_chars=1000)
    monkeypatch.setattr(service, "_extract_trafilatura", lambda *_: "short")
    monkeypatch.setattr(service, "_extract_readability", lambda *_: "")
    monkeypatch.setattr(service, "_extract_markitdown", lambda *_: "")
    monkeypatch.setattr(service, "_extract_raw", lambda *_: "tiny")
    result = service.extract(
        url="https://example.com/low-quality",
        raw_html="<html><body>tiny</body></html>",
        quality_threshold=0.95,
    )
    assert result.method in {"trafilatura", "raw"}
    assert result.quality_score < 0.95
    assert result.content_hash

