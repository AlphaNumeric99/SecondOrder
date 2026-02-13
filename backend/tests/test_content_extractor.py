from __future__ import annotations

from app.config import settings
from app.tools import content_extractor


def test_extract_main_content_uses_trafilatura_path(monkeypatch):
    monkeypatch.setattr(
        content_extractor,
        "_extract_with_trafilatura",
        lambda *_args, **_kwargs: " ".join(
            [
                "Key evidence about Owen Richard Evans and AJJ timeline."
            ]
            * 30
        ),
    )
    monkeypatch.setattr(content_extractor, "_extract_with_readabilipy", lambda *_args, **_kwargs: ("", ""))

    result = content_extractor.extract_main_content(
        "https://example.com/article",
        "<html><head><title>Article</title></head><body>Body</body></html>",
        max_chars=1000,
    )

    assert result.method == "trafilatura"
    assert result.fallback_used is False
    assert "Owen Richard Evans" in result.text


def test_extract_main_content_falls_back_to_readabilipy(monkeypatch):
    monkeypatch.setattr(content_extractor, "_extract_with_trafilatura", lambda *_args, **_kwargs: "")
    monkeypatch.setattr(
        content_extractor,
        "_extract_with_readabilipy",
        lambda *_args, **_kwargs: (
            "Readability Title",
            " ".join(["Recovered article body with Christmas Kids details."] * 30),
        ),
    )

    result = content_extractor.extract_main_content(
        "https://example.com/fallback",
        "<html><head><title>Original</title></head><body>Body</body></html>",
        max_chars=1000,
    )

    assert result.fallback_used is True
    assert result.method.startswith("readabilipy")
    assert result.title == "Readability Title"
    assert "Christmas Kids" in result.text


def test_extract_main_content_returns_raw_when_fallback_disabled(monkeypatch):
    monkeypatch.setattr(settings, "extractor_fallback", "none")
    monkeypatch.setattr(content_extractor, "_extract_with_trafilatura", lambda *_args, **_kwargs: "")

    raw = "Main menu Navigation " + ("fact " * 300)
    result = content_extractor.extract_main_content(
        "https://example.com/raw",
        raw,
        max_chars=500,
    )

    assert result.method == "raw"
    assert result.fallback_used is False
    assert len(result.text) <= 503
