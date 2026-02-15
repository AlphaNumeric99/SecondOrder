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
    monkeypatch.setattr(content_extractor, "_extract_with_markitdown", lambda *_args, **_kwargs: "")

    result = content_extractor.extract_main_content(
        "https://example.com/article",
        "<html><head><title>Article</title></head><body>Body</body></html>",
        max_chars=1000,
    )

    assert result.method == "trafilatura"
    assert result.fallback_used is False
    assert "Owen Richard Evans" in result.text


def test_extract_main_content_falls_back_to_markitdown(monkeypatch):
    monkeypatch.setattr(
        content_extractor,
        "_extract_with_trafilatura",
        lambda *_args, **_kwargs: "",
    )
    monkeypatch.setattr(
        content_extractor,
        "_extract_with_markitdown",
        lambda *_args, **_kwargs: (
            " ".join(["Recovered article body with Christmas Kids details."] * 30)
        ),
    )

    result = content_extractor.extract_main_content(
        "https://example.com/fallback",
        "<html><head><title>Original</title></head><body>Body</body></html>",
        max_chars=1000,
    )

    assert result.fallback_used is True
    assert result.method == "markitdown"
    assert result.title == "Original"
    assert "Christmas Kids" in result.text


def test_clean_markitdown_text_removes_image_and_url_noise():
    raw = (
        "# Title\n\n"
        "![cover](https://cdn.example.com/cover.jpg)\n"
        "![inline-data](data:image/png;base64,AAAA)\n"
        "![ref-image][img1]\n"
        "[img1]: data:image/png;base64,BBBB\n"
        "<img src=\"https://cdn.example.com/figure.svg\" alt=\"figure\" />\n"
        "[source link](https://example.com/source)\n"
        "https://example.com/only-url\n"
        "data:image/png;base64,CCCC\n"
        "Key fact line.\n"
    )

    cleaned = content_extractor._clean_markitdown_text(raw)

    assert "![cover]" not in cleaned
    assert "![inline-data]" not in cleaned
    assert "![ref-image][img1]" not in cleaned
    assert "[img1]:" not in cleaned
    assert "<img" not in cleaned
    assert "https://example.com/only-url" not in cleaned
    assert "data:image/png;base64,CCCC" not in cleaned
    assert "source link" in cleaned
    assert "Key fact line." in cleaned


def test_extract_main_content_returns_raw_when_fallback_disabled(monkeypatch):
    monkeypatch.setattr(settings, "extractor_fallback", "none")
    monkeypatch.setattr(
        content_extractor,
        "_extract_with_trafilatura",
        lambda *_args, **_kwargs: "",
    )

    raw = "Main menu Navigation " + ("fact " * 300)
    result = content_extractor.extract_main_content(
        "https://example.com/raw",
        raw,
        max_chars=500,
    )

    assert result.method == "raw"
    assert result.fallback_used is False
    assert len(result.text) <= 503
