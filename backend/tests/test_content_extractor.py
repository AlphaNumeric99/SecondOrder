from __future__ import annotations

import readabilipy

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


def test_parse_readabilipy_payload_handles_plain_text_dicts():
    payload = {
        "title": "Readability Title",
        "plain_text": [
            {"text": "Line one"},
            {"text": "Line two"},
        ],
    }

    title, text = content_extractor._parse_readabilipy_payload(payload)

    assert title == "Readability Title"
    assert "Line one" in text
    assert "Line two" in text


def test_extract_main_content_prefers_readabilipy_fast_first(monkeypatch):
    monkeypatch.setattr(content_extractor, "_extract_with_trafilatura", lambda *_args, **_kwargs: "")
    calls: list[bool] = []

    def _fake_readabilipy(_raw_html: str, *, use_readability: bool):
        calls.append(use_readability)
        if not use_readability:
            return "Fast Title", " ".join(["Fast fallback content"] * 50)
        return "", ""

    monkeypatch.setattr(content_extractor, "_extract_with_readabilipy", _fake_readabilipy)

    result = content_extractor.extract_main_content(
        "https://example.com/fallback-fast",
        "<html><head><title>Original</title></head><body>Body</body></html>",
        max_chars=1000,
    )

    assert result.method == "readabilipy_fast"
    assert result.fallback_used is True
    assert calls == [False]


def test_extract_with_readabilipy_disables_js_when_unavailable(monkeypatch):
    observed: dict[str, bool] = {}

    def _fake_simple_json_from_html_string(_html: str, *, use_readability: bool):
        observed["use_readability"] = use_readability
        return {
            "title": "Test",
            "plain_text": [{"text": "Recovered text"}],
        }

    monkeypatch.setattr(content_extractor, "_readabilipy_js_ready", lambda: False)
    monkeypatch.setattr(
        readabilipy,
        "simple_json_from_html_string",
        _fake_simple_json_from_html_string,
    )

    _title, text = content_extractor._extract_with_readabilipy(
        "<html><body>test</body></html>",
        use_readability=True,
    )

    assert observed["use_readability"] is False
    assert "Recovered text" in text


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
