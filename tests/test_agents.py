"""Tests for agent tools and wrappers."""
import pytest
from app.tools.web_utils import is_valid_url, clean_content, extract_domain


class TestWebUtils:
    def test_valid_urls(self):
        assert is_valid_url("https://example.com") is True
        assert is_valid_url("http://example.com/path") is True
        assert is_valid_url("https://sub.domain.com/path?q=1") is True

    def test_invalid_urls(self):
        assert is_valid_url("not-a-url") is False
        assert is_valid_url("ftp://example.com") is False
        assert is_valid_url("") is False

    def test_clean_content(self):
        text = "  hello   world  \n\n  foo  "
        assert clean_content(text) == "hello world foo"

    def test_clean_content_truncation(self):
        text = "a" * 10000
        result = clean_content(text, max_length=100)
        assert len(result) == 103  # 100 + "..."
        assert result.endswith("...")

    def test_extract_domain(self):
        assert extract_domain("https://www.example.com/path") == "www.example.com"
        assert extract_domain("https://api.test.io") == "api.test.io"
