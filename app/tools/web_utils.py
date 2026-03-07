from __future__ import annotations

import re
from urllib.parse import urlparse


def is_valid_url(url: str) -> bool:
    """Basic URL validation."""
    try:
        result = urlparse(url)
        return all([result.scheme in ("http", "https"), result.netloc])
    except Exception:
        return False


def clean_content(text: str, max_length: int = 8000) -> str:
    """Clean scraped content: collapse whitespace, trim to max length."""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_length:
        text = text[:max_length] + "..."
    return text


def extract_domain(url: str) -> str:
    """Extract domain from URL for display."""
    try:
        return urlparse(url).netloc
    except Exception:
        return url
