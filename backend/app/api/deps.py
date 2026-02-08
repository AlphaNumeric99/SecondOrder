from __future__ import annotations

from app.config import settings


def get_available_models() -> list[dict[str, str]]:
    """Return the list of available Claude models for research."""
    return [
        {
            "id": "claude-opus-4-6",
            "name": "Claude Opus 4.6",
            "description": "Most capable model. Best for complex, nuanced research requiring deep analysis.",
        },
        {
            "id": "claude-sonnet-4-5-20250929",
            "name": "Claude Sonnet 4.5",
            "description": "Fast and capable. Good balance of speed and quality for most research tasks.",
        },
        {
            "id": "claude-haiku-4-5-20251001",
            "name": "Claude Haiku 4.5",
            "description": "Fastest model. Best for quick lookups and lightweight research tasks.",
        },
    ]
