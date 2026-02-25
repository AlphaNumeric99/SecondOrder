from __future__ import annotations

import pytest

from app.services.prompt_store import render_prompt


def test_render_prompt_substitutes_template_values():
    prompt = render_prompt(
        "search_agent.system_prompt",
        today_iso="2026-02-21",
        today_year=2026,
    )
    assert "2026-02-21" in prompt
    assert "(2026)" in prompt


def test_render_prompt_raises_for_unknown_key():
    with pytest.raises(KeyError):
        render_prompt("missing.prompt.key")
