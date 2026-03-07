from __future__ import annotations

import json
from pathlib import Path
from string import Template
from typing import Any


PROMPTS_PATH = Path(__file__).resolve().parents[1] / "prompts" / "prompts.json"
_catalog_cache: dict[str, Any] | None = None
_catalog_mtime_ns: int | None = None


def _load_catalog() -> dict[str, Any]:
    global _catalog_cache, _catalog_mtime_ns
    mtime_ns = PROMPTS_PATH.stat().st_mtime_ns
    if _catalog_cache is not None and _catalog_mtime_ns == mtime_ns:
        return _catalog_cache

    payload = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Prompt catalog must be a JSON object.")
    _catalog_cache = payload
    _catalog_mtime_ns = mtime_ns
    return payload


def _resolve_prompt_entry(key: str) -> str:
    node: Any = _load_catalog()
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"Prompt key not found: {key}")
        node = node[part]
    if not isinstance(node, str):
        raise TypeError(f"Prompt key must map to a string: {key}")
    return node


def render_prompt(key: str, **values: Any) -> str:
    template = Template(_resolve_prompt_entry(key))
    try:
        return template.substitute(**values)
    except KeyError as exc:
        missing = str(exc.args[0])
        raise KeyError(f"Missing template value '{missing}' for prompt '{key}'") from exc


def clear_prompt_cache() -> None:
    global _catalog_cache, _catalog_mtime_ns
    _catalog_cache = None
    _catalog_mtime_ns = None
