from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


def _ensure_min_env() -> None:
    """Allow this utility to run even if full backend env vars are missing."""
    defaults = {
        "OPENROUTER_API_KEY": "debug-placeholder",
        "TAVILY_API_KEY": "debug-placeholder",
        "HASDATA_API_KEY": "debug-placeholder",
        "SUPABASE_URL": "https://debug.local",
        "SUPABASE_ANON_KEY": "debug-placeholder",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)


def _load_cached_payload(
    url: str,
    *,
    render_js: bool,
    output_format: str,
) -> tuple[Path, dict[str, Any] | None]:
    from app.tools import scrape_cache

    cache_file = scrape_cache.cache_path(
        url,
        render_js=render_js,
        output_format=output_format,
    )
    if not cache_file.exists():
        return cache_file, None

    try:
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
    except Exception:
        return cache_file, None
    if not isinstance(payload, dict):
        return cache_file, None
    return cache_file, payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Test content extractor behavior against a cached Hasdata HTML file.",
    )
    parser.add_argument(
        "--url",
        default="https://worldpopulationreview.com/country-rankings/gun-deaths-by-country",
        help="Target URL used to derive cache key (default: WPR gun deaths page).",
    )
    parser.add_argument(
        "--render-js",
        action="store_true",
        help="Use render_js=True when resolving cache key.",
    )
    parser.add_argument(
        "--output-format",
        default="html",
        choices=["html", "text", "markdown"],
        help="Cache output format to inspect.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=120000,
        help="Max chars passed to extractor.",
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=1200,
        help="How many extracted chars to print as preview.",
    )
    parser.add_argument(
        "--needle",
        action="append",
        default=[],
        help="Keyword to test presence in raw HTML and extracted text (repeatable).",
    )
    parser.add_argument(
        "--dump-extracted",
        default="",
        help="Optional file path to write full extracted text.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    _ensure_min_env()

    try:
        from app.tools import content_extractor
    except Exception as exc:
        print(f"Failed to import extractor modules: {exc}", file=sys.stderr)
        return 2

    cache_path, payload = _load_cached_payload(
        args.url,
        render_js=args.render_js,
        output_format=args.output_format,
    )

    print(f"URL: {args.url}")
    print(f"Cache file: {cache_path}")
    if payload is None:
        print("Cache payload: NOT FOUND or unreadable")
        return 1

    raw_content = payload.get("content", "")
    if not isinstance(raw_content, str) or not raw_content:
        print("Cache payload has no usable 'content' field.")
        return 1

    extracted = content_extractor.extract_main_content(
        args.url,
        raw_content,
        max_chars=args.max_chars,
    )

    print("---- Metadata ----")
    print(f"status_code: {payload.get('status_code')}")
    print(f"fetched_at: {payload.get('fetched_at')}")
    print(f"payload_output_format: {payload.get('output_format')}")
    print(f"raw_length: {len(raw_content)}")
    print(f"extractor_method: {extracted.method}")
    print(f"fallback_used: {extracted.fallback_used}")
    print(f"extracted_length: {len(extracted.text)}")
    print(f"title: {extracted.title[:200]}")

    needles = args.needle or [
        "GunHomicideVictimsRateViaUN_2022",
        "GunHomicideVictimsRateViaUN_2023",
        "Austria",
        "Switzerland",
        "Singapore",
    ]
    print("---- Needle Checks ----")
    for needle in needles:
        in_raw = needle in raw_content
        in_extracted = needle in extracted.text
        print(f"{needle!r}: raw={in_raw} extracted={in_extracted}")

    preview_len = max(int(args.preview_chars), 0)
    print("---- Extracted Preview ----")
    if preview_len > 0:
        print(extracted.text[:preview_len])
    else:
        print("(preview disabled)")

    if args.dump_extracted:
        out_path = Path(args.dump_extracted)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(extracted.text, encoding="utf-8")
        print(f"Wrote extracted text to: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
