from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


def _ensure_min_env() -> None:
    """Allow settings import even when unrelated backend env vars are absent."""
    defaults = {
        "OPENROUTER_API_KEY": "debug-placeholder",
        "DEFAULT_MODEL": "debug-model",
        "OPENROUTER_MODEL": "debug-model",
        "BENCHMARK_JUDGE_MODEL": "debug-model",
        "TAVILY_API_KEY": "debug-placeholder",
        "HASDATA_API_KEY": "debug-placeholder",
        "SUPABASE_URL": "https://debug.local",
        "SUPABASE_ANON_KEY": "debug-placeholder",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)


def _normalize_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars]


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_")
    return cleaned[:100] or "target"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Debug scrape pipeline end-to-end: raw HTML -> extraction methods -> clean_content. "
            "Use this to detect whether loss is in scraping, extraction, or cleanup."
        )
    )
    parser.add_argument("--url", default="", help="Target URL to scrape.")
    parser.add_argument(
        "--raw-html-file",
        default="",
        help="Optional existing HTML file. If provided, scraping is skipped.",
    )
    parser.add_argument(
        "--provider",
        default="",
        help="Override scrape provider (firecrawl|playwright|jina_reader|auto).",
    )
    parser.add_argument(
        "--render-mode",
        default="headless_default",
        choices=["headless_default", "http_only"],
        help="Scrape request render mode (ignored when --raw-html-file is used).",
    )
    parser.add_argument(
        "--timeout-profile",
        default="standard",
        choices=["fast", "standard", "slow"],
        help="Scrape timeout profile.",
    )
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=-1.0,
        help="Extraction quality threshold (default: from settings).",
    )
    parser.add_argument(
        "--max-extract-chars",
        type=int,
        default=-1,
        help="Extractor max chars (default: from settings).",
    )
    parser.add_argument(
        "--max-clean-chars",
        type=int,
        default=-1,
        help="clean_content max length (default: from settings).",
    )
    parser.add_argument(
        "--needle",
        action="append",
        default=[],
        help="Keyword expected on page. Repeat for multiple checks.",
    )
    parser.add_argument(
        "--out-dir",
        default=".cache/scrape_debug",
        help="Directory for debug artifacts.",
    )
    return parser


def _infer_issue(
    *,
    raw_html_len: int,
    selected_len: int,
    cleaned_len: int,
    truncated_by_cleaner: bool,
    needle_hits: dict[str, dict[str, bool]],
) -> str:
    if needle_hits:
        any_raw = any(hit["raw_html"] for hit in needle_hits.values())
        any_selected = any(hit["selected_extract"] for hit in needle_hits.values())
        any_cleaned = any(hit["cleaned"] for hit in needle_hits.values())
        if not any_raw:
            return "Likely scrape issue (or wrong page): needles missing in raw HTML."
        if any_raw and not any_selected:
            return "Likely extraction issue: needles exist in raw HTML but not extracted text."
        if any_selected and not any_cleaned:
            return "Likely cleanup/truncation issue: needles dropped after clean_content."

    if raw_html_len < 2000:
        return "Likely scrape issue: raw HTML is very small."
    if raw_html_len > 10000 and selected_len < 600:
        return "Likely extraction issue: large HTML but tiny extracted text."
    if truncated_by_cleaner:
        return "Likely cleanup/truncation issue: clean_content max_length truncated output."
    if selected_len > 0 and cleaned_len < int(selected_len * 0.5):
        return "Possible cleanup issue: cleaned output shrank significantly."
    return "No obvious pipeline loss detected."


async def _run(args: argparse.Namespace) -> int:
    _ensure_min_env()

    from app.config import settings
    from app.research_core.extract.service import ExtractService
    from app.research_core.models.interfaces import ScrapeRequest
    from app.research_core.scrape.service import ScrapeService
    from app.tools.web_utils import clean_content

    provider = (args.provider or settings.scrape_provider).strip().lower()
    quality_threshold = (
        float(settings.scrape_quality_threshold)
        if args.quality_threshold < 0
        else float(args.quality_threshold)
    )
    max_extract_chars = (
        int(settings.extractor_max_page_chars)
        if args.max_extract_chars <= 0
        else int(args.max_extract_chars)
    )
    max_clean_chars = (
        int(settings.extractor_max_page_chars)
        if args.max_clean_chars <= 0
        else int(args.max_clean_chars)
    )

    raw_html = ""
    target_url = args.url.strip()
    scrape_meta: dict[str, Any] = {}

    if args.raw_html_file:
        html_path = Path(args.raw_html_file).expanduser().resolve()
        if not html_path.exists():
            print(f"raw html file not found: {html_path}", file=sys.stderr)
            return 2
        raw_html = html_path.read_text(encoding="utf-8", errors="ignore")
        if not target_url:
            target_url = html_path.as_uri()
        scrape_meta = {"mode": "file", "source_path": str(html_path)}
    else:
        if not target_url:
            print("Provide --url or --raw-html-file", file=sys.stderr)
            return 2

        scrape_service = ScrapeService(
            retry_max=int(settings.scrape_retry_max),
            provider=provider,
            firecrawl_base_url=str(settings.firecrawl_base_url),
            firecrawl_api_key=str(settings.firecrawl_api_key),
            jina_reader_base_url=str(settings.jina_reader_base_url),
        )
        request = ScrapeRequest(
            url=target_url,
            render_mode=args.render_mode,
            timeout_profile=args.timeout_profile,
            domain_policy_id="debug",
        )
        artifact = await scrape_service.scrape(request)
        html_path = Path(artifact.rendered_html_path)
        raw_html = html_path.read_text(encoding="utf-8", errors="ignore")
        scrape_meta = {
            "mode": "live",
            "provider": provider,
            "request": {
                "render_mode": request.render_mode,
                "timeout_profile": request.timeout_profile,
                "domain_policy_id": request.domain_policy_id,
            },
            "artifact": {
                "url": artifact.url,
                "final_url": artifact.final_url,
                "status_code": artifact.status_code,
                "rendered_html_path": artifact.rendered_html_path,
                "screenshot_path": artifact.screenshot_path,
                "timing_ms": artifact.timing_ms,
                "attempts": artifact.attempts,
                "policy_applied": artifact.policy_applied,
            },
        }

    if not raw_html.strip():
        print("empty HTML payload; cannot continue", file=sys.stderr)
        return 1

    extractor = ExtractService(max_chars=max_extract_chars)
    method_order = [
        ("trafilatura", extractor._extract_trafilatura),
        ("readability", extractor._extract_readability),
        ("markitdown", extractor._extract_markitdown),
        ("raw", extractor._extract_raw),
    ]

    per_method: list[dict[str, Any]] = []
    method_texts: dict[str, str] = {}
    for method_name, method_fn in method_order:
        error = ""
        try:
            candidate_raw = method_fn(raw_html)
            if not isinstance(candidate_raw, str):
                candidate_raw = ""
        except Exception as exc:
            candidate_raw = ""
            error = str(exc)
        candidate = _truncate(_normalize_text(candidate_raw), max_extract_chars)
        score, flags = extractor.score_quality(candidate)
        method_texts[method_name] = candidate
        per_method.append(
            {
                "method": method_name,
                "length": len(candidate),
                "quality_score": round(score, 4),
                "quality_flags": flags,
                "error": error,
            }
        )

    selected = extractor.extract(
        url=target_url,
        raw_html=raw_html,
        quality_threshold=quality_threshold,
    )
    cleaned = clean_content(selected.content_text, max_length=max_clean_chars)
    raw_text = _normalize_text(BeautifulSoup(raw_html, "html.parser").get_text("\n"))

    needles = [needle for needle in args.needle if needle.strip()]
    needle_hits: dict[str, dict[str, bool]] = {}
    for needle in needles:
        token = needle.lower()
        needle_hits[needle] = {
            "raw_html": token in raw_html.lower(),
            "raw_text": token in raw_text.lower(),
            "selected_extract": token in selected.content_text.lower(),
            "cleaned": token in cleaned.lower(),
        }

    truncated_by_cleaner = (
        len(selected.content_text) > max_clean_chars and cleaned.endswith("...")
    )
    verdict = _infer_issue(
        raw_html_len=len(raw_html),
        selected_len=len(selected.content_text),
        cleaned_len=len(cleaned),
        truncated_by_cleaner=truncated_by_cleaner,
        needle_hits=needle_hits,
    )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    target_slug = _slug(target_url or Path(args.raw_html_file).stem)
    output_dir = Path(args.out_dir).expanduser().resolve() / f"{run_id}_{target_slug}"
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "raw.html").write_text(raw_html, encoding="utf-8")
    (output_dir / "raw_text.txt").write_text(raw_text, encoding="utf-8")
    for method_name, text in method_texts.items():
        (output_dir / f"extract_{method_name}.txt").write_text(text, encoding="utf-8")
    (output_dir / "extract_selected.txt").write_text(selected.content_text, encoding="utf-8")
    (output_dir / "cleaned_selected.txt").write_text(cleaned, encoding="utf-8")

    summary = {
        "target_url": target_url,
        "scrape": scrape_meta,
        "settings_snapshot": {
            "provider": provider,
            "quality_threshold": quality_threshold,
            "max_extract_chars": max_extract_chars,
            "max_clean_chars": max_clean_chars,
        },
        "lengths": {
            "raw_html": len(raw_html),
            "raw_text": len(raw_text),
            "selected_extract": len(selected.content_text),
            "cleaned": len(cleaned),
        },
        "selected_extraction": {
            "method": selected.method,
            "quality_score": selected.quality_score,
            "quality_flags": selected.quality_flags,
            "content_hash": selected.content_hash,
        },
        "per_method": per_method,
        "needles": needle_hits,
        "truncated_by_cleaner": truncated_by_cleaner,
        "verdict": verdict,
        "output_dir": str(output_dir),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Output directory: {output_dir}")
    print(f"Raw HTML chars: {len(raw_html)}")
    print(
        "Selected extraction: "
        f"method={selected.method}, score={selected.quality_score}, chars={len(selected.content_text)}"
    )
    print(f"Cleaned chars: {len(cleaned)} (max={max_clean_chars})")
    print(f"Verdict: {verdict}")
    if needles:
        print("Needle checks:")
        for needle, hit in needle_hits.items():
            print(
                f"  - {needle!r}: raw_html={hit['raw_html']} "
                f"selected={hit['selected_extract']} cleaned={hit['cleaned']}"
            )

    return 0


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
