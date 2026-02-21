from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


RenderMode = Literal["headless_default", "http_only"]
ExtractMethod = Literal["trafilatura", "readability", "markitdown", "raw"]


@dataclass(slots=True)
class ScrapeRequest:
    url: str
    render_mode: RenderMode = "headless_default"
    timeout_profile: str = "standard"
    domain_policy_id: str = "default"


@dataclass(slots=True)
class ScrapeArtifact:
    url: str
    final_url: str
    status_code: int
    rendered_html_path: str
    screenshot_path: str | None
    timing_ms: int
    attempts: int
    policy_applied: str


@dataclass(slots=True)
class ExtractionResult:
    url: str
    method: ExtractMethod
    quality_score: float
    quality_flags: list[str]
    content_text: str
    content_hash: str


@dataclass(slots=True)
class EvidenceRecord:
    evidence_id: str
    url: str
    chunk_id: str
    chunk_text: str
    extractor_method: str
    retrieval_score: float
    metadata: dict[str, Any] = field(default_factory=dict)

