from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from app.research_core.extract.service import chunk_text
from app.research_core.models.interfaces import (
    EvidenceRecord,
    ExtractionResult,
    ScrapeArtifact,
)


class EvidenceRepository:
    """Local persistence for scrape artifacts, extraction outputs, and evidence chunks."""

    def __init__(self, *, base_dir: str = ".cache/research/evidence"):
        self.base_dir = Path(base_dir)
        self.artifacts_dir = self.base_dir / "artifacts"
        self.extractions_dir = self.base_dir / "extractions"
        self.records_dir = self.base_dir / "records"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.extractions_dir.mkdir(parents=True, exist_ok=True)
        self.records_dir.mkdir(parents=True, exist_ok=True)

    def persist_artifact(self, artifact: ScrapeArtifact) -> Path:
        artifact_key = hashlib.sha1(artifact.url.encode("utf-8")).hexdigest()
        path = self.artifacts_dir / f"{artifact_key}.json"
        payload = {
            "url": artifact.url,
            "final_url": artifact.final_url,
            "status_code": artifact.status_code,
            "rendered_html_path": artifact.rendered_html_path,
            "screenshot_path": artifact.screenshot_path,
            "timing_ms": artifact.timing_ms,
            "attempts": artifact.attempts,
            "policy_applied": artifact.policy_applied,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def persist_extraction(self, extraction: ExtractionResult) -> Path:
        key = hashlib.sha1(extraction.url.encode("utf-8")).hexdigest()
        path = self.extractions_dir / f"{key}.json"
        payload = {
            "url": extraction.url,
            "method": extraction.method,
            "quality_score": extraction.quality_score,
            "quality_flags": extraction.quality_flags,
            "content_hash": extraction.content_hash,
            "content_text": extraction.content_text,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def build_records(
        self,
        *,
        url: str,
        extraction: ExtractionResult,
        retrieval_score: float = 0.0,
        chunk_size: int = 1800,
        overlap: int = 200,
        metadata: dict[str, Any] | None = None,
    ) -> list[EvidenceRecord]:
        chunks = chunk_text(extraction.content_text, chunk_size=chunk_size, overlap=overlap)
        records: list[EvidenceRecord] = []
        for index, chunk in enumerate(chunks):
            chunk_id = f"{extraction.content_hash}:{index}"
            evidence_id = hashlib.sha1(
                f"{url}|{extraction.content_hash}|{index}".encode("utf-8")
            ).hexdigest()
            records.append(
                EvidenceRecord(
                    evidence_id=evidence_id,
                    url=url,
                    chunk_id=chunk_id,
                    chunk_text=chunk,
                    extractor_method=extraction.method,
                    retrieval_score=float(retrieval_score),
                    metadata=dict(metadata or {}),
                )
            )
        return records

    def persist_records(self, *, url: str, records: list[EvidenceRecord]) -> Path:
        key = hashlib.sha1(url.encode("utf-8")).hexdigest()
        path = self.records_dir / f"{key}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(
                    json.dumps(
                        {
                            "evidence_id": record.evidence_id,
                            "url": record.url,
                            "chunk_id": record.chunk_id,
                            "chunk_text": record.chunk_text,
                            "extractor_method": record.extractor_method,
                            "retrieval_score": record.retrieval_score,
                            "metadata": record.metadata,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        return path
