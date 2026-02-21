from __future__ import annotations

from pathlib import Path

from app.research_core.evidence.repository import EvidenceRepository
from app.research_core.models.interfaces import ExtractionResult, ScrapeArtifact


def test_build_records_uses_deterministic_ids(tmp_path: Path):
    repo = EvidenceRepository(base_dir=str(tmp_path / "evidence"))
    extraction = ExtractionResult(
        url="https://example.com",
        method="trafilatura",
        quality_score=0.9,
        quality_flags=[],
        content_text=" ".join(["evidence chunk data"] * 200),
        content_hash="abc123hash",
    )

    first = repo.build_records(url="https://example.com", extraction=extraction)
    second = repo.build_records(url="https://example.com", extraction=extraction)

    assert [r.evidence_id for r in first] == [r.evidence_id for r in second]
    assert [r.chunk_id for r in first] == [r.chunk_id for r in second]


def test_persist_artifacts_and_extractions_and_records(tmp_path: Path):
    repo = EvidenceRepository(base_dir=str(tmp_path / "evidence"))
    artifact = ScrapeArtifact(
        url="https://example.com",
        final_url="https://example.com/final",
        status_code=200,
        rendered_html_path=str(tmp_path / "rendered.html"),
        screenshot_path=None,
        timing_ms=1200,
        attempts=1,
        policy_applied="default",
    )
    extraction = ExtractionResult(
        url="https://example.com",
        method="markitdown",
        quality_score=0.8,
        quality_flags=["ok"],
        content_text=" ".join(["long form extracted text"] * 100),
        content_hash="hash-1",
    )
    Path(artifact.rendered_html_path).write_text("<html></html>", encoding="utf-8")

    artifact_path = repo.persist_artifact(artifact)
    extraction_path = repo.persist_extraction(extraction)
    records = repo.build_records(url="https://example.com", extraction=extraction)
    records_path = repo.persist_records(url="https://example.com", records=records)

    assert artifact_path.exists()
    assert extraction_path.exists()
    assert records_path.exists()
    assert len(records) > 0

