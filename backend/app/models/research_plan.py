from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel


class StepStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    REVISED = "revised"


class ResearchStep(BaseModel):
    """A single step in a research plan."""
    id: str  # UUID or generated ID
    query: str  # The search query for this step
    purpose: str  # What this step aims to find
    dependencies: list[str] = []  # IDs of steps this depends on
    status: StepStatus = StepStatus.PENDING
    revised_from: Optional[str] = None  # ID of step this replaced
    revision_reason: Optional[str] = None  # Why this was revised


class ResearchPlan(BaseModel):
    """A structured research plan with metadata."""
    id: str  # UUID
    original_query: str
    steps: list[ResearchStep]
    version: int = 1  # Incremented on revision
    created_at: str
    updated_at: str
