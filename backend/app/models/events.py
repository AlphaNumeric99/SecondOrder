from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EventType(str, Enum):
    PLAN_CREATED = "plan_created"
    PLAN_REVISED = "plan_revised"
    EXECUTION_COMPILED = "execution_compiled"
    MESH_STAGE_STARTED = "mesh_stage_started"
    MESH_STAGE_COMPLETED = "mesh_stage_completed"
    MEMORY_UPSERTED = "memory_upserted"
    VERIFICATION_STARTED = "verification_started"
    VERIFICATION_COMPLETED = "verification_completed"
    AGENT_STARTED = "agent_started"
    AGENT_PROGRESS = "agent_progress"
    AGENT_COMPLETED = "agent_completed"
    SEARCH_RESULT = "search_result"
    SCRAPE_RESULT = "scrape_result"
    SYNTHESIS_STARTED = "synthesis_started"
    SYNTHESIS_PROGRESS = "synthesis_progress"
    RESEARCH_COMPLETE = "research_complete"
    ERROR = "error"


@dataclass
class SSEEvent:
    event: EventType
    data: dict[str, Any] = field(default_factory=dict)

    def format(self) -> str:
        return f"event: {self.event.value}\ndata: {json.dumps(self.data)}\n\n"
