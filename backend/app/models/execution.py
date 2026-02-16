from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class NodeType(StrEnum):
    SEARCH = "search"
    EXTRACT = "extract"
    VERIFY = "verify"


class NodeStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass(slots=True)
class ExecutionNode:
    id: str
    node_type: NodeType
    label: str
    depends_on: list[str] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)
    status: NodeStatus = NodeStatus.PENDING


@dataclass(slots=True)
class ExecutionGraph:
    version: int
    query: str
    nodes: list[ExecutionNode] = field(default_factory=list)

    def node_map(self) -> dict[str, ExecutionNode]:
        return {node.id: node for node in self.nodes}

    def summary(self) -> dict[str, Any]:
        counts = {
            NodeType.SEARCH.value: 0,
            NodeType.EXTRACT.value: 0,
            NodeType.VERIFY.value: 0,
        }
        for node in self.nodes:
            counts[node.node_type.value] += 1
        return {
            "version": self.version,
            "node_count": len(self.nodes),
            "search_nodes": counts[NodeType.SEARCH.value],
            "extract_nodes": counts[NodeType.EXTRACT.value],
            "verify_nodes": counts[NodeType.VERIFY.value],
            "dependency_depth": _dependency_depth(self.nodes),
        }


@dataclass(slots=True)
class VerificationTask:
    id: str
    claim: str
    step_id: str | None = None
    constraints: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class VerificationResult:
    task_id: str
    status: str
    score: float
    reason: str
    citations: list[dict[str, Any]] = field(default_factory=list)


def _dependency_depth(nodes: list[ExecutionNode]) -> int:
    if not nodes:
        return 0
    node_ids = {node.id for node in nodes}
    depth_cache: dict[str, int] = {}

    def depth(node_id: str) -> int:
        if node_id in depth_cache:
            return depth_cache[node_id]
        node = next(n for n in nodes if n.id == node_id)
        valid_deps = [dep for dep in node.depends_on if dep in node_ids]
        value = 1 if not valid_deps else 1 + max(depth(dep) for dep in valid_deps)
        depth_cache[node_id] = value
        return value

    return max(depth(node.id) for node in nodes)
