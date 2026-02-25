from __future__ import annotations

from collections import deque
from typing import Any

from app.models.execution import ExecutionGraph, ExecutionNode, NodeType


def compile_execution_graph(query: str, plan_steps: list[str]) -> ExecutionGraph:
    """Compile legacy plan steps into a normalized DAG for hybrid execution."""
    cleaned_steps = _normalize_steps(plan_steps)
    nodes: list[ExecutionNode] = []

    for index, step in enumerate(cleaned_steps):
        search_id = f"search_{index}"
        extract_id = f"extract_{index}"
        verify_id = f"verify_{index}"

        nodes.append(
            ExecutionNode(
                id=search_id,
                node_type=NodeType.SEARCH,
                label=step,
                payload={"query": step, "plan_index": index},
            )
        )
        nodes.append(
            ExecutionNode(
                id=extract_id,
                node_type=NodeType.EXTRACT,
                label=f"Extract evidence for: {step}",
                depends_on=[search_id],
                payload={"query": step, "plan_index": index},
            )
        )
        nodes.append(
            ExecutionNode(
                id=verify_id,
                node_type=NodeType.VERIFY,
                label=f"Verify claim coverage for: {step}",
                depends_on=[extract_id],
                payload={"query": step, "plan_index": index},
            )
        )

    graph = ExecutionGraph(version=1, query=query, nodes=nodes)
    validate_graph(graph)
    return graph


def validate_graph(graph: ExecutionGraph) -> None:
    node_map = graph.node_map()
    for node in graph.nodes:
        for dep in node.depends_on:
            if dep not in node_map:
                raise ValueError(f"Execution graph has unknown dependency: {dep}")
    _ensure_acyclic(graph)


def topological_order(graph: ExecutionGraph) -> list[ExecutionNode]:
    node_map = graph.node_map()
    indegree: dict[str, int] = {node.id: 0 for node in graph.nodes}
    outgoing: dict[str, list[str]] = {node.id: [] for node in graph.nodes}

    for node in graph.nodes:
        for dep in node.depends_on:
            indegree[node.id] += 1
            outgoing[dep].append(node.id)

    queue = deque(sorted([node_id for node_id, deg in indegree.items() if deg == 0]))
    ordered_ids: list[str] = []

    while queue:
        node_id = queue.popleft()
        ordered_ids.append(node_id)
        for nxt in sorted(outgoing[node_id]):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)

    if len(ordered_ids) != len(graph.nodes):
        raise ValueError("Execution graph contains a dependency cycle.")
    return [node_map[node_id] for node_id in ordered_ids]


def stage_groups(graph: ExecutionGraph) -> dict[NodeType, list[ExecutionNode]]:
    return {
        NodeType.SEARCH: [n for n in graph.nodes if n.node_type == NodeType.SEARCH],
        NodeType.EXTRACT: [n for n in graph.nodes if n.node_type == NodeType.EXTRACT],
        NodeType.VERIFY: [n for n in graph.nodes if n.node_type == NodeType.VERIFY],
    }


def graph_summary(graph: ExecutionGraph) -> dict[str, Any]:
    return graph.summary()


def _normalize_steps(plan_steps: list[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for step in plan_steps:
        value = " ".join(step.split()).strip()
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(value)
    return cleaned


def _ensure_acyclic(graph: ExecutionGraph) -> None:
    _ = topological_order(graph)
