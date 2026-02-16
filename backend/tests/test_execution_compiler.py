from __future__ import annotations

import pytest

from app.models.execution import ExecutionGraph, ExecutionNode, NodeType
from app.services.execution_compiler import (
    compile_execution_graph,
    graph_summary,
    topological_order,
    validate_graph,
)


def test_compile_execution_graph_builds_expected_nodes():
    graph = compile_execution_graph("query", ["step one", "step two"])

    assert graph.version == 1
    assert len(graph.nodes) == 6
    assert graph.nodes[0].id == "search_0"
    assert graph.nodes[1].depends_on == ["search_0"]
    assert graph.nodes[2].depends_on == ["extract_0"]

    summary = graph_summary(graph)
    assert summary["search_nodes"] == 2
    assert summary["extract_nodes"] == 2
    assert summary["verify_nodes"] == 2
    assert summary["dependency_depth"] >= 3


def test_topological_order_respects_dependencies():
    graph = compile_execution_graph("query", ["step one"])
    ordered = topological_order(graph)
    ids = [node.id for node in ordered]
    assert ids.index("search_0") < ids.index("extract_0") < ids.index("verify_0")


def test_validate_graph_rejects_cycles():
    graph = ExecutionGraph(
        version=1,
        query="q",
        nodes=[
            ExecutionNode(id="a", node_type=NodeType.SEARCH, label="a", depends_on=["b"]),
            ExecutionNode(id="b", node_type=NodeType.EXTRACT, label="b", depends_on=["a"]),
        ],
    )

    with pytest.raises(ValueError):
        validate_graph(graph)
