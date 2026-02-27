from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.services import search_executor
from app.tools.search_provider import SearchResponse
from app.tools.jina_search import SearchResult


def test_build_step_queries_adds_canonical_and_timeline_variants():
    step = 'artist who played in "AJJ" from 2016 to 2021 viral songs'
    queries = search_executor.build_step_queries(step, max_queries=4)

    assert queries
    assert queries[0] == step
    assert any("site:wikipedia.org" in query for query in queries)
    assert any("timeline" in query for query in queries)
    assert len(queries) <= 4


def test_build_step_queries_prefers_explicit_query_clause():
    step = (
        "Search Phoenix New Times Best of Phoenix 2006 to identify winner — "
        'query: site:phoenixnewtimes.com "Best of Phoenix 2006" "Best Holy Local Band"'
    )
    queries = search_executor.build_step_queries(step, max_queries=3)

    assert queries
    assert queries[0].startswith("site:phoenixnewtimes.com")
    assert "Search Phoenix New Times" not in queries[0]


def test_build_step_queries_removes_unresolved_placeholders():
    step = 'query: site:wikipedia.org "[Drummer Name]" AJJ drummer 2016 2021'
    queries = search_executor.build_step_queries(step, max_queries=3)

    assert queries
    assert "[" not in queries[0]
    assert "]" not in queries[0]


def test_build_step_queries_keeps_context_when_site_operator_is_inline():
    step = (
        "Compute the intersection and resolve country names using Wikipedia "
        "(site:wikipedia.org) to standardize entities."
    )
    queries = search_executor.build_step_queries(step, max_queries=2)

    assert queries
    assert "compute the intersection" in queries[0].lower()
    assert "site:wikipedia.org" in queries[0].lower()
    assert not queries[0].lower().startswith("site:wikipedia.org)")


def test_build_step_queries_adds_domain_variant_from_source_label():
    step = (
        "Retrieve GPI 2023 top 10 countries from Vision of Humanity "
        "source: visionofhumanity.org and report ranked names."
    )
    queries = search_executor.build_step_queries(step, max_queries=4)

    assert queries
    assert any("site:visionofhumanity.org" in query.lower() for query in queries)


@pytest.mark.asyncio
async def test_run_deterministic_searches_emits_step_events():
    async def fake_search(query: str, **kwargs):
        return SearchResponse(
            results=[
                SearchResult(
                    title=f"title-{query}",
                    url=f"https://example.com/{abs(hash(query)) % 1000}",
                    content="snippet",
                    score=0.9,
                )
            ],
            provider="jina",
        )

    with (
        patch(
            "app.services.search_executor.build_step_queries",
            side_effect=lambda step, max_queries: [step],
        ),
        patch(
            "app.services.search_executor.search_provider.search",
            new=AsyncMock(side_effect=fake_search),
        ),
    ):
        results, events = await search_executor.run_deterministic_searches(
            ["step one", "step two"],
            step_offset=3,
            max_parallel=2,
            max_queries_per_step=1,
            max_results_per_query=5,
        )

    assert len(results) == 2
    started_steps = [e.data.get("step") for e in events if e.event.value == "agent_started"]
    completed_steps = [e.data.get("step") for e in events if e.event.value == "agent_completed"]
    assert started_steps == [3, 4]
    assert completed_steps == [3, 4]
