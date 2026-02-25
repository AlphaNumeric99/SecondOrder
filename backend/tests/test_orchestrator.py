"""Tests for the research orchestrator."""
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from app.models.events import EventType


class TestOrchestratorPlan:
    """Test plan generation logic."""

    @pytest.mark.asyncio
    async def test_generate_plan_parses_json(self):
        with patch("app.config.settings") as mock_settings:
            mock_settings.openrouter_api_key = "test"
            mock_settings.openrouter_model = ""
            mock_settings.default_model = "openai/gpt-4o-mini"
            mock_settings.app_log_level = "INFO"
            mock_settings.noisy_log_level = "WARNING"

            from app.agents.orchestrator import ResearchOrchestrator

            orchestrator = ResearchOrchestrator()

            # Mock the OpenRouter-backed client
            mock_response = MagicMock()
            mock_content = MagicMock()
            mock_content.text = '["query 1", "query 2", "query 3"]'
            mock_response.content = [mock_content]

            orchestrator.client = AsyncMock()
            orchestrator.client.messages.create = AsyncMock(return_value=mock_response)

            plan = await orchestrator._generate_plan("test query")
            assert plan == ["query 1", "query 2", "query 3"]

    @pytest.mark.asyncio
    async def test_generate_plan_fallback(self):
        with patch("app.config.settings") as mock_settings:
            mock_settings.openrouter_api_key = "test"
            mock_settings.openrouter_model = ""
            mock_settings.default_model = "openai/gpt-4o-mini"
            mock_settings.app_log_level = "INFO"
            mock_settings.noisy_log_level = "WARNING"

            from app.agents.orchestrator import ResearchOrchestrator

            orchestrator = ResearchOrchestrator()

            # Mock response with invalid JSON
            mock_response = MagicMock()
            mock_content = MagicMock()
            mock_content.text = "not valid json"
            mock_response.content = [mock_content]

            orchestrator.client = AsyncMock()
            orchestrator.client.messages.create = AsyncMock(return_value=mock_response)

            plan = await orchestrator._generate_plan("test query")
            assert "test query" in plan
            assert len(plan) >= 1

    @pytest.mark.asyncio
    async def test_generate_plan_uses_planner_model_override(self):
        from app.agents import orchestrator as orch_module

        with patch.object(orch_module.settings, "planner_model", "openai/gpt-5"):
            orchestrator = orch_module.ResearchOrchestrator()

            mock_response = MagicMock()
            mock_content = MagicMock()
            mock_content.text = '["query 1"]'
            mock_response.content = [mock_content]

            orchestrator.client = AsyncMock()
            orchestrator.client.messages.create = AsyncMock(return_value=mock_response)

            plan = await orchestrator._generate_plan("test query")
            assert "query 1" in plan
            assert orchestrator.client.messages.create.await_args.kwargs["model"] == "openai/gpt-5"

    @pytest.mark.asyncio
    async def test_generate_plan_augments_short_chain_query(self):
        with patch("app.config.settings") as mock_settings:
            mock_settings.openrouter_api_key = "test"
            mock_settings.openrouter_model = ""
            mock_settings.default_model = "openai/gpt-4o-mini"
            mock_settings.app_log_level = "INFO"
            mock_settings.noisy_log_level = "WARNING"

            from app.agents.orchestrator import ResearchOrchestrator

            orchestrator = ResearchOrchestrator()

            mock_response = MagicMock()
            mock_content = MagicMock()
            mock_content.text = '["single broad step"]'
            mock_response.content = [mock_content]

            orchestrator.client = AsyncMock()
            orchestrator.client.messages.create = AsyncMock(return_value=mock_response)

            query = (
                'List the two books by the researcher who from 2012 to 2018 worked in a lab '
                'that won "Research Excellence Award" in 2010.'
            )
            plan = await orchestrator._generate_plan(query)
            assert len(plan) >= 4
            assert "winner" in plan[0].lower()
            assert any("timeline" in step.lower() for step in plan)
            assert any("book" in step.lower() for step in plan)


class TestURLSelection:
    @pytest.mark.asyncio
    async def test_select_top_urls_deduplicates(self):
        with patch("app.config.settings") as mock_settings:
            mock_settings.openrouter_api_key = "test"
            mock_settings.openrouter_model = ""
            mock_settings.default_model = "openai/gpt-4o-mini"
            mock_settings.app_log_level = "INFO"
            mock_settings.noisy_log_level = "WARNING"

            from app.agents.orchestrator import ResearchOrchestrator

            orchestrator = ResearchOrchestrator()

            results = [
                {"url": "https://example.com/a", "score": 0.9},
                {"url": "https://example.com/a", "score": 0.8},  # duplicate
                {"url": "https://example.com/b", "score": 0.7},
            ]

            urls = await orchestrator._select_top_urls(results, max_urls=5)
            assert len(urls) == 2
            assert urls[0] == "https://example.com/a"  # highest score
            assert urls[1] == "https://example.com/b"

    @pytest.mark.asyncio
    async def test_select_top_urls_prioritizes_wikipedia_for_scrape(self):
        with patch("app.config.settings") as mock_settings:
            mock_settings.openrouter_api_key = "test"
            mock_settings.openrouter_model = ""
            mock_settings.default_model = "openai/gpt-4o-mini"
            mock_settings.app_log_level = "INFO"
            mock_settings.noisy_log_level = "WARNING"

            from app.agents.orchestrator import ResearchOrchestrator

            orchestrator = ResearchOrchestrator()

            results = [
                {"url": "https://news.example.com/a", "score": 0.95},
                {"url": "https://blog.example.com/b", "score": 0.85},
                {"url": "https://en.wikipedia.org/wiki/Roar_(musician)", "score": 0.10},
            ]

            urls = await orchestrator._select_top_urls(results, max_urls=2)
            assert len(urls) == 2
            assert any("wikipedia.org" in u for u in urls)

    def test_promote_full_name_mentions_expands_short_name(self):
        with patch("app.config.settings") as mock_settings:
            mock_settings.openrouter_api_key = "test"
            mock_settings.openrouter_model = ""
            mock_settings.default_model = "openai/gpt-4o-mini"
            mock_settings.app_log_level = "INFO"
            mock_settings.noisy_log_level = "WARNING"

            from app.agents.orchestrator import ResearchOrchestrator

            orchestrator = ResearchOrchestrator()
            report = "The artist Alex Smith released two songs that went viral."
            upgraded = orchestrator._promote_full_name_mentions(
                report,
                ["Alex Jordan Smith"],
            )
            assert "Alex Jordan Smith (Alex Smith)" in upgraded

    def test_extract_title_case_phrases_keeps_single_word_musician_entity(self):
        with patch("app.config.settings") as mock_settings:
            mock_settings.openrouter_api_key = "test"
            mock_settings.openrouter_model = ""
            mock_settings.default_model = "openai/gpt-4o-mini"
            mock_settings.app_log_level = "INFO"
            mock_settings.noisy_log_level = "WARNING"

            from app.agents.orchestrator import ResearchOrchestrator

            phrases = ResearchOrchestrator._extract_title_case_phrases(
                "Roar (musician) - Wikipedia"
            )
            assert "Roar" in phrases

    def test_build_chain_bootstrap_queries_filters_generic_labels(self):
        with patch("app.config.settings") as mock_settings:
            mock_settings.openrouter_api_key = "test"
            mock_settings.openrouter_model = ""
            mock_settings.default_model = "openai/gpt-4o-mini"
            mock_settings.app_log_level = "INFO"
            mock_settings.noisy_log_level = "WARNING"

            from app.agents.orchestrator import ResearchOrchestrator

            orchestrator = ResearchOrchestrator()
            query = (
                'Name two albums by the artist who played keyboard in a band that won '
                '"Best New Artist" in 2018.'
            )
            search_results = [
                {
                    "title": "Ada Stone (musician) - Wikipedia",
                    "content": "American musician and songwriter.",
                    "url": "https://en.wikipedia.org/wiki/Ada_Stone_(musician)",
                },
                {
                    "title": "Best New Artist | City Music Awards 2024",
                    "content": "Local nightlife category winner",
                    "url": "https://example.com/awards/best-new-artist-2024",
                },
            ]
            queries = orchestrator._build_chain_bootstrap_queries(query, search_results)

            assert any("Ada Stone" in q for q in queries)
            assert not any("Best New Artist" in q for q in queries)

    def test_extract_band_candidates_from_scraped_parses_heading_block(self):
        with patch("app.config.settings") as mock_settings:
            mock_settings.openrouter_api_key = "test"
            mock_settings.openrouter_model = ""
            mock_settings.default_model = "openai/gpt-4o-mini"
            mock_settings.app_log_level = "INFO"
            mock_settings.noisy_log_level = "WARNING"

            from app.agents.orchestrator import ResearchOrchestrator

            orchestrator = ResearchOrchestrator()
            query = 'Find the group that won "Best New Artist" in 2018.'
            scraped = {
                "https://example.com/awards/2018": (
                    ("N" * 9200)
                    + "\n### BEST NEW ARTIST\n\n## The Midnight Rails\n"
                    + "\nShare BEST NEW ARTIST\n"
                )
            }

            candidates = orchestrator._extract_band_candidates_from_scraped(query, scraped)
            assert "The Midnight Rails" in candidates

    def test_build_chain_enrichment_queries_prioritizes_scraped_band_candidates(self):
        with patch("app.config.settings") as mock_settings:
            mock_settings.openrouter_api_key = "test"
            mock_settings.openrouter_model = ""
            mock_settings.default_model = "openai/gpt-4o-mini"
            mock_settings.app_log_level = "INFO"
            mock_settings.noisy_log_level = "WARNING"

            from app.agents.orchestrator import ResearchOrchestrator

            orchestrator = ResearchOrchestrator()
            query = (
                'Name two songs by the artist who played drums in a band that won '
                '"Best New Artist" in 2018.'
            )

            with (
                patch.object(
                    orchestrator,
                    "_extract_band_candidates_from_scraped",
                    return_value=["Andrew Jackson Jihad"],
                ),
                patch.object(
                    orchestrator,
                    "_extract_band_candidates_from_results",
                    return_value=["Noisy Candidate"],
                ),
                patch.object(
                    orchestrator,
                    "_extract_person_candidates_from_content",
                    return_value=[],
                ),
            ):
                queries = orchestrator._build_chain_enrichment_queries(query, [], {})

            assert queries
            assert any("Andrew Jackson Jihad" in q for q in queries)
            assert not any("joined 2016 left 2021" in q for q in queries)
            assert not any("Roar" in q for q in queries)


class TestPromptBookending:
    def test_bookend_query_context_query_pattern(self):
        from app.agents import orchestrator as orch_module

        with (
            patch.object(orch_module.settings, "prompt_bookend_enabled", True),
            patch.object(orch_module.settings, "prompt_bookend_pattern", "query_context_query"),
        ):
            orchestrator = orch_module.ResearchOrchestrator()
            wrapped = orchestrator._bookend_query_context("Find X", "Evidence about X")

            assert wrapped.count("<query>") == 2
            assert wrapped.count("<context>") == 1
            assert wrapped.index("<query>") < wrapped.index("<context>")

    def test_build_synthesis_messages_collapses_to_single_user_message_when_bookended(self):
        from app.agents import orchestrator as orch_module

        with (
            patch.object(orch_module.settings, "prompt_bookend_enabled", True),
            patch.object(orch_module.settings, "prompt_bookend_pattern", "query_context_query"),
        ):
            orchestrator = orch_module.ResearchOrchestrator()
            messages = orchestrator._build_synthesis_messages(
                "Only provide country names.",
                "Source A says ... Source B says ...",
            )

            assert len(messages) == 1
            assert messages[0]["role"] == "user"
            content = messages[0]["content"]
            assert content.count("<query>") >= 2
            assert "<context>" in content


