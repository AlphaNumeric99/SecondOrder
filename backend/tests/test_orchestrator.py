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
    async def test_generate_plan_augments_short_chain_query(self):
        with patch("app.config.settings") as mock_settings:
            mock_settings.openrouter_api_key = "test"
            mock_settings.openrouter_model = ""
            mock_settings.default_model = "openai/gpt-4o-mini"

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
            assert len(plan) >= 3
            assert any("wikipedia" in step.lower() for step in plan)
            assert query in plan


class TestURLSelection:
    @pytest.mark.asyncio
    async def test_select_top_urls_deduplicates(self):
        with patch("app.config.settings") as mock_settings:
            mock_settings.openrouter_api_key = "test"
            mock_settings.openrouter_model = ""
            mock_settings.default_model = "openai/gpt-4o-mini"

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

            from app.agents.orchestrator import ResearchOrchestrator

            orchestrator = ResearchOrchestrator()
            report = "The artist Alex Smith released two songs that went viral."
            upgraded = orchestrator._promote_full_name_mentions(
                report,
                ["Alex Jordan Smith"],
            )
            assert "Alex Jordan Smith (Alex Smith)" in upgraded
