"""Tests for the research orchestrator."""
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from app.models.events import EventType


class TestOrchestratorPlan:
    """Test plan generation logic."""

    @pytest.mark.asyncio
    async def test_generate_plan_parses_json(self):
        with patch("app.config.settings") as mock_settings:
            mock_settings.anthropic_api_key = "test"
            mock_settings.default_model = "claude-sonnet-4-5-20250929"

            from app.agents.orchestrator import ResearchOrchestrator

            orchestrator = ResearchOrchestrator()

            # Mock the Anthropic client
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
            mock_settings.anthropic_api_key = "test"
            mock_settings.default_model = "claude-sonnet-4-5-20250929"

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
            assert plan == ["test query"]  # Falls back to original query


class TestURLSelection:
    @pytest.mark.asyncio
    async def test_select_top_urls_deduplicates(self):
        with patch("app.config.settings") as mock_settings:
            mock_settings.anthropic_api_key = "test"
            mock_settings.default_model = "claude-sonnet-4-5-20250929"

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
