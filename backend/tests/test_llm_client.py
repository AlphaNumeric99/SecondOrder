"""Tests for OpenRouter-only LLM client factory."""
import sys
import types
from unittest.mock import MagicMock, patch

from app.llm_client import get_client, get_model


class TestGetModel:
    """Test model selection logic."""

    def test_get_model_returns_default_when_no_override(self):
        with patch("app.llm_client.settings") as mock_settings:
            mock_settings.openrouter_model = ""
            mock_settings.default_model = "openai/gpt-4o-mini"

            assert get_model() == "openai/gpt-4o-mini"

    def test_get_model_returns_openrouter_override(self):
        with patch("app.llm_client.settings") as mock_settings:
            mock_settings.openrouter_model = "openai/gpt-4.1"
            mock_settings.default_model = "openai/gpt-4o-mini"

            assert get_model() == "openai/gpt-4.1"

    def test_get_model_supports_various_openrouter_models(self):
        model_ids = [
            "openai/gpt-4.1",
            "openai/gpt-4o-mini",
            "google/gemini-2.0-flash-001",
            "meta-llama/llama-3.3-70b-instruct",
        ]

        for model_id in model_ids:
            with patch("app.llm_client.settings") as mock_settings:
                mock_settings.openrouter_model = model_id
                mock_settings.default_model = "openai/gpt-4o-mini"
                assert get_model() == model_id


class TestGetClient:
    """Test OpenRouter client initialization."""

    def test_get_client_uses_openrouter(self):
        with patch("app.llm_client.settings") as mock_settings:
            mock_settings.openrouter_api_key = "sk-or-valid-key"

            openai_module = types.ModuleType("openai")
            mock_openai = MagicMock()
            openai_module.AsyncOpenAI = mock_openai

            with patch.dict(sys.modules, {"openai": openai_module}):
                get_client()

            mock_openai.assert_called_once_with(
                api_key="sk-or-valid-key",
                base_url="https://openrouter.ai/api/v1",
            )
