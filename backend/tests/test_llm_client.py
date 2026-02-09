"""Tests for LLM client factory (Anthropic vs OpenRouter)."""
import pytest
from unittest.mock import patch, MagicMock
from app.llm_client import get_client, get_model


class TestGetModel:
    """Test model selection logic."""

    def test_get_model_returns_default_when_no_openrouter(self):
        """When OpenRouter is not configured, return default Anthropic model."""
        with patch("app.llm_client.settings") as mock_settings:
            mock_settings.openrouter_model = ""
            mock_settings.openrouter_api_key = ""
            mock_settings.default_model = "claude-sonnet-4-5-20250929"

            result = get_model()
            assert result == "claude-sonnet-4-5-20250929"

    def test_get_model_returns_openrouter_when_configured(self):
        """When OpenRouter is configured, return OpenRouter model ID."""
        with patch("app.llm_client.settings") as mock_settings:
            mock_settings.openrouter_model = "openai/gpt-4"
            mock_settings.openrouter_api_key = "sk-or-valid-key"
            mock_settings.default_model = "claude-sonnet-4-5-20250929"

            result = get_model()
            assert result == "openai/gpt-4"

    def test_get_model_ignores_openrouter_model_without_api_key(self):
        """If OpenRouter model is set but API key is missing, use default."""
        with patch("app.llm_client.settings") as mock_settings:
            mock_settings.openrouter_model = "openai/gpt-4"
            mock_settings.openrouter_api_key = ""
            mock_settings.default_model = "claude-sonnet-4-5-20250929"

            result = get_model()
            assert result == "claude-sonnet-4-5-20250929"

    def test_get_model_ignores_openrouter_api_key_without_model(self):
        """If API key is set but OpenRouter model is missing, use default."""
        with patch("app.llm_client.settings") as mock_settings:
            mock_settings.openrouter_model = ""
            mock_settings.openrouter_api_key = "sk-or-valid-key"
            mock_settings.default_model = "claude-sonnet-4-5-20250929"

            result = get_model()
            assert result == "claude-sonnet-4-5-20250929"

    def test_get_model_supports_various_openrouter_models(self):
        """OpenRouter model selection works with different model IDs."""
        model_ids = [
            "openai/gpt-4",
            "openai/gpt-3.5-turbo",
            "anthropic/claude-3-opus",
            "google/gemini-pro",
            "meta-llama/llama-2-70b-chat",
        ]

        for model_id in model_ids:
            with patch("app.llm_client.settings") as mock_settings:
                mock_settings.openrouter_model = model_id
                mock_settings.openrouter_api_key = "sk-or-valid-key"
                mock_settings.default_model = "claude-sonnet-4-5-20250929"

                result = get_model()
                assert result == model_id


class TestGetClient:
    """Test client initialization logic."""

    def test_get_client_returns_anthropic_by_default(self):
        """Without OpenRouter config, return Anthropic AsyncAnthropic client."""
        with patch("app.llm_client.settings") as mock_settings:
            mock_settings.openrouter_model = ""
            mock_settings.openrouter_api_key = ""
            mock_settings.anthropic_api_key = "sk-ant-test-key"

            with patch("anthropic.AsyncAnthropic") as mock_anthropic:
                get_client()
                # Verify called with Anthropic API key, no base_url override
                mock_anthropic.assert_called_once_with(api_key="sk-ant-test-key")

    def test_get_client_returns_openrouter_when_configured(self):
        """With OpenRouter config, return client pointing to OpenRouter base_url."""
        with patch("app.llm_client.settings") as mock_settings:
            mock_settings.openrouter_model = "openai/gpt-4"
            mock_settings.openrouter_api_key = "sk-or-valid-key"
            mock_settings.anthropic_api_key = "sk-ant-test-key"

            with patch("anthropic.AsyncAnthropic") as mock_anthropic:
                get_client()
                # Verify called with OpenRouter API key and base_url
                mock_anthropic.assert_called_once_with(
                    api_key="sk-or-valid-key",
                    base_url="https://openrouter.io/api/v1",
                )

    def test_get_client_ignores_openrouter_without_both_settings(self):
        """If either OpenRouter setting is missing, fall back to Anthropic."""
        test_cases = [
            ("openai/gpt-4", ""),  # Model but no key
            ("", "sk-or-valid-key"),  # Key but no model
        ]

        for model, key in test_cases:
            with patch("app.llm_client.settings") as mock_settings:
                mock_settings.openrouter_model = model
                mock_settings.openrouter_api_key = key
                mock_settings.anthropic_api_key = "sk-ant-test-key"

                with patch("anthropic.AsyncAnthropic") as mock_anthropic:
                    get_client()
                    # Should always use Anthropic API key, no base_url
                    mock_anthropic.assert_called_once_with(api_key="sk-ant-test-key")

    def test_get_client_uses_openrouter_base_url_correctly(self):
        """OpenRouter client uses correct API endpoint."""
        with patch("app.llm_client.settings") as mock_settings:
            mock_settings.openrouter_model = "anthropic/claude-3-opus"
            mock_settings.openrouter_api_key = "sk-or-test-key"
            mock_settings.anthropic_api_key = "sk-ant-test-key"

            with patch("anthropic.AsyncAnthropic") as mock_anthropic:
                get_client()
                call_kwargs = mock_anthropic.call_args[1]
                assert call_kwargs["base_url"] == "https://openrouter.io/api/v1"
