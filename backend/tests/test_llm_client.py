"""Tests for OpenRouter-only LLM client factory."""
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.llm_client import OpenRouterMessagesAdapter, OpenRouterStream, get_client, get_model


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
            mock_settings.openrouter_base_url = "https://openrouter.ai/api/v1"

            openai_module = types.ModuleType("openai")
            mock_openai = MagicMock()
            openai_module.AsyncOpenAI = mock_openai

            with patch.dict(sys.modules, {"openai": openai_module}):
                get_client()

            mock_openai.assert_called_once_with(
                api_key="sk-or-valid-key",
                base_url="https://openrouter.ai/api/v1",
            )


class TestOpenRouterAdapter:
    def test_to_openai_messages_maps_tool_results(self):
        adapter = OpenRouterMessagesAdapter(openai_client=MagicMock())
        messages = [
            {"role": "user", "content": "find sources"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I'll use a tool."},
                    {"type": "tool_use", "id": "tool_1", "name": "web_search", "input": {"query": "ai"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "tool_1", "content": "ok"},
                    {"type": "tool_result", "tool_use_id": "tool_2", "content": "failed", "is_error": True},
                ],
            },
        ]

        mapped = adapter._to_openai_messages("sys", messages)

        assert mapped[0] == {"role": "system", "content": "sys"}
        assert mapped[2]["role"] == "assistant"
        assert mapped[2]["tool_calls"][0]["function"]["name"] == "web_search"
        assert mapped[3] == {"role": "tool", "tool_call_id": "tool_1", "content": "ok"}
        assert mapped[4] == {"role": "tool", "tool_call_id": "tool_2", "content": "ERROR: failed"}

    def test_from_openai_response_maps_text_tool_calls_and_usage(self):
        adapter = OpenRouterMessagesAdapter(openai_client=MagicMock())
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="Done.",
                        tool_calls=[
                            SimpleNamespace(
                                id="call_1",
                                function=SimpleNamespace(name="scrape_url", arguments='{"url":"https://a.com"}'),
                            ),
                            SimpleNamespace(
                                id="call_2",
                                function=SimpleNamespace(name="scrape_url", arguments="{invalid"),
                            ),
                        ],
                    )
                )
            ],
            usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7),
        )

        mapped = adapter._from_openai_response(response)

        assert mapped.usage.input_tokens == 11
        assert mapped.usage.output_tokens == 7
        assert mapped.content[0].type == "text"
        assert mapped.content[0].text == "Done."
        assert mapped.content[1].type == "tool_use"
        assert mapped.content[1].input == {"url": "https://a.com"}
        assert mapped.content[2].input == {}


class TestOpenRouterStream:
    @pytest.mark.asyncio
    async def test_stream_yields_text_and_maps_usage(self):
        async def chunk_iter():
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="Hello "))],
                usage=None,
            )
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="world"))],
                usage=SimpleNamespace(prompt_tokens=12, completion_tokens=9),
            )

        class FakeStream:
            def __aiter__(self):
                return chunk_iter()

            async def close(self):
                return None

        async def fake_stream_coro():
            return FakeStream()

        stream = OpenRouterStream(fake_stream_coro())
        async with stream as s:
            chunks = []
            async for text in s.text_stream:
                chunks.append(text)
            final_msg = await s.get_final_message()

        assert "".join(chunks) == "Hello world"
        assert final_msg.usage.input_tokens == 12
        assert final_msg.usage.output_tokens == 9
