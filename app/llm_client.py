"""OpenRouter-only LLM client factory with legacy message adapter methods."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, AsyncIterator

from app.config import settings
from app.services.env_safety import sanitize_ssl_keylogfile


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class TextBlock:
    type: str
    text: str


@dataclass
class ToolUseBlock:
    type: str
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class MessageResponse:
    content: list[Any]
    usage: Usage


class OpenRouterStream:
    def __init__(self, stream_coro: Any):
        self._stream_coro = stream_coro
        self._stream: Any | None = None
        self._usage = Usage()
        self._finished = False

    async def __aenter__(self) -> "OpenRouterStream":
        self._stream = await self._stream_coro
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._stream is not None:
            await self._stream.close()

    async def _iter_text(self) -> AsyncIterator[str]:
        if self._stream is None:
            return
        async for chunk in self._stream:
            choices = getattr(chunk, "choices", None) or []
            usage = getattr(chunk, "usage", None)
            if usage:
                self._usage = Usage(
                    input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                    output_tokens=getattr(usage, "completion_tokens", 0) or 0,
                )

            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            if not delta:
                continue
            text = getattr(delta, "content", None)
            if text:
                yield text
        self._finished = True

    @property
    def text_stream(self) -> AsyncIterator[str]:
        return self._iter_text()

    async def get_final_message(self) -> MessageResponse:
        if not self._finished:
            async for _ in self.text_stream:
                pass
        return MessageResponse(content=[], usage=self._usage)


class OpenRouterMessagesAdapter:
    def __init__(self, openai_client: Any):
        self._client = openai_client

    @staticmethod
    def _temperature_for_model(model: str) -> int:
        # Some OpenAI GPT-5-compatible gateways reject temperature=0.
        lowered = (model or "").lower()
        if "gpt-5" in lowered:
            return 1
        return 0

    def _to_openai_messages(self, system: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        openai_messages: list[dict[str, Any]] = [{"role": "system", "content": system}]

        for message in messages:
            role = message["role"]
            content = message["content"]

            if isinstance(content, str):
                openai_messages.append({"role": role, "content": content})
                continue

            if role == "assistant" and isinstance(content, list):
                text_parts: list[str] = []
                tool_calls: list[dict[str, Any]] = []
                for block in content:
                    btype = getattr(block, "type", None) if not isinstance(block, dict) else block.get("type")
                    if btype == "text":
                        text_value = getattr(block, "text", None) if not isinstance(block, dict) else block.get("text")
                        if text_value:
                            text_parts.append(text_value)
                    elif btype == "tool_use":
                        bid = getattr(block, "id", None) if not isinstance(block, dict) else block.get("id")
                        bname = getattr(block, "name", None) if not isinstance(block, dict) else block.get("name")
                        binput = getattr(block, "input", None) if not isinstance(block, dict) else block.get("input")
                        tool_calls.append(
                            {
                                "id": bid,
                                "type": "function",
                                "function": {"name": bname, "arguments": json.dumps(binput or {})},
                            }
                        )
                msg: dict[str, Any] = {"role": "assistant"}
                msg["content"] = "\n".join(text_parts) if text_parts else None
                if tool_calls:
                    msg["tool_calls"] = tool_calls
                openai_messages.append(msg)
                continue

            if role == "user" and isinstance(content, list):
                for tool_result in content:
                    if tool_result.get("type") != "tool_result":
                        continue
                    tool_content = str(tool_result.get("content", ""))
                    if tool_result.get("is_error"):
                        tool_content = f"ERROR: {tool_content}"
                    openai_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_result.get("tool_use_id", ""),
                            "content": tool_content,
                        }
                    )
                continue

            openai_messages.append({"role": role, "content": str(content)})

        return openai_messages

    def _to_openai_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
                },
            }
            for t in tools
        ]

    def _from_openai_response(self, response: Any) -> MessageResponse:
        import json

        choice = response.choices[0].message
        content: list[Any] = []

        text = getattr(choice, "content", None)
        if text:
            content.append(TextBlock(type="text", text=text))

        for tc in getattr(choice, "tool_calls", []) or []:
            args = getattr(tc.function, "arguments", "{}") or "{}"
            try:
                parsed_args = json.loads(args)
            except json.JSONDecodeError:
                parsed_args = {}
            content.append(
                ToolUseBlock(
                    type="tool_use",
                    id=tc.id,
                    name=tc.function.name,
                    input=parsed_args,
                )
            )

        usage = getattr(response, "usage", None)
        mapped_usage = Usage(
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )

        return MessageResponse(content=content, usage=mapped_usage)

    async def create(
        self,
        *,
        model: str,
        max_tokens: int,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> MessageResponse:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": self._to_openai_messages(system, messages),
            "max_tokens": max_tokens,
            "temperature": self._temperature_for_model(model),
        }
        if tools:
            kwargs["tools"] = self._to_openai_tools(tools)
            kwargs["tool_choice"] = "auto"

        response = await self._client.chat.completions.create(**kwargs)
        return self._from_openai_response(response)

    def stream(
        self,
        *,
        model: str,
        max_tokens: int,
        system: str,
        messages: list[dict[str, Any]],
    ) -> OpenRouterStream:
        stream = self._client.chat.completions.create(
            model=model,
            messages=self._to_openai_messages(system, messages),
            max_tokens=max_tokens,
            temperature=self._temperature_for_model(model),
            stream=True,
            stream_options={"include_usage": True},
        )
        return OpenRouterStream(stream)


class OpenRouterClientAdapter:
    def __init__(self, openai_client: Any):
        self.messages = OpenRouterMessagesAdapter(openai_client)


def get_client() -> OpenRouterClientAdapter:
    """Get OpenRouter client via OpenAI-compatible SDK."""
    from openai import AsyncOpenAI

    sanitize_ssl_keylogfile()
    base_url = settings.openrouter_base_url.strip() or "https://openrouter.ai/api/v1"
    openai_client = AsyncOpenAI(
        api_key=settings.openrouter_api_key,
        base_url=base_url,
    )
    return OpenRouterClientAdapter(openai_client)


def get_model() -> str:
    """Get the active OpenRouter model id."""
    if settings.openrouter_model:
        return settings.openrouter_model
    return settings.default_model


_client: OpenRouterClientAdapter | None = None


def client() -> OpenRouterClientAdapter:
    """Get or create the LLM client."""
    global _client
    if _client is None:
        _client = get_client()
    return _client
