import json
from typing import Any

from openai import AsyncOpenAI

from app.mini_agent.schema import (
    FunctionCall,
    LLMProvider,
    LLMResponse,
    Message,
    TokenUsage,
    ToolCall,
)


class LLMClient:
    def __init__(
        self,
        api_key: str,
        api_base: str = "https://openrouter.ai/api/v1",
        model: str = "openai/gpt-4o-mini",
    ):
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.model = model
        self._client = AsyncOpenAI(api_key=api_key, base_url=self.api_base)

    async def generate(
        self,
        messages: list[Message],
        tools: list | None = None,
    ) -> LLMResponse:
        openai_messages = self._to_openai_messages(messages)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": openai_messages,
            "max_tokens": 8192,
        }

        if tools:
            kwargs["tools"] = self._to_openai_tools(tools)
            kwargs["tool_choice"] = "auto"

        response = await self._client.chat.completions.create(**kwargs)
        return self._from_openai_response(response)

    def _to_openai_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == "system":
                result.append({"role": "system", "content": msg.content})
            elif msg.role == "user":
                if isinstance(msg.content, str):
                    result.append({"role": "user", "content": msg.content})
                elif isinstance(msg.content, list):
                    for block in msg.content:
                        if isinstance(block, dict):
                            btype = block.get("type")
                            if btype == "text":
                                result.append(
                                    {"role": "user", "content": block.get("text", "")}
                                )
                            elif btype == "tool_use":
                                continue
                continue

            elif msg.role == "assistant":
                if msg.tool_calls:
                    assistant_msg: dict[str, Any] = {
                        "role": "assistant",
                        "content": msg.content if isinstance(msg.content, str) else "",
                    }
                    assistant_msg["tool_calls"] = []
                    for tc in msg.tool_calls:
                        assistant_msg["tool_calls"].append(
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": json.dumps(tc.function.arguments),
                                },
                            }
                        )
                    result.append(assistant_msg)
                elif msg.content:
                    result.append({"role": "assistant", "content": msg.content})
                continue

            elif msg.role == "tool":
                tool_content = str(msg.content)
                result.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.tool_call_id or "",
                        "content": tool_content,
                    }
                )
                continue

        return result

    def _to_openai_tools(self, tools: list) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in tools
        ]

    def _from_openai_response(self, response: Any) -> LLMResponse:
        choice = response.choices[0].message
        content = ""
        tool_calls: list[ToolCall] | None = None

        if choice.content:
            content = choice.content

        if getattr(choice, "tool_calls", None):
            tool_calls = []
            for tc in choice.tool_calls:
                args = json.loads(tc.function.arguments or "{}")
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        type="function",
                        function=FunctionCall(name=tc.function.name, arguments=args),
                    )
                )

        usage = response.usage
        token_usage = None
        if usage:
            token_usage = TokenUsage(
                prompt_tokens=usage.prompt_tokens or 0,
                completion_tokens=usage.completion_tokens or 0,
                total_tokens=usage.total_tokens or 0,
            )

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            usage=token_usage,
        )
