from enum import Enum
from typing import Any

from pydantic import BaseModel


class LLMProvider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


class FunctionCall(BaseModel):
    name: str
    arguments: dict[str, Any]


class ToolCall(BaseModel):
    id: str
    type: str
    function: FunctionCall


class Message(BaseModel):
    role: str
    content: str | list[dict[str, Any]]
    thinking: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LLMResponse(BaseModel):
    content: str
    thinking: str | None = None
    tool_calls: list[ToolCall] | None = None
    finish_reason: str
    usage: TokenUsage | None = None
