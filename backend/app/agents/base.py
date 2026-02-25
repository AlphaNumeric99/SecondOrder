from __future__ import annotations

import time
from typing import Any, AsyncGenerator

from app.llm_client import client as llm_client, get_model
from app.models.events import SSEEvent
from app.services import logger as log_service
from app.services.prompt_store import render_prompt


class BaseAgent:
    """Base agent that wraps the OpenRouter tool-use loop.

    Subclasses define `system_prompt`, `tools`, and `handle_tool_call`.
    The `run` method is an async generator that yields SSEEvents as the agent works.
    """

    name: str = "base"
    system_prompt: str = render_prompt("base.system_prompt")
    tools: list[dict[str, Any]] = []

    def __init__(self, model: str | None = None, session_id: str | None = None):
        self.model = model or get_model()
        self.session_id = session_id
        self.client = None

    async def handle_tool_call(
        self, tool_name: str, tool_input: dict[str, Any]
    ) -> tuple[str, list[SSEEvent]]:
        """Execute a tool call and return (result_text, events_to_emit).

        Must be overridden by subclasses that define tools.
        """
        raise NotImplementedError(f"Tool {tool_name} not handled")

    async def run(
        self,
        user_message: str,
        *,
        context: str = "",
        max_turns: int = 10,
    ) -> AsyncGenerator[SSEEvent, None]:
        """Run the agent's tool-use loop, yielding SSE events along the way.

        Returns final text response via the last event or can be collected.
        """
        messages: list[dict[str, Any]] = []

        if context:
            messages.append({"role": "user", "content": f"<context>\n{context}\n</context>"})
            messages.append(
                {
                    "role": "assistant",
                    "content": render_prompt("base.context_ack_prompt"),
                }
            )

        messages.append({"role": "user", "content": user_message})

        for _ in range(max_turns):
            active_client = self.client or llm_client()
            kwargs: dict[str, Any] = {
                "model": self.model,
                "max_tokens": 8192,
                "system": self.system_prompt,
                "messages": messages,
            }
            if self.tools:
                kwargs["tools"] = self.tools

            t0 = time.monotonic()
            response = await active_client.messages.create(**kwargs)
            elapsed_ms = int((time.monotonic() - t0) * 1000)

            # Log the LLM call
            try:
                from uuid import UUID
                from app.services import database as db

                usage = response.usage
                await db.log_llm_call(
                    model=self.model,
                    caller=self.name,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    duration_ms=elapsed_ms,
                    session_id=UUID(self.session_id) if self.session_id else None,
                )
                # Also log to file for debugging
                log_service.log_llm_call(
                    model=self.model,
                    caller=self.name,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    duration_ms=elapsed_ms,
                )
            except Exception as e:
                try:
                    log_service.log_event(
                        event_type="logging_error",
                        message=f"Failed to log LLM call in {self.name}",
                        error=str(e),
                        model=self.model,
                    )
                except Exception:
                    # Never break agent execution because logging failed.
                    pass

            # Check if the response contains tool use
            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            text_blocks = [b for b in response.content if b.type == "text"]

            if not tool_use_blocks:
                # No tools called — agent is done
                final_text = "\n".join(b.text for b in text_blocks)
                self._final_text = final_text
                return

            # Process tool calls
            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for tool_block in tool_use_blocks:
                try:
                    result_text, events = await self.handle_tool_call(
                        tool_block.name, tool_block.input
                    )
                    for event in events:
                        yield event
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_block.id,
                        "content": result_text,
                    })
                except Exception as e:
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_block.id,
                        "content": f"Error: {e}",
                        "is_error": True,
                    })

            messages.append({"role": "user", "content": tool_results})

        self._final_text = "Max turns reached."

    @property
    def final_text(self) -> str:
        return getattr(self, "_final_text", "")

    async def run_to_completion(
        self,
        user_message: str,
        *,
        context: str = "",
        max_turns: int = 10,
    ) -> tuple[str, list[SSEEvent]]:
        """Convenience: run agent collecting all events, return (final_text, all_events)."""
        events: list[SSEEvent] = []
        async for event in self.run(user_message, context=context, max_turns=max_turns):
            events.append(event)
        return self.final_text, events
