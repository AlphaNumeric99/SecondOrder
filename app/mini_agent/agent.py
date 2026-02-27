import asyncio
from pathlib import Path
from typing import Optional

from loguru import logger

from app.mini_agent.llm import LLMClient
from app.mini_agent.schema import Message
from app.mini_agent.tools import Tool, ToolResult


class Agent:
    def __init__(
        self,
        llm_client: LLMClient,
        system_prompt: str,
        tools: list[Tool],
        max_steps: int = 50,
        workspace_dir: str = "./workspace",
    ):
        self.llm = llm_client
        self.tools = {tool.name: tool for tool in tools}
        self.max_steps = max_steps
        self.workspace_dir = Path(workspace_dir)
        self.cancel_event: Optional[asyncio.Event] = None
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        if "Current Workspace" not in system_prompt:
            system_prompt = (
                system_prompt
                + f"\n\n## Current Workspace\nWorking in: `{self.workspace_dir.absolute()}`"
            )

        self.system_prompt = system_prompt
        self.messages: list[Message] = [Message(role="system", content=system_prompt)]

    def add_user_message(self, content: str):
        self.messages.append(Message(role="user", content=content))

    def _check_cancelled(self) -> bool:
        if self.cancel_event is not None and self.cancel_event.is_set():
            return True
        return False

    async def run(self, cancel_event: Optional[asyncio.Event] = None) -> str:
        if cancel_event is not None:
            self.cancel_event = cancel_event

        step = 0
        while step < self.max_steps:
            if self._check_cancelled():
                return "Task cancelled by user."

            tool_list = list(self.tools.values())

            try:
                response = await self.llm.generate(
                    messages=self.messages, tools=tool_list
                )
            except Exception as e:
                logger.error(f"LLM call failed: {e}")
                return f"Error: {str(e)}"

            if not response.tool_calls:
                return response.content

            self.messages.append(
                Message(
                    role="assistant",
                    content=response.content,
                    tool_calls=response.tool_calls,
                )
            )

            for tool_call in response.tool_calls:
                tool_name = tool_call.function.name
                arguments = tool_call.function.arguments

                if tool_name not in self.tools:
                    result = ToolResult(
                        success=False,
                        content="",
                        error=f"Unknown tool: {tool_name}",
                    )
                else:
                    try:
                        tool = self.tools[tool_name]
                        result = await tool.execute(**arguments)
                    except Exception as e:
                        result = ToolResult(
                            success=False,
                            content="",
                            error=f"Tool execution failed: {str(e)}",
                        )

                self.messages.append(
                    Message(
                        role="tool",
                        content=result.content
                        if result.success
                        else f"Error: {result.error}",
                        tool_call_id=tool_call.id,
                        name=tool_name,
                    )
                )

            step += 1

        return f"Max steps ({self.max_steps}) reached."

    def get_history(self) -> list[Message]:
        return self.messages.copy()
