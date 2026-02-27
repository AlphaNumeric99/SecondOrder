from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator

from loguru import logger

from app.agents.mini_agent_tools import get_research_tools
from app.config import settings
from app.llm_client import client as llm_client, get_model
from app.mini_agent import Agent, LLMClient as MiniAgentLLMClient
from app.mini_agent.tools import Tool
from app.mini_agent.tools.mcp_loader import load_mcp_tools_async
from app.models.events import SSEEvent
from app.services import streaming


@dataclass
class AgentTask:
    id: str
    query: str
    description: str
    dependencies: list[str] = field(default_factory=list)


@dataclass
class AgentResult:
    task_id: str
    success: bool
    result: str
    error: str | None = None


class ParallelAgentOrchestrator:
    def __init__(
        self,
        model: str | None = None,
        session_id: str | None = None,
        workspace_dir: str = "./workspace",
        max_steps_per_agent: int = 100,
        mcp_config_path: str | None = None,
    ):
        self.model = model or get_model()
        self.session_id = session_id
        self.workspace_dir = workspace_dir
        self.max_steps_per_agent = max_steps_per_agent
        self.mcp_config_path = mcp_config_path
        self._llm_client = None
        self._mcp_tools: list[Tool] = []

    async def _load_mcp_tools(self) -> list[Tool]:
        if self._mcp_tools:
            return self._mcp_tools
        if self.mcp_config_path:
            self._mcp_tools = await load_mcp_tools_async(self.mcp_config_path)
        return self._mcp_tools

    @property
    def llm_client(self):
        if self._llm_client is None:
            self._llm_client = llm_client()
        return self._llm_client

    def _get_mini_agent_llm_client(self) -> MiniAgentLLMClient:
        return MiniAgentLLMClient(
            api_key=settings.openrouter_api_key,
            api_base=settings.openrouter_base_url or "https://openrouter.ai/api/v1",
            model=self.model,
        )

    async def _decompose_query(self, query: str) -> list[AgentTask]:
        decomposition_prompt = f"""You are a task decomposition assistant. Given a complex research query, break it down into independent sub-tasks that can be executed in parallel.

For each sub-task, provide:
1. A unique ID
2. The specific search/query to execute
3. A brief description of what information this task aims to find
4. Any dependencies on other tasks (usually none for parallel execution)

Query: "{query}"

Guidelines:
- Identify independent information needs that can be fetched in parallel
- Each sub-task should be self-contained
- Don't make tasks dependent unless absolutely necessary
- If the query is simple enough, return only 1 task
- Maximum 10 tasks

Return your response as a JSON array of task objects:
[
  {{"id": "task-1", "query": "...", "description": "...", "dependencies": []}},
  ...
]

Only return the JSON, no other text."""

        response = await self.llm_client.messages.create(
            model=self.model,
            max_tokens=4096,
            system="You are a task decomposition expert. Break down complex queries into parallel sub-tasks.",
            messages=[{"role": "user", "content": decomposition_prompt}],
        )

        text_blocks = [b.text for b in response.content if hasattr(b, "text")]
        text = "\n".join(text_blocks)

        try:
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

            tasks_data = json.loads(text)
            tasks = []
            for task_data in tasks_data:
                tasks.append(
                    AgentTask(
                        id=task_data.get("id", f"task-{len(tasks)}"),
                        query=task_data.get("query", ""),
                        description=task_data.get("description", ""),
                        dependencies=task_data.get("dependencies", []),
                    )
                )
            return tasks
        except (json.JSONDecodeError, IndexError) as e:
            logger.warning(f"Failed to parse task decomposition: {e}, using fallback")
            return [AgentTask(id="task-1", query=query, description="Main query")]

    async def _get_all_tools(self) -> list[Tool]:
        tools = get_research_tools()
        mcp_tools = await self._load_mcp_tools()
        tools.extend(mcp_tools)
        return tools

    def _create_agent_system_prompt(self, task: AgentTask) -> str:
        tool_names = ["web_search", "fetch_content"]
        if self._mcp_tools:
            tool_names.extend([t.name for t in self._mcp_tools])

        return f"""You are a research agent tasked with finding specific information.

Your Task:
- Query: {task.query}
- Description: {task.description}

Instructions:
1. Use web_search to search for relevant information
2. Use fetch_content to get detailed content from specific URLs
3. Use MCP tools if available for specialized tasks
4. Be thorough - search multiple sources if needed
5. Provide a clear, concise answer to your query

Available tools: {", ".join(tool_names)}

Current Workspace: {self.workspace_dir}
"""

    async def _run_single_agent(
        self,
        task: AgentTask,
        event_queue: asyncio.Queue[SSEEvent],
    ) -> AgentResult:
        try:
            await event_queue.put(
                streaming.agent_started("parallel", step=0, query=task.query)
            )

            mini_agent_llm = self._get_mini_agent_llm_client()
            tools = await self._get_all_tools()

            agent = Agent(
                llm_client=mini_agent_llm,
                system_prompt=self._create_agent_system_prompt(task),
                tools=tools,
                max_steps=self.max_steps_per_agent,
                workspace_dir=self.workspace_dir,
            )

            agent.add_user_message(
                f"Task: {task.query}\n\nPlease find the information requested and provide a clear answer."
            )

            result = await agent.run()

            await event_queue.put(streaming.agent_completed("parallel", step=0))

            return AgentResult(
                task_id=task.id,
                success=True,
                result=result,
            )

        except Exception as e:
            logger.error(f"Agent {task.id} failed: {e}")
            await event_queue.put(streaming.error(str(e), agent="parallel"))
            return AgentResult(
                task_id=task.id,
                success=False,
                result="",
                error=str(e),
            )

    async def _synthesize_results(
        self,
        query: str,
        task_results: list[AgentResult],
    ) -> str:
        synthesis_prompt = f"""You are a research synthesis assistant. Given a complex query and results from multiple research agents, provide a final comprehensive answer.

Original Query: {query}

Agent Results:
"""

        for result in task_results:
            if result.success:
                synthesis_prompt += (
                    f"\n\n--- Agent {result.task_id} ---\n{result.result}"
                )
            else:
                synthesis_prompt += (
                    f"\n\n--- Agent {result.task_id} (FAILED) ---\n{result.error}"
                )

        synthesis_prompt += """

Instructions:
1. Carefully analyze all agent results
2. Combine the information to answer the original query
3. If some agents failed, note that and work with available information
4. Provide a clear, well-structured final answer
5. Only include information that was actually found, don't hallucinate"""

        response = await self.llm_client.messages.create(
            model=self.model,
            max_tokens=8192,
            system="You are a research synthesis expert. Combine results from multiple agents into a coherent answer.",
            messages=[{"role": "user", "content": synthesis_prompt}],
        )

        text_blocks = [b.text for b in response.content if hasattr(b, "text")]
        return "\n".join(text_blocks)

    async def run(
        self,
        query: str,
    ) -> AsyncGenerator[SSEEvent, None]:
        yield streaming.agent_started("orchestrator", step=0, query=query)

        tasks = await self._decompose_query(query)
        logger.info(f"Decomposed query into {len(tasks)} tasks")

        if not tasks:
            tasks = [AgentTask(id="task-1", query=query, description="Main query")]

        yield streaming.message(f"Executing {len(tasks)} parallel tasks...")

        event_queue: asyncio.Queue[SSEEvent] = asyncio.Queue()

        async def run_all_agents():
            agent_tasks = [self._run_single_agent(task, event_queue) for task in tasks]
            results = await asyncio.gather(*agent_tasks, return_exceptions=True)
            return results

        results_or_exceptions = await run_all_agents()

        task_results: list[AgentResult] = []
        for i, result in enumerate(results_or_exceptions):
            if isinstance(result, Exception):
                task_results.append(
                    AgentResult(
                        task_id=tasks[i].id,
                        success=False,
                        result="",
                        error=str(result),
                    )
                )
            else:
                task_results.append(result)

        while not event_queue.empty():
            event = await event_queue.get()
            yield event

        yield streaming.message("Synthesizing results from all agents...")

        final_answer = await self._synthesize_results(query, task_results)

        yield streaming.message("Research complete!")

        yield SSEEvent(
            event="message",
            data={"content": final_answer, "type": "final"},
        )

    async def run_to_completion(self, query: str) -> tuple[str, list[SSEEvent]]:
        events: list[SSEEvent] = []
        async for event in self.run(query):
            events.append(event)
        final_content = ""
        for event in events:
            if event.event == "message":
                data = event.data
                if isinstance(data, dict) and data.get("type") == "final":
                    final_content = data.get("content", "")
                    break
        return final_content, events
