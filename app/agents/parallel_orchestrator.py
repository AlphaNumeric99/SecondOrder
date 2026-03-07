from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator

from loguru import logger

LLM_TIMEOUT_SECONDS = 45
LLM_MAX_RETRIES = 3
LLM_RETRY_DELAY = 2

from app.agents.mini_agent_tools import get_research_tools
from app.config import settings
from app.llm_client import client as llm_client, get_model
from app.mini_agent import Agent, LLMClient as MiniAgentLLMClient
from app.mini_agent.tools import Tool
from app.mini_agent.tools.mcp_loader import (
    cleanup_mcp_connections,
    load_mcp_tools_async,
)
from app.models.events import SSEEvent
from app.services import streaming


@dataclass
class AgentTask:
    id: str
    query: str
    description: str
    dependencies: list[str] = field(default_factory=list)
    output_schema: str = ""
    downstream_use: str = ""


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
        max_steps_per_agent: int = 15,
        agent_timeout_seconds: int = 120,
        mcp_config_path: str | None = None,
    ):
        self.model = model or get_model()
        self.session_id = session_id
        self.workspace_dir = workspace_dir
        self.max_steps_per_agent = max_steps_per_agent
        self.agent_timeout_seconds = agent_timeout_seconds
        self.mcp_config_path = mcp_config_path
        self._llm_client = None
        self._mcp_tools: list[Tool] = []
        self._mcp_loading_lock = asyncio.Lock()

    async def _load_mcp_tools(self) -> list[Tool]:
        if self._mcp_tools:
            return self._mcp_tools
        async with self._mcp_loading_lock:
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

    async def _llm_call(self, **kwargs) -> Any:
        last_error = None
        for attempt in range(LLM_MAX_RETRIES):
            try:
                async with asyncio.timeout(LLM_TIMEOUT_SECONDS):
                    return await self.llm_client.messages.create(**kwargs)
            except asyncio.TimeoutError:
                last_error = f"Timeout after {LLM_TIMEOUT_SECONDS}s"
                logger.warning(
                    f"LLM call timed out (attempt {attempt + 1}/{LLM_MAX_RETRIES})"
                )
            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"LLM call failed (attempt {attempt + 1}/{LLM_MAX_RETRIES}): {e}"
                )

            if attempt < LLM_MAX_RETRIES - 1:
                await asyncio.sleep(LLM_RETRY_DELAY)

        raise Exception(
            f"LLM call failed after {LLM_MAX_RETRIES} retries: {last_error}"
        )

    def _get_mini_agent_llm_client(self) -> MiniAgentLLMClient:
        return MiniAgentLLMClient(
            api_key=settings.openrouter_api_key,
            api_base=settings.openrouter_base_url or "https://openrouter.ai/api/v1",
            model=self.model,
        )

    async def _decompose_query(self, query: str) -> list[AgentTask]:
        decomposition_prompt = f"""You are a task decomposition assistant. Given a complex research query, break it down into sub-tasks that need to be executed SEQUENTIALLY.

For each sub-task, provide:
1. A unique ID
2. The specific search/query to execute
3. A brief description of what information this task aims to find
4. Dependencies on other tasks (if this task needs results from a previous task)
5. output_schema: What this task will return (e.g., "band name", "drummer name and years", "list of songs")
6. downstream_use: What downstream tasks will use this for (e.g., "to find band members")

Query: "{query}"

Guidelines:
- If later tasks need information from earlier tasks, create DEPENDENCIES
- Example: If task-2 needs the band name from task-1, task-2 should have "dependencies": ["task-1"]
- The LLM will summarize relevant parts before passing to downstream tasks
- Each task should build on previous results
- Maximum 10 tasks
- ALWAYS use web search or browser tools - do NOT rely on training memory

Return your response as a JSON array of task objects:
[
  {{"id": "task-1", "query": "...", "description": "...", "dependencies": [], "output_schema": "...", "downstream_use": "..."}},
  {{"id": "task-2", "query": "...", "description": "...", "dependencies": ["task-1"], "output_schema": "...", "downstream_use": "..."}},
  ...
]

Only return the JSON, no other text."""

        response = await self._llm_call(
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
                        output_schema=task_data.get("output_schema", ""),
                        downstream_use=task_data.get("downstream_use", ""),
                    )
                )
            return tasks
        except (json.JSONDecodeError, IndexError) as e:
            logger.warning(f"Failed to parse task decomposition: {e}, using fallback")
            return [
                AgentTask(
                    id="task-1",
                    query=query,
                    description="Main query",
                    output_schema="answer",
                    downstream_use="final answer",
                )
            ]

    async def _get_all_tools(self) -> list[Tool]:
        tools = get_research_tools()
        mcp_tools = await self._load_mcp_tools()
        tools.extend(mcp_tools)
        return tools

    async def _summarize_context(
        self, dep_id: str, dep_result: str, downstream_use: str
    ) -> str:
        """Summarize relevant parts of previous task results for downstream use."""
        if len(dep_result) < 500:
            return dep_result

        summary_prompt = f"""From the following result, extract ONLY the information needed for: {downstream_use}

Previous task result:
{dep_result}

Extract and summarize only the relevant facts needed for: {downstream_use}
Be concise and direct. Return only the extracted information, no explanations."""

        try:
            response = await self._llm_call(
                model=self.model,
                max_tokens=1024,
                system="You extract relevant information from research results. Be concise.",
                messages=[{"role": "user", "content": summary_prompt}],
            )
            text_blocks = [b.text for b in response.content if hasattr(b, "text")]
            return "\n".join(text_blocks) or dep_result[:500]
        except Exception as e:
            logger.warning(f"Failed to summarize context: {e}")
            return dep_result[:500]

    def _create_agent_system_prompt(self, task: AgentTask) -> str:
        tool_names = ["web_search", "fetch_content"]
        if self._mcp_tools:
            tool_names.extend([t.name for t in self._mcp_tools])

        output_info = (
            f"\nExpected output: {task.output_schema}" if task.output_schema else ""
        )
        downstream_info = (
            f"\nDownstream tasks will use this for: {task.downstream_use}"
            if task.downstream_use
            else ""
        )

        return f"""You are a research agent tasked with finding specific information.

Your Task:
- Query: {task.query}
- Description: {task.description}{output_info}{downstream_info}

IMPORTANT - Search First:
- ALWAYS use web_search or browser tools to find current information
- Do NOT rely on your training data or memory - search for the answer
- Your training data may be outdated, so always verify with a search
- Use browser tools to navigate to websites and extract information

Instructions:
1. Use web_search to search for relevant information
2. Use fetch_content or browser tools to get detailed content from URLs
3. Use MCP browser tools to navigate websites if needed
4. Be thorough - search multiple sources if needed
5. Provide a clear, concise answer matching the expected output format

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

            try:
                async with asyncio.timeout(self.agent_timeout_seconds):
                    result = await agent.run()
            except asyncio.TimeoutError:
                logger.warning(
                    f"Agent {task.id} timed out after {self.agent_timeout_seconds}s"
                )
                result = f"Agent timed out after {self.agent_timeout_seconds} seconds"

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

        response = await self._llm_call(
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

        yield streaming.message(f"Executing {len(tasks)} tasks (with dependencies)...")

        event_queue: asyncio.Queue[SSEEvent] = asyncio.Queue()

        results_map: dict[str, AgentResult] = {}

        for task in tasks:
            logger.info(
                f"Task {task.id}: {task.query[:100]}... (deps: {task.dependencies})"
            )
            for dep_id in task.dependencies:
                if dep_id in results_map:
                    dep_result = results_map[dep_id]
                    if dep_result.success:
                        downstream_use = task.downstream_use or "your task"
                        summarized = await self._summarize_context(
                            dep_id, dep_result.result, downstream_use
                        )
                        task.query += f"\n\n[Context from {dep_id}]: {summarized}"
                        logger.info(
                            f"Added summarized context from {dep_id} to task {task.id}"
                        )

            result = await self._run_single_agent(task, event_queue)
            results_map[task.id] = result

        task_results = list(results_map.values())

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

        try:
            await cleanup_mcp_connections()
        except Exception:
            pass

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
