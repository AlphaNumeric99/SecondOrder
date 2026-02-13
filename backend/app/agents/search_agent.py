from __future__ import annotations

from datetime import date
from typing import Any

from app.agents.base import BaseAgent
from app.models.events import SSEEvent
from app.services import streaming
from app.tools import search_provider


class SearchAgent(BaseAgent):
    """Agent that searches the web using configured provider(s)."""

    name = "search"

    @property
    def system_prompt(self) -> str:
        today = date.today()
        return (
            f"You are a web search specialist. Today's date is {today.isoformat()} ({today.year}). "
            "Given a research query, use the search tool "
            "to find the most relevant and recent information. "
            "First, execute one search using the user's query verbatim. "
            "Then run additional focused searches to resolve entities and verify facts. "
            "For entity resolution, prefer canonical sources (e.g., site:wikipedia.org) before secondary sources. "
            f"Always include the current year ({today.year}) in your search queries when looking for "
            "current data, statistics, or recent developments. "
            "Search multiple angles if needed. "
            "After searching, summarize the key findings with source URLs."
        )
    tools = [
        {
            "name": "web_search",
            "description": (
                "Search the web for information. Use specific, targeted queries. "
                "You can call this tool multiple times with different queries to cover "
                "different angles of the research topic."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query. Be specific and targeted.",
                    },
                    "search_depth": {
                        "type": "string",
                        "enum": ["basic", "advanced"],
                        "description": "Use 'advanced' for thorough research, 'basic' for quick lookups.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Number of results to return (1-20).",
                        "default": 10,
                    },
                    "time_range": {
                        "type": "string",
                        "enum": ["day", "week", "month", "year"],
                        "description": "Filter results by recency. Omit for all time.",
                    },
                },
                "required": ["query"],
            },
        }
    ]

    def __init__(self, model: str | None = None, step_index: int = 0, session_id: str | None = None):
        super().__init__(model, session_id=session_id)
        self.step_index = step_index
        self.all_results: list[dict[str, Any]] = []

    async def handle_tool_call(
        self, tool_name: str, tool_input: dict[str, Any]
    ) -> tuple[str, list[SSEEvent]]:
        events: list[SSEEvent] = []

        if tool_name == "web_search":
            query = tool_input["query"]
            events.append(streaming.agent_progress(
                self.name, step=self.step_index, query=query, status="searching"
            ))

            search_response = await search_provider.search(
                query=query,
                search_depth=tool_input.get("search_depth", "advanced"),
                max_results=tool_input.get("max_results", 10),
                time_range=tool_input.get("time_range"),
            )
            results = search_response.results

            result_dicts = search_provider.results_to_dicts(results)
            self.all_results.extend(result_dicts)

            events.append(
                streaming.search_result(
                    self.step_index,
                    result_dicts,
                    provider=search_response.provider,
                    fallback_from=search_response.fallback_from,
                    fallback_reason=search_response.fallback_reason,
                )
            )

            return (
                f"Found {len(results)} results for '{query}' via {search_response.provider}:\n"
                + "\n".join(
                    f"- [{r.title}]({r.url}): {r.content[:200]}" for r in results
                ),
                events,
            )

        raise NotImplementedError(f"Unknown tool: {tool_name}")
