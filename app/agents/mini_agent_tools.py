from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from app.mini_agent.tools import Tool, ToolResult
from app.tools.jina_search import search as jina_search


@dataclass
class SearchTool(Tool):
    name: str = "web_search"
    description: str = (
        "Search the web for information. Use this to find relevant sources."
    )

    async def execute(self, query: str, max_results: int = 10) -> ToolResult:
        try:
            results = await jina_search(query, max_results=max_results)
            if not results:
                return ToolResult(success=True, content="No results found.")

            formatted = []
            for i, r in enumerate(results, 1):
                formatted.append(
                    f"[{i}] {r.title}\n    URL: {r.url}\n    {r.content[:300]}..."
                )

            return ToolResult(success=True, content="\n\n".join(formatted))
        except Exception as e:
            return ToolResult(
                success=False, content="", error=f"Search failed: {str(e)}"
            )

    def to_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                    "max_results": {
                        "type": "integer",
                        "description": "Max results (default 10)",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
        }

    def to_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.to_schema()["input_schema"],
            },
        }


@dataclass
class FetchTool(Tool):
    name: str = "fetch_content"
    description: str = "Fetch and extract content from a URL."

    async def execute(self, url: str, max_chars: int = 50000) -> ToolResult:
        try:
            from app.tools.jina_scraper import scrape_url

            content = await scrape_url(url, max_chars=max_chars)
            if not content:
                return ToolResult(
                    success=True, content="Could not fetch content from URL."
                )
            return ToolResult(success=True, content=content)
        except Exception as e:
            return ToolResult(
                success=False, content="", error=f"Fetch failed: {str(e)}"
            )

    def to_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to fetch"},
                    "max_chars": {
                        "type": "integer",
                        "description": "Max chars (default 50000)",
                        "default": 50000,
                    },
                },
                "required": ["url"],
            },
        }

    def to_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.to_schema()["input_schema"],
            },
        }


def get_research_tools() -> list[Tool]:
    return [SearchTool(), FetchTool()]
