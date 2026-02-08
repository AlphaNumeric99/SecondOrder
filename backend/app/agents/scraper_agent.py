from __future__ import annotations

from typing import Any

from app.agents.base import BaseAgent
from app.models.events import SSEEvent
from app.services import streaming
from app.tools import hasdata_scraper, web_utils


class ScraperAgent(BaseAgent):
    """Agent that scrapes web pages to extract detailed content."""

    name = "scraper"
    system_prompt = (
        "You are a web content extraction specialist. Given URLs, use the scrape tool "
        "to extract their full content. Focus on extracting the most relevant information "
        "from each page. Summarize the key content from each source."
    )
    tools = [
        {
            "name": "scrape_url",
            "description": (
                "Scrape the full content of a web page. Use this to get detailed "
                "information from URLs found during search."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to scrape.",
                    },
                    "render_js": {
                        "type": "boolean",
                        "description": "Whether to render JavaScript. Use for dynamic pages.",
                        "default": False,
                    },
                },
                "required": ["url"],
            },
        }
    ]

    def __init__(self, model: str | None = None, session_id: str | None = None):
        super().__init__(model, session_id=session_id)
        self.scraped_content: dict[str, str] = {}

    async def handle_tool_call(
        self, tool_name: str, tool_input: dict[str, Any]
    ) -> tuple[str, list[SSEEvent]]:
        events: list[SSEEvent] = []

        if tool_name == "scrape_url":
            url = tool_input["url"]

            if not web_utils.is_valid_url(url):
                return f"Invalid URL: {url}", []

            events.append(streaming.agent_progress(
                self.name, url=url, status="scraping"
            ))

            try:
                result = await hasdata_scraper.scrape(
                    url, render_js=tool_input.get("render_js", False)
                )
                cleaned = web_utils.clean_content(result.content)
                self.scraped_content[url] = cleaned

                events.append(streaming.scrape_result(url, cleaned[:500]))

                return (
                    f"Content from {web_utils.extract_domain(url)} "
                    f"({len(cleaned)} chars):\n{cleaned}",
                    events,
                )
            except Exception as e:
                return f"Failed to scrape {url}: {e}", events

        raise NotImplementedError(f"Unknown tool: {tool_name}")
