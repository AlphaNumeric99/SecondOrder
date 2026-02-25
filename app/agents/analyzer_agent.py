from __future__ import annotations

from datetime import date

from app.agents.base import BaseAgent
from app.services.prompt_store import render_prompt


class AnalyzerAgent(BaseAgent):
    """Agent that synthesizes collected research into a comprehensive report.

    This agent has no tools — it uses the configured LLM's reasoning to analyze and synthesize
    the provided research data into a well-structured report with citations.
    """

    name = "analyzer"

    @staticmethod
    def get_system_prompt() -> str:
        today = date.today()
        return render_prompt(
            "analyzer_agent.system_prompt",
            today_iso=today.isoformat(),
            today_year=today.year,
        )

    system_prompt = get_system_prompt()
    tools = []  # No tools — pure analysis
