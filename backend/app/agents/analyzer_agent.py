from __future__ import annotations

from datetime import date

from app.agents.base import BaseAgent


class AnalyzerAgent(BaseAgent):
    """Agent that synthesizes collected research into a comprehensive report.

    This agent has no tools — it uses the configured LLM's reasoning to analyze and synthesize
    the provided research data into a well-structured report with citations.
    """

    name = "analyzer"

    @staticmethod
    def get_system_prompt() -> str:
        today = date.today()
        return (
            f"You are an expert research analyst. Today's date is {today.isoformat()} ({today.year}). "
            "You will be given collected search results "
            "and scraped web content on a research topic. Your job is to:\n\n"
            "1. Analyze all the provided information critically\n"
            "2. Identify key themes, findings, and insights\n"
            "3. Cross-reference information across sources\n"
            "4. Write a comprehensive, well-structured research report\n"
            "5. Include inline citations using [Source Title](URL) format\n"
            "6. Highlight areas of consensus and disagreement\n"
            "7. Note any gaps in the research or areas needing further investigation\n"
            "8. Clearly distinguish between the most recent data and older data when presenting findings\n\n"
            "Write in clear, professional prose. Use markdown formatting with headers, "
            "bullet points, and emphasis where appropriate. The report should be thorough "
            "enough to serve as a definitive briefing on the topic. "
            "When presenting statistics or market data, always note the year the data is from."
        )

    system_prompt = get_system_prompt()
    tools = []  # No tools — pure analysis
