from __future__ import annotations

def get_available_models() -> list[dict[str, str]]:
    """Return a curated list of OpenRouter model IDs for research."""
    return [
        {
            "id": "openai/gpt-4o-mini",
            "name": "GPT-4o Mini",
            "description": "Fast and cost-efficient default for everyday deep research.",
        },
        {
            "id": "openai/gpt-4.1",
            "name": "GPT-4.1",
            "description": "Higher quality reasoning for harder multi-source synthesis tasks.",
        },
        {
            "id": "google/gemini-2.0-flash-001",
            "name": "Gemini 2.0 Flash",
            "description": "High-throughput option for broad exploratory query expansion.",
        },
        {
            "id": "meta-llama/llama-3.3-70b-instruct",
            "name": "Llama 3.3 70B Instruct",
            "description": "Strong open-weight alternative for lower-cost long-form synthesis.",
        },
    ]
