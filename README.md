# SecondOrder

**Deep research that thinks beyond the obvious.**

SecondOrder is an AI-powered deep research tool that goes beyond first-pass answers. Where typical search tools retrieve surface-level results, SecondOrder applies *second-order thinking* — it plans research angles, searches in parallel, scrapes primary sources, and synthesizes everything into a comprehensive, cited report.

## Why "SecondOrder"?

In decision-making, **first-order thinking** asks: *"What's the answer?"*
**Second-order thinking** asks: *"What are the consequences of that answer? What connections am I missing? What does the evidence actually say when you dig deeper?"*

Most research tools stop at first-order — they return search snippets and call it done. SecondOrder decomposes your question into multiple angles, investigates each one independently, reads the actual sources, cross-references findings, and produces a report with real citations. It thinks about your question the way a skilled researcher would.

## How It Works

```
You ask a question
       |
       v
  [Orchestrator] — breaks query into 3-5 research angles
       |
       v
  [Search Agents] — parallel web searches via Tavily (one per angle)
       |
       v
  [Scraper Agents] — extracts full content from top sources via Hasdata
       |
       v
  [Analyzer] — Claude synthesizes everything into a cited report
       |
       v
  Streamed to you in real time via SSE
```

Every step streams progress to the UI — you see the plan form, searches execute, sources get scraped, and the final report writes itself token by token.

## Architecture

- **Backend**: Python + FastAPI, raw asyncio orchestration with Anthropic SDK
- **Frontend**: Next.js 16, React 19, Tailwind CSS 4
- **Database**: Supabase (PostgreSQL) for sessions, messages, research steps, and LLM call logging
- **Search**: Tavily API
- **Scraping**: Hasdata API
- **Models**: Claude Opus 4.6 / Sonnet 4.5 / Haiku 4.5 (user-selectable)

No agent frameworks. No LangChain. No LlamaIndex. Just Python, asyncio, and the Anthropic SDK — maximum control, zero lock-in.

## Quick Start

### Prerequisites

- Python 3.11+ (with uv)
- Node.js 18+
- API keys: Anthropic, Tavily, Hasdata
- Supabase project

### Backend

```bash
cd backend
cp .env.example .env          # Fill in your API keys
uv pip install -e ".[dev]"
.venv/Scripts/python.exe -m uvicorn app.main:app --reload
```

### Frontend

```bash
cd frontend
cp .env.local.example .env.local  # Set backend URL
npm install
npm run dev
```

### Database

Run the SQL migrations in `supabase/migrations/` against your Supabase project.

## Benchmarks

SecondOrder is evaluated against three industry-standard deep research benchmarks:

| Benchmark | Source | Tasks | What It Measures |
|-----------|--------|-------|------------------|
| **DRACO** | Perplexity | 100 | Rubric-based research quality across domains |
| **ResearchRubrics** | Scale AI | 2,500+ | Expert rubric compliance across 9 domains |
| **DeepSearchQA** | Google DeepMind | 900 | Factual answer extraction (F1 score) |

```bash
cd backend
.venv/Scripts/python.exe -m benchmarks.run draco --limit 5
.venv/Scripts/python.exe -m benchmarks.run deepsearchqa --limit 10
.venv/Scripts/python.exe -m benchmarks.run all --limit 3
```

## License

MIT
