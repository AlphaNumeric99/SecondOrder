from __future__ import annotations

import argparse
import asyncio
import os


def _ensure_min_env() -> None:
    defaults = {
        "OPENROUTER_API_KEY": "debug-placeholder",
        "TAVILY_API_KEY": "debug-placeholder",
        "HASDATA_API_KEY": "debug-placeholder",
        "SUPABASE_URL": "https://debug.local",
        "SUPABASE_ANON_KEY": "debug-placeholder",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)


async def _run(ttl_hours: int) -> int:
    from app.config import settings
    from app.services.memory_store import get_memory_store

    store = get_memory_store()
    expired = await store.cleanup_expired_sessions(ttl_hours=ttl_hours)
    print(f"Chroma cleanup complete. Deleted sessions: {len(expired)}")
    for session_id in expired:
        print(f"- {session_id}")
    print(f"Persist dir: {settings.chroma_persist_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Cleanup expired Chroma session collections.")
    parser.add_argument(
        "--ttl-hours",
        type=int,
        default=168,
        help="Delete sessions older than this many hours.",
    )
    args = parser.parse_args()

    _ensure_min_env()
    return asyncio.run(_run(args.ttl_hours))


if __name__ == "__main__":
    raise SystemExit(main())
