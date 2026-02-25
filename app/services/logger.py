"""Centralized logging service using loguru."""

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from app.config import settings

# Configure loguru
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Remove default handler
logger.remove()

# Add console handler with color
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level=settings.app_log_level.upper(),
    colorize=True,
)

# Add file handler
logger.add(
    LOG_DIR / "secondorder_{time:YYYY-MM-DD}.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="DEBUG",
    rotation="00:00",  # New file at midnight
    retention="7 days",  # Keep logs for 7 days
    compression="zip",
)

# Reduce noise from framework/network libraries
import logging

for logger_name in (
    "uvicorn",
    "uvicorn.error",
    "uvicorn.access",
    "fastapi",
    "sse_starlette.sse",
    "httpx",
    "httpcore",
    "hpack",
    "openai._base_client",
    "asyncio",
):
    logging.getLogger(logger_name).setLevel(settings.noisy_log_level.upper())


def log_llm_call(
    model: str,
    caller: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    duration_ms: int = 0,
    status: str = "success",
    error: Optional[str] = None,
) -> None:
    """Log an LLM API call."""
    call_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "caller": caller,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "duration_ms": duration_ms,
        "status": status,
        "error": error,
    }
    if error:
        logger.error(f"LLM_CALL_FAILED: {call_data}")
    else:
        logger.info(f"LLM_CALL: {call_data}")


def log_research_step(
    session_id: str,
    step_type: str,
    status: str,
    data: Optional[dict] = None,
) -> None:
    """Log a research step."""
    step_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "step_type": step_type,
        "status": status,
        "data": data,
    }
    logger.info(f"RESEARCH_STEP: {step_data}")


def log_db_operation(
    operation: str,
    table: str,
    status: str,
    details: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    """Log a database operation."""
    op_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "operation": operation,
        "table": table,
        "status": status,
        "details": details,
        "error": error,
    }
    if error:
        logger.error(f"DB_OPERATION_FAILED: {op_data}")
    else:
        logger.info(f"DB_OPERATION: {op_data}")


def log_event(
    event_type: str,
    message: str,
    **kwargs,
) -> None:
    """Log a generic event."""
    event_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        "message": message,
        **kwargs,
    }
    logger.info(f"EVENT: {event_data}")
