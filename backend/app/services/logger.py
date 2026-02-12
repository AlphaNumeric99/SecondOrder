"""Centralized logging service for debugging and monitoring."""
import logging
import json
from datetime import datetime
from typing import Any, Optional
from pathlib import Path

# Create logs directory
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "secondorder.log"),
        logging.StreamHandler(),  # Also print to console
    ],
)

logger = logging.getLogger("secondorder")


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
        "timestamp": datetime.utcnow().isoformat(),
        "model": model,
        "caller": caller,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "duration_ms": duration_ms,
        "status": status,
        "error": error,
    }
    logger.info(f"LLM_CALL: {json.dumps(call_data)}")


def log_research_step(
    session_id: str,
    step_type: str,
    status: str,
    data: Optional[dict] = None,
) -> None:
    """Log a research step."""
    step_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "session_id": session_id,
        "step_type": step_type,
        "status": status,
        "data": data,
    }
    logger.info(f"RESEARCH_STEP: {json.dumps(step_data)}")


def log_db_operation(
    operation: str,
    table: str,
    status: str,
    details: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    """Log a database operation."""
    op_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "operation": operation,
        "table": table,
        "status": status,
        "details": details,
        "error": error,
    }
    logger.info(f"DB_OPERATION: {json.dumps(op_data)}")


def log_event(
    event_type: str,
    message: str,
    **kwargs,
) -> None:
    """Log a generic event."""
    event_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "event_type": event_type,
        "message": message,
        **kwargs,
    }
    logger.info(f"EVENT: {json.dumps(event_data)}")
