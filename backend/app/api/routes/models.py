from __future__ import annotations

from fastapi import APIRouter

from app.api.deps import get_available_models
from app.models.schemas import ModelInfo, ModelsResponse

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("", response_model=ModelsResponse)
async def list_models():
    """List available Claude models for research."""
    models = get_available_models()
    return ModelsResponse(models=[ModelInfo(**m) for m in models])
