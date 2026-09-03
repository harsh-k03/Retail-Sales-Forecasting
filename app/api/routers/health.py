"""Liveness and model status."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.schemas import HealthResponse
from app.api.services import ModelService, get_service
from src.config import Config

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(service: ModelService = Depends(get_service)) -> HealthResponse:
    bundle = service.bundle
    return HealthResponse(
        status="ok",
        version=str(Config.load().get("project.version", "0.0.0")),
        model_loaded=bundle is not None,
        model_name=bundle.model_name if bundle else None,
        uptime_seconds=round(service.uptime, 2),
    )


@router.post("/model/reload", tags=["model"])
def reload_model(service: ModelService = Depends(get_service)) -> dict:
    bundle = service.reload_model()
    return {"reloaded": bundle is not None, "model_name": bundle.model_name if bundle else None}
