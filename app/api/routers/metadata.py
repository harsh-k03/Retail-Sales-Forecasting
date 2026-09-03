"""Model metadata, dataset schema and prediction history."""
from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.schemas import HistoryEntry, ModelMetadata, SchemaResponse
from app.api.services import ModelService, get_service
from src.config import available_schemas
from src.data import schema as S

router = APIRouter(tags=["metadata"])


@router.get("/model/metadata", response_model=ModelMetadata)
def model_metadata(service: ModelService = Depends(get_service)) -> ModelMetadata:
    bundle = service.bundle
    if bundle is None:
        raise HTTPException(status_code=503, detail="No trained model available")
    return ModelMetadata(
        model_name=bundle.model_name,
        target=bundle.target,
        n_features=len(bundle.feature_names),
        metrics=bundle.metrics,
        created_at=bundle.created_at,
        feature_names=bundle.feature_names,
        extra=bundle.metadata,
    )


@router.get("/schema", response_model=SchemaResponse)
def dataset_schema() -> SchemaResponse:
    return SchemaResponse(
        required=S.REQUIRED_COLUMNS,
        optional=S.OPTIONAL_COLUMNS,
        presets=available_schemas(),
        descriptions=S.DESCRIPTIONS,
    )


@router.get("/predictions/history", response_model=List[HistoryEntry])
def history(limit: int = Query(50, ge=1, le=500), service: ModelService = Depends(get_service)) -> List[HistoryEntry]:
    return [HistoryEntry(**entry) for entry in service.history(limit)]
