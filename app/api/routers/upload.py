"""CSV upload and schema validation."""
from __future__ import annotations

import io
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from app.api.schemas import UploadResponse, ValidationIssue
from app.api.services import ModelService, get_service
from src.config import Config, available_schemas

router = APIRouter(prefix="/data", tags=["data"])


@router.post("/upload", response_model=UploadResponse)
async def upload(
    file: UploadFile = File(...),
    preset: Optional[str] = Query(None, description="Schema preset; inferred when omitted"),
    service: ModelService = Depends(get_service),
) -> UploadResponse:
    if not file.filename.lower().endswith((".csv", ".parquet", ".pq")):
        raise HTTPException(status_code=415, detail="Only CSV or Parquet files are supported")

    payload = await file.read()
    max_mb = float(Config.load().get("api.max_upload_mb", 50))
    if len(payload) > max_mb * 1024**2:
        raise HTTPException(status_code=413, detail=f"File exceeds {max_mb} MB limit")

    if preset and preset not in available_schemas():
        raise HTTPException(status_code=400, detail=f"Unknown preset. Available: {available_schemas()}")

    buffer = io.BytesIO(payload)
    buffer.name = file.filename
    try:
        dataset_id, frame, report = service.ingest_upload(buffer, preset=preset)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not parse file: {exc}") from exc

    service.record("upload", rows=len(frame))
    return UploadResponse(
        dataset_id=dataset_id,
        filename=file.filename,
        rows=len(frame),
        columns=list(frame.columns),
        passed=report.passed,
        issues=[ValidationIssue(rule=i.rule, severity=i.severity, message=i.message, count=i.count) for i in report.issues],
        summary=report.summary,
    )


@router.get("/presets")
def presets() -> dict:
    return {"presets": available_schemas()}
