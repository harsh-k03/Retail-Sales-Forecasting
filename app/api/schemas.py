"""Pydantic request/response models for the REST API."""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    model_loaded: bool
    model_name: Optional[str] = None
    uptime_seconds: float


class ValidationIssue(BaseModel):
    rule: str
    severity: str
    message: str
    count: int = 0


class UploadResponse(BaseModel):
    dataset_id: str
    filename: str
    rows: int
    columns: List[str]
    passed: bool
    issues: List[ValidationIssue] = Field(default_factory=list)
    summary: Dict[str, Any] = Field(default_factory=dict)


class ForecastRequest(BaseModel):
    horizon: int = Field(30, ge=1, le=365)
    dataset_id: Optional[str] = None
    store: Optional[str] = None
    product: Optional[str] = None
    overrides: Dict[str, float] = Field(default_factory=dict, description="Scenario levers, e.g. {'promotion': 1}")
    aggregate: bool = False


class ForecastRecord(BaseModel):
    date: date
    store: Optional[str] = None
    product: Optional[str] = None
    forecast: float


class ForecastResponse(BaseModel):
    request_id: str
    model_name: str
    horizon: int
    generated_at: datetime
    summary: Dict[str, float]
    records: List[ForecastRecord]


class ModelMetadata(BaseModel):
    model_name: str
    target: str
    n_features: int
    metrics: Dict[str, float] = Field(default_factory=dict)
    created_at: Optional[str] = None
    feature_names: List[str] = Field(default_factory=list)
    extra: Dict[str, Any] = Field(default_factory=dict)


class SchemaResponse(BaseModel):
    required: List[str]
    optional: List[str]
    presets: List[str]
    descriptions: Dict[str, str]


class HistoryEntry(BaseModel):
    request_id: str
    endpoint: str
    horizon: Optional[int] = None
    model_name: Optional[str] = None
    rows: int = 0
    created_at: datetime


class InsightRecord(BaseModel):
    priority: str
    category: str
    title: str
    detail: str
    action: str
