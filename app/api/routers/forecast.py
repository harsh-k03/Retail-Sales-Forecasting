"""Forecast generation and business insights."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import List

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from app.api.schemas import ForecastRecord, ForecastRequest, ForecastResponse, InsightRecord
from app.api.services import ModelService, get_service
from src.data import schema as S
from src.forecast.recursive import forecast as run_forecast
from src.forecast.recursive import forecast_summary
from src.insights.engine import InsightEngine

router = APIRouter(tags=["forecast"])


def _filtered(frame: pd.DataFrame, store, product) -> pd.DataFrame:
    if store is not None:
        frame = frame[frame[S.STORE].astype(str) == str(store)]
    if product is not None:
        frame = frame[frame[S.PRODUCT].astype(str) == str(product)]
    if frame.empty:
        raise HTTPException(status_code=404, detail="No rows match the requested store/product")
    return frame


@router.post("/forecast", response_model=ForecastResponse)
def forecast(request: ForecastRequest, service: ModelService = Depends(get_service)) -> ForecastResponse:
    bundle = service.bundle
    if bundle is None:
        raise HTTPException(status_code=503, detail="No trained model available. Run the pipeline first.")
    try:
        history = _filtered(service.get_dataset(request.dataset_id), request.store, request.product)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    frame = run_forecast(
        bundle.model, history, bundle.builder,
        horizon=request.horizon, overrides=request.overrides or None, config=service.config,
    )
    if request.aggregate:
        agg = frame.groupby(S.DATE, observed=True)["forecast"].sum().reset_index()
        records = [ForecastRecord(date=row[S.DATE].date(), forecast=float(row["forecast"])) for _, row in agg.iterrows()]
    else:
        records = [
            ForecastRecord(
                date=row[S.DATE].date(), store=str(row[S.STORE]),
                product=str(row[S.PRODUCT]), forecast=float(row["forecast"]),
            )
            for _, row in frame.iterrows()
        ]

    request_id = service.record("forecast", horizon=request.horizon, model_name=bundle.model_name, rows=len(records))
    return ForecastResponse(
        request_id=request_id,
        model_name=bundle.model_name,
        horizon=request.horizon,
        generated_at=datetime.now(timezone.utc),
        summary={k: float(v) for k, v in forecast_summary(frame, history).items()},
        records=records,
    )


@router.post("/insights", response_model=List[InsightRecord])
def insights(request: ForecastRequest, service: ModelService = Depends(get_service)) -> List[InsightRecord]:
    bundle = service.bundle
    history = service.get_dataset(request.dataset_id)
    frame = None
    if bundle is not None:
        frame = run_forecast(bundle.model, history, bundle.builder, horizon=request.horizon, config=service.config)
    importance = bundle.model.feature_importance() if bundle else None
    table = InsightEngine(history, frame, importance).to_frame()
    service.record("insights", horizon=request.horizon, rows=len(table))
    return [InsightRecord(**row) for row in table.to_dict("records")]
