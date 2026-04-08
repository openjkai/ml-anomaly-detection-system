"""Minimal FastAPI service for batch metric scoring (loads models at startup)."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Annotated

import pandas as pd
import uvicorn
from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

from features import FEATURE_COLUMNS
from predict import PredictorBundle, load_predictors, predict_dataframe

_bundle: PredictorBundle | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _bundle
    _bundle = load_predictors()
    yield
    _bundle = None


app = FastAPI(title="Anomaly detection", version="0.1.0", lifespan=lifespan)


class MetricRow(BaseModel):
    """One row of operational metrics (same order as ``FEATURE_COLUMNS``)."""

    cpu_usage: float = Field(..., description="CPU utilization %")
    memory_usage: float = Field(..., description="Memory %")
    request_latency_ms: float
    error_rate: float
    request_count: float
    disk_io: float
    network_in_mb: float


class PredictRequest(BaseModel):
    rows: list[MetricRow] = Field(..., min_length=1)


def get_bundle() -> PredictorBundle:
    if _bundle is None:
        raise RuntimeError("Predictors not loaded")
    return _bundle


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "features": ",".join(FEATURE_COLUMNS)}


@app.post("/predict")
def predict_batch(
    req: PredictRequest,
    bundle: Annotated[PredictorBundle, Depends(get_bundle)],
) -> dict[str, list[dict[str, float | int]]]:
    df = pd.DataFrame([r.model_dump() for r in req.rows])
    out = predict_dataframe(df, bundle)
    return {"rows": out.to_dict(orient="records")}


def main() -> None:
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
