"""Minimal FastAPI service for batch metric scoring (loads models at startup)."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Annotated

import pandas as pd
import uvicorn
from fastapi import Depends, FastAPI, Request
from pydantic import BaseModel, Field, create_model

from features import FEATURE_COLUMNS, FEATURE_DISPLAY_NAMES, feature_metadata
from predict import PredictorBundle, load_predictors, predict_dataframe

MetricRow = create_model(
    "MetricRow",
    __doc__="One row of operational metrics (same order as FEATURE_COLUMNS).",
    **{
        col: (float, Field(..., description=FEATURE_DISPLAY_NAMES[col]))
        for col in FEATURE_COLUMNS
    },
)


def create_app(bundle: PredictorBundle | None = None) -> FastAPI:
    """
    Build the FastAPI app.

    Pass ``bundle`` to inject trained models (e.g. tests) instead of loading
    from ``config`` paths at startup.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.predictor_bundle = bundle if bundle is not None else load_predictors()
        try:
            yield
        finally:
            del app.state.predictor_bundle

    app = FastAPI(title="Anomaly detection", version="0.1.0", lifespan=lifespan)
    _register_routes(app)
    return app


class PredictRequest(BaseModel):
    rows: list[MetricRow] = Field(..., min_length=1)


def get_bundle(request: Request) -> PredictorBundle:
    b = getattr(request.app.state, "predictor_bundle", None)
    if b is None:
        raise RuntimeError("Predictors not loaded")
    return b


def _register_routes(app: FastAPI) -> None:
    @app.get("/health")
    def health() -> dict[str, str | dict[str, str]]:
        meta = feature_metadata()
        return {
            "status": "ok",
            "features": ",".join(FEATURE_COLUMNS),
            "feature_display_names": {m["name"]: m["display_name"] for m in meta},
        }

    @app.post("/predict")
    def predict_batch(
        req: PredictRequest,
        bundle: Annotated[PredictorBundle, Depends(get_bundle)],
    ) -> dict[str, list[dict[str, float | int]]]:
        df = pd.DataFrame([r.model_dump() for r in req.rows])
        out = predict_dataframe(df, bundle)
        return {"rows": out.to_dict(orient="records")}


app = create_app()


def main() -> None:
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
