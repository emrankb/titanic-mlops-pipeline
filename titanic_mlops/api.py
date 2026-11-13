"""Local FastAPI service for Titanic survival estimates.

Only the seven inference features are accepted. No passenger ID, name,
ticket, target label, or request body is stored or logged by this module.
"""
from __future__ import annotations

from pathlib import Path
import os

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr

from .data import RAW_FEATURES


class PassengerFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    Pclass: StrictInt = Field(ge=1, le=3)
    Sex: StrictStr = Field(pattern="^(male|female)$")
    Age: float | None = Field(default=None, ge=0, le=120, allow_inf_nan=False)
    SibSp: StrictInt = Field(ge=0, le=30)
    Parch: StrictInt = Field(ge=0, le=30)
    Fare: float = Field(ge=0, allow_inf_nan=False)
    Embarked: str | None = Field(default=None, pattern="^(S|C|Q)$")


class SurvivalPrediction(BaseModel):
    survived: int = Field(ge=0, le=1)
    survival_probability: float = Field(ge=0, le=1)


def load_trusted_model(path: str | Path):
    """Deserialize only the model produced by the trusted local training step."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise RuntimeError("Model artifact missing. Train the model before serving.")
    model = joblib.load(path)
    if not all(hasattr(model, attr) for attr in ("predict", "predict_proba", "classes_")):
        raise RuntimeError("Model does not meet the expected classifier contract.")
    if set(model.classes_) != {0, 1}:
        raise RuntimeError("Model must have both binary survival labels.")
    return model


def create_app(*, model=None, model_path: str | Path | None = None) -> FastAPI:
    """Build an API with its already fitted model loaded exactly once."""
    if model is None:
        model = load_trusted_model(
            model_path or os.environ.get("TITANIC_MODEL_PATH", "artifacts/survival-model/model.joblib")
        )
    app = FastAPI(title="Titanic Survival API", version="1.0")

    @app.get("/health")
    def health():
        return {"status": "ready", "model_loaded": True}

    @app.post("/predict", response_model=SurvivalPrediction)
    def predict(body: PassengerFeatures):
        row = pd.DataFrame([body.model_dump()], columns=list(RAW_FEATURES))
        predicted = int(model.predict(row)[0])
        labels = list(model.classes_)
        probability = float(model.predict_proba(row)[0][labels.index(1)])
        return SurvivalPrediction(survived=predicted, survival_probability=probability)

    return app
