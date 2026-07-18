"""FastAPI service: predict a used-car price from its attributes.

Input uses ``year`` (user-friendly); the service derives ``age`` the same way training
did, so the API contract stays natural while the model sees the feature it was trained
on. The model is loaded once at startup.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from car_price_ml import config
from car_price_ml import model as model_module

_state: dict = {"model": None}

# Only years the model was actually trained on (age in [0, AGE_MAX]) — avoids silent
# extrapolation beyond the training range.
_MIN_YEAR = config.REFERENCE_YEAR - config.AGE_MAX


class CarFeatures(BaseModel):
    """Validated input for a valuation request."""

    mark: str = Field(max_length=40, examples=["opel"])
    model: str = Field(max_length=60, examples=["combo"])
    fuel: str = Field(examples=["Diesel"])
    province: str = Field(max_length=60, examples=["Mazowieckie"])
    year: int = Field(ge=_MIN_YEAR, le=config.REFERENCE_YEAR, examples=[2015])
    mileage: int = Field(ge=0, le=int(config.MILEAGE_MAX), examples=[139568])
    vol_engine: int = Field(ge=0, le=int(config.VOL_ENGINE_MAX), examples=[1248])

    @field_validator("fuel")
    @classmethod
    def _known_fuel(cls, value: str) -> str:
        if value not in config.KNOWN_FUELS:
            raise ValueError(f"fuel must be one of {config.KNOWN_FUELS}")
        return value


class PricePrediction(BaseModel):
    predicted_price_pln: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _state["model"] = model_module.load_model()["model"]
    except FileNotFoundError:
        _state["model"] = None  # /health reports it; /predict returns 503
    yield


app = FastAPI(
    title="car-price-ml",
    description="Used-car price prediction for the Polish market",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _state["model"] is not None}


@app.post("/predict", response_model=PricePrediction)
def predict(car: CarFeatures) -> PricePrediction:
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="model not loaded — train first")
    row = pd.DataFrame([{
        "mark": car.mark,
        "model": car.model,
        "fuel": car.fuel,
        "province": car.province,
        "age": config.REFERENCE_YEAR - car.year,
        "mileage": car.mileage,
        "vol_engine": car.vol_engine,
    }])
    price = float(_state["model"].predict(row)[0])
    return PricePrediction(predicted_price_pln=round(price, 2))


# Serve the static valuation form (docs/app) at the site root — same origin as /predict, so
# the browser needs no CORS. Located relative to this file so it resolves whether the package
# is installed editable (local dev) or copied into the image (Docker). Mounted last so the API
# routes above take precedence over the catch-all static mount.
_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "docs" / "app"
if _FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=_FRONTEND_DIR, html=True), name="frontend")
