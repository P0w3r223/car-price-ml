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
from pydantic import BaseModel, Field, field_validator, model_validator

from car_price_ml import config, data
from car_price_ml import model as model_module

_state: dict = {"model": None, "vocabulary": {}}

# Only years the model was actually trained on (age in [0, AGE_MAX]) — avoids silent
# extrapolation beyond the training range.
_MIN_YEAR = config.REFERENCE_YEAR - config.AGE_MAX


class CarFeatures(BaseModel):
    """Validated input for a valuation request."""

    mark: str = Field(max_length=config.MARK_MAX_LENGTH, examples=["opel"])
    model: str = Field(max_length=config.MODEL_MAX_LENGTH, examples=["combo"])

    @field_validator("mark", "model")
    @classmethod
    def _casefold(cls, value: str) -> str:
        """Normalise case; membership is checked against the artifact's own vocabulary.

        The dataset spells every make and model in lower case, so "Opel" is a spelling
        variant rather than a different car — but only the loaded artifact knows which
        values it was fit on, so the domain check lives in the endpoint.

        Delegated rather than done here, so the published page's refusal table can execute
        the same rule instead of re-implementing it and going stale against it.
        """
        normalised = data.canonical_mark_or_model(value)
        if normalised is None:
            raise ValueError("must not be blank")
        return normalised
    fuel: str = Field(max_length=20, examples=["Diesel"])
    province: str = Field(max_length=60, examples=["Mazowieckie"])
    year: int = Field(ge=_MIN_YEAR, le=config.REFERENCE_YEAR, examples=[2015])
    mileage: int = Field(ge=0, le=int(config.MILEAGE_MAX), examples=[139568])
    vol_engine: int = Field(ge=0, le=int(config.VOL_ENGINE_MAX), examples=[1248])

    @field_validator("fuel")
    @classmethod
    def _known_fuel(cls, value: str) -> str:
        """Reject unknown fuels; normalise the case of known ones.

        Symmetric with the province validator below — a caller sending "diesel" and
        "mazowieckie" in one payload should not have one accepted and the other refused.
        """
        canonical = data.canonical_fuel(value)
        if canonical is None:
            raise ValueError(f"fuel must be one of {config.KNOWN_FUELS}")
        return canonical

    @field_validator("province")
    @classmethod
    def _known_province(cls, value: str) -> str:
        """Reject unknown provinces; normalise the rest to the trained spelling.

        Without this the one-hot encoder would answer an unrecognised province with an
        all-zero row — a valuation computed as if the car had no location, returned with a
        200 and no indication that anything was ignored. Callers that send a differently
        capitalised or unaccented name get normalised rather than refused.
        """
        canonical = data.canonical_province(value)
        if canonical is None:
            raise ValueError(f"province must be one of {config.PROVINCES}")
        return canonical

    @model_validator(mode="after")
    def _displacement_matches_fuel(self) -> CarFeatures:
        """Zero displacement is only meaningful for an EV — the same rule training applies.

        Cleaning drops combustion adverts with no displacement, so the model has never seen
        one; asked anyway, it answered from what zero displacement means in the data it did
        see (an EV) and priced a diesel 11 % above the same car with a real engine.
        """
        if not data.has_plausible_displacement(self.fuel, self.vol_engine):
            raise ValueError(
                f"vol_engine 0 is only valid for Electric, not {self.fuel} — "
                f"give the engine capacity in cm³"
            )
        return self


class PricePrediction(BaseModel):
    """A valuation, and the market it is a valuation of.

    The price level is a property of the training data's vintage, not of today. Returning a
    bare number invites the reader to supply "today" themselves, which is the claim the model
    cannot support — Polish used-car prices moved roughly −10 % to −24 % after this snapshot
    while general inflation ran +41 %, so the error from assuming currency is large and
    signed the wrong way. See docs/research/temporal-recalibration.md.
    """

    predicted_price_pln: float
    valuation_as_of: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        bundle = model_module.load_model()
        _state["model"] = bundle["model"]
        _state["vocabulary"] = bundle["metadata"].get("vocabulary", {})
    except (FileNotFoundError, ValueError):
        # Missing artifact, or one whose feature spec / age anchor no longer matches the
        # code. Both are deployment faults rather than bad requests: the service starts so
        # /health can report it and the form still loads, but /predict refuses with a 503
        # rather than serving numbers from a mismatched model.
        _state["model"] = None
        _state["vocabulary"] = {}
    yield


app = FastAPI(
    title="car-price-ml",
    description="Used-car price prediction for the Polish market",
    # 0.2.0: `fuel` and `province` are now closed vocabularies — an unknown value is a 422
    # where it previously returned a 200 computed from an all-zero category block. That is a
    # behavioural break for any existing caller, so it carries a minor-version bump.
    version="0.2.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _state["model"] is not None}


@app.get("/vocabulary")
def vocabulary() -> dict:
    """The closed domains a valuation request must fall inside, keyed by request field.

    With an artifact loaded every value comes from it — the only thing that knows what the
    model can price, and already checked at load against the code's declared domains. With
    no artifact, `mark`/`model` are empty (nothing can be said about them) while
    `fuel`/`province` fall back to the compile-time domains, which are correct by
    construction because the encoder is built from them.

    Order is the artifact's, not re-sorted: Python's default collation places Ł, Ś and Ż
    after Z, which is wrong for Polish and would make the form's list look scrambled.
    """
    fallback = {"fuel": list(config.KNOWN_FUELS), "province": list(config.PROVINCES)}
    stamped = _state["vocabulary"]
    return {field: list(stamped.get(field) or fallback.get(field, [])) for field in
            ("mark", "model", "fuel", "province")}


def _require_known(field: str, value: str) -> None:
    """Reject a make/model the artifact was not fit on.

    Unlike the one-hot columns, an unseen category here does not raise inside the pipeline —
    ``TargetEncoder`` substitutes the global target mean, so an unknown car comes back as a
    confident, entirely fictional price. Measured on the served artifact, "Ferrari"/"f40"
    and "zzzz"/"qqqq" both returned 33 282 PLN.
    """
    known = _state["vocabulary"].get(field)
    if not known:
        # Failing open here would resurrect exactly the bug this guard exists to close, and
        # do it invisibly. An artifact whose vocabulary cannot be read is unservable, which
        # is a deployment fault rather than a bad request.
        raise HTTPException(
            status_code=503,
            detail=f"loaded model carries no {field} vocabulary — cannot verify input",
        )
    if value not in known:
        raise HTTPException(
            status_code=422,
            detail=f"unknown {field}: {value!r} — the model was not trained on it "
                   f"(see GET /vocabulary)",
        )


@app.post("/predict", response_model=PricePrediction)
def predict(car: CarFeatures) -> PricePrediction:
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="model not loaded — train first")
    _require_known("mark", car.mark)
    _require_known("model", car.model)
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
    return PricePrediction(
        predicted_price_pln=round(price, 2), valuation_as_of=config.REFERENCE_YEAR
    )


# Serve the static valuation form (docs/app) at the site root — same origin as /predict, so
# the browser needs no CORS. Located relative to this file so it resolves whether the package
# is installed editable (local dev) or copied into the image (Docker). Mounted last so the API
# routes above take precedence over the catch-all static mount.
_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "docs" / "app"
if _FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=_FRONTEND_DIR, html=True), name="frontend")
