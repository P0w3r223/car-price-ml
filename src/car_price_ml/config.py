"""Central configuration: dataset, paths, features, outlier rules, model settings.

No I/O here — only constants. Columns match the "Car Prices Poland" Kaggle dataset
(aleksandrglotov): mark, model, generation_name, year, mileage, vol_engine, fuel, city,
province, price. See docs/research/data-and-methodology.md.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Paths -------------------------------------------------------------------
# PROJECT_ROOT assumes the src/ layout, which only holds for an editable install. Installed
# non-editable — as the Dockerfile does — this file lives in site-packages and the derived
# paths point inside the interpreter's library directory, nowhere near the repository. The
# artifact and data locations are therefore overridable by environment variable, so a
# container can say where it put things instead of the package guessing. (Before this, the
# image ran with a permanently unloadable model: /predict answered 503 forever and the form
# silently fell back to its offline heuristic.)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _dir_from_env(variable: str, default: Path) -> Path:
    override = os.environ.get(variable)
    return Path(override) if override else default


DATA_DIR = _dir_from_env("CAR_PRICE_DATA_DIR", PROJECT_ROOT / "data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = _dir_from_env("CAR_PRICE_MODELS_DIR", PROJECT_ROOT / "models")
FIGURES_DIR = _dir_from_env("CAR_PRICE_FIGURES_DIR", PROJECT_ROOT / "reports" / "figures")

# The published site and the small aggregates it is rendered from. Split deliberately: the
# aggregates are committed, so the page can be rebuilt (and diffed) from inputs that are in
# the repository, while producing them needs the 14 MB artifact and the dataset, which are
# not. Only `site.export` writes SITE_DATA_DIR; only `site.build` reads it.
DOCS_DIR = _dir_from_env("CAR_PRICE_DOCS_DIR", PROJECT_ROOT / "docs")
SITE_DATA_DIR = DOCS_DIR / "data"
# The static valuation form. Two of the files in it are generated from this module by
# `site.form`, for the same reason the report page is generated: a value restated by hand in
# a second language drifts, and the one that drifted here priced ~7 % of the market with no
# location at all.
SITE_APP_DIR = DOCS_DIR / "app"

# --- Dataset -----------------------------------------------------------------
KAGGLE_DATASET = "aleksandrglotov/car-prices-poland"  # CC0-1.0; attribute in README
DATASET_CSV = RAW_DIR / "Car_Prices_Poland_Kaggle.csv"

# --- Target & features -------------------------------------------------------
TARGET = "price"
# Prices are right-skewed, so every model trains on log1p(price) and inverts with expm1
# before any metric — see model._wrap. That is a methodology commitment rather than a
# setting: a flag here would imply the pipeline still works with it switched off, and the
# metric-inversion code that makes the numbers PLN would silently become wrong.

# Reference year for deriving `age`, anchored to the dataset's own vintage: the source is a
# single January 2022 scrape, not the "~2021-2023" range previously documented here. The
# model-year distribution corroborates it — 2021 is 9 % of rows, 2022 is 1.8 %, and nothing
# follows. Anchoring anywhere later invents ages the data cannot support: at 2024 the model
# had never seen a car younger than `age` 2, while the API happily accepted model year 2024
# and asked for a prediction at `age` 0. A fixed anchor (rather than "today") is still
# required, because training and inference must derive `age` identically.
REFERENCE_YEAR = 2022

NUMERIC_FEATURES = ("age", "mileage", "vol_engine")
LOW_CARD_CATEGORICAL = ("fuel", "province")          # one-hot
HIGH_CARD_CATEGORICAL = ("mark", "model")            # out-of-fold target encoding

# Clean fuel domain (from the dataset) — used to validate API input so nonsense fuels
# are rejected rather than silently priced.
KNOWN_FUELS = ("CNG", "Diesel", "Electric", "Gasoline", "Hybrid", "LPG")

# The one fuel for which `vol_engine == 0` is a fact rather than a missing value. Named
# because three places compare against it — the cleaning rule, the API's displacement check
# and the web form — and a literal repeated in three languages is how the province bug
# happened. See `data.has_plausible_displacement`.
ELECTRIC_FUEL = "Electric"

# Upper bounds on the free-text fields at the API boundary. They are not domain knowledge —
# the vocabulary decides what is valid — but a request has to be bounded before it is parsed,
# and the form shows the same limit rather than letting the user type past it.
MARK_MAX_LENGTH = 40
MODEL_MAX_LENGTH = 60

# The 16 Polish provinces, in correct Polish orthography (only the first element of a
# compound name is capitalised: "Kujawsko-pomorskie") — which is also how they are spelled
# in the dataset. This tuple is the single vocabulary shared by cleaning, the API and the
# web form: one spelling, defined once. A one-hot encoder cannot report an unseen category,
# it just emits an all-zero row, so a province that fails to match here would be dropped
# silently instead of rejected — hence the vocabulary is validated at the boundary and the
# raw data is normalised onto it. The dataset also carries a handful of foreign adverts
# (Berlin, Wiedeń, …); they fall outside this domain and are dropped as out-of-scope.
PROVINCES = (
    "Dolnośląskie",
    "Kujawsko-pomorskie",
    "Lubelskie",
    "Lubuskie",
    "Łódzkie",
    "Małopolskie",
    "Mazowieckie",
    "Opolskie",
    "Podkarpackie",
    "Podlaskie",
    "Pomorskie",
    "Śląskie",
    "Świętokrzyskie",
    "Warmińsko-mazurskie",
    "Wielkopolskie",
    "Zachodniopomorskie",
)

# --- Outlier rules (domain-based; documented) --------------------------------
PRICE_MIN = 1_000.0
PRICE_MAX = 1_000_000.0
MILEAGE_MAX = 1_000_000.0
VOL_ENGINE_MIN = 0.0  # allow EVs (vol_engine == 0)
VOL_ENGINE_MAX = 8_000.0
AGE_MAX = 40

# --- Model / evaluation ------------------------------------------------------
CV_FOLDS = 5
RANDOM_STATE = 42
# No TEST_SIZE: model selection is k-fold over pooled out-of-fold predictions, so there is no
# single holdout split to size. A leftover constant here would suggest one exists.
