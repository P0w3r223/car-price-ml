"""Central configuration: dataset, paths, features, outlier rules, model settings.

No I/O here — only constants. Columns match the "Car Prices Poland" Kaggle dataset
(aleksandrglotov): mark, model, generation_name, year, mileage, vol_engine, fuel, city,
province, price. See docs/research/data-and-methodology.md.
"""

from __future__ import annotations

from pathlib import Path

# --- Paths -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

# --- Dataset -----------------------------------------------------------------
KAGGLE_DATASET = "aleksandrglotov/car-prices-poland"  # CC0-1.0; attribute in README
DATASET_CSV = RAW_DIR / "Car_Prices_Poland_Kaggle.csv"

# --- Target & features -------------------------------------------------------
TARGET = "price"
# Prices are right-skewed → train on log1p(price), invert with expm1 before metrics.
LOG_TARGET = True

# Reference year for deriving `age` (dataset vintage ~2021-2023). Using a fixed
# reference keeps `age` stable at inference; the constant offset does not change
# relative relationships. Documented as a known bias in the README.
REFERENCE_YEAR = 2024

NUMERIC_FEATURES = ("age", "mileage", "vol_engine")
LOW_CARD_CATEGORICAL = ("fuel", "province")          # one-hot
HIGH_CARD_CATEGORICAL = ("mark", "model")            # out-of-fold target encoding

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
TEST_SIZE = 0.2
