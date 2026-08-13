"""Load and clean the used-car dataset.

Only ``load_raw`` touches disk; the remaining functions are pure transforms on DataFrames,
so they are unit-testable without the file.
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

from car_price_ml import config


def _fold(value: str) -> str:
    """Key a province name for matching: case, diacritics and separators removed.

    ``"Kujawsko-Pomorskie"``, ``"kujawsko pomorskie"`` and ``"Kujawsko-pomorskie"`` all fold
    to the same key, so a caller's capitalisation or missing Polish characters cannot turn
    into a silently unmatched category. ``ł`` has no canonical decomposition, so it is
    replaced before NFKD strips the remaining combining marks.
    """
    latinised = value.replace("ł", "l").replace("Ł", "L")
    decomposed = unicodedata.normalize("NFKD", latinised)
    without_marks = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"[\s\-_]+", "-", without_marks.strip()).casefold()


_PROVINCE_BY_KEY = {_fold(name): name for name in config.PROVINCES}


def canonical_province(value: object) -> str | None:
    """Return the canonical spelling of a province, or ``None`` if it is not a Polish one.

    ``None`` covers everything outside the domain: foreign regions present in the raw data,
    missing values, and non-string input.
    """
    if not isinstance(value, str):
        return None
    return _PROVINCE_BY_KEY.get(_fold(value))


def canonical_mark_or_model(value: object) -> str | None:
    """Normalise a make or model onto the spelling the dataset uses, or ``None``.

    Unlike the two domains above there is no vocabulary here to match against: which makes
    and models exist is a property of the fitted artifact, so membership is checked where
    that artifact is loaded. This only normalises — the dataset spells every make and model
    in lower case, so "Opel" is a spelling variant of a known car rather than an unknown one,
    and priced as unknown it came back 14 % high.

    It lives here rather than in the API validator because three callers need the same
    answer: the service, the published page's refusal table, and any future browser runtime.
    A hand-copied ``.casefold()`` in each was how the province vocabulary drifted apart in
    the first place.
    """
    if not isinstance(value, str):
        return None
    normalised = value.strip().casefold()
    return normalised or None


_FUEL_BY_KEY = {name.casefold(): name for name in config.KNOWN_FUELS}


def canonical_fuel(value: object) -> str | None:
    """Return the canonical spelling of a fuel type, or ``None`` if it is outside the domain.

    Case-insensitive only. Unlike provinces, the fuel domain has no diacritics and no
    separator variants — and a genuinely different word ("Petrol" for "Gasoline") is a
    different vocabulary, not a spelling variant, so it is rejected rather than guessed at.
    """
    if not isinstance(value, str):
        return None
    return _FUEL_BY_KEY.get(value.strip().casefold())


def normalize_provinces(df: pd.DataFrame) -> pd.DataFrame:
    """Map the ``province`` column onto the canonical vocabulary (unknown → ``pd.NA``)."""
    if "province" not in df.columns:
        raise KeyError("missing column: province")
    out = df.copy()
    out["province"] = out["province"].map(lambda v: canonical_province(v) or pd.NA)
    return out


def normalize_fuels(df: pd.DataFrame) -> pd.DataFrame:
    """Map the ``fuel`` column onto the canonical vocabulary (unknown left untouched).

    Unknown values are deliberately *not* blanked: the one-hot encoder is configured to
    raise on them, which is the loud failure a new source with a different fuel vocabulary
    should get. What this fixes is casing — ``drop_missing_displacement`` compares against
    ``"Electric"`` exactly, so a source spelling it ``"electric"`` would have every EV
    deleted as a missing-displacement row, in direct violation of the rule that keeps them.
    """
    if "fuel" not in df.columns:
        raise KeyError("missing column: fuel")
    out = df.copy()
    out["fuel"] = out["fuel"].map(lambda v: canonical_fuel(v) or v)
    return out


def drop_missing_provinces(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows whose ``province`` is missing.

    Named for what it does rather than what it is used for: it is a ``notna`` filter, so it
    only removes out-of-domain adverts once ``normalize_provinces`` has mapped them to
    ``pd.NA``. Run against raw data it would happily keep ``"Berlin"``.
    """
    return df[df["province"].notna()].reset_index(drop=True)


def load_raw(path=config.DATASET_CSV) -> pd.DataFrame:
    """Read the raw CSV and drop the unnamed index column."""
    df = pd.read_csv(path)
    return df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")


def add_age(df: pd.DataFrame, reference_year: int = config.REFERENCE_YEAR) -> pd.DataFrame:
    """Derive ``age = reference_year - year`` (causal quantity; avoids year drift/leakage)."""
    out = df.copy()
    out["age"] = reference_year - out["year"]
    return out


def filter_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Drop implausible rows by documented domain rules (price/mileage/vol/age)."""
    mask = (
        df["price"].between(config.PRICE_MIN, config.PRICE_MAX)
        & (df["mileage"].between(0, config.MILEAGE_MAX))
        & df["vol_engine"].between(config.VOL_ENGINE_MIN, config.VOL_ENGINE_MAX)
        & df["age"].between(0, config.AGE_MAX)
    )
    return df[mask].reset_index(drop=True)


def drop_duplicate_adverts(df: pd.DataFrame) -> pd.DataFrame:
    """Drop repeated adverts, keeping one row per distinct advert.

    9.8 % of the raw rows sit in exact-duplicate groups — 31 groups hold 22 identical
    copies — consistent with a concatenation of per-model scrapes where some models were
    ingested twice. With shuffled k-fold, identical adverts land in both the training and
    the test fold, so every metric is optimistic. The inflation is not uniform across
    models either: measured on this dataset, duplicates flatter RandomForest by 256 PLN of
    MAE (``min_samples_leaf=3`` can form a pure leaf on three copies of one advert) against
    LightGBM's 19 PLN, which is enough to distort a comparison between them.

    Comparison is on the advert's own attributes; ``age`` is derived, so it is excluded.
    The key is raw-row equality rather than equality over the modelled features: two adverts
    differing only in `city` or `generation_name` are treated as two adverts, which leaves
    ~0.9 % of rows still identical in feature space. That residual is deliberate — collapsing
    it would discard genuine repeated listings of similar cars, which is signal about how
    common such a car is.
    """
    subset = [c for c in df.columns if c != "age"]
    return df.drop_duplicates(subset=subset).reset_index(drop=True)


def has_plausible_displacement(fuel: object, vol_engine: float) -> bool:
    """Zero displacement is a fact for an EV and a missing value for anything that burns fuel.

    The single statement of this rule. Training drops the rows that fail it and the API
    refuses the requests that fail it — stated once because the two enforcing it live in
    different modules, and a rule applied on only one side is worse than no rule: the model
    then has no combustion-with-zero-displacement example to reason from, yet still answers.
    """
    return vol_engine > 0 or fuel == "Electric"


def drop_missing_displacement(df: pd.DataFrame) -> pd.DataFrame:
    """Drop combustion rows whose ``vol_engine`` is 0 (see ``has_plausible_displacement``).

    Keeping the ~374 petrol/diesel/hybrid/LPG rows teaches the model that zero displacement
    means "expensive EV" and misprices exactly those cars. Vectorised for the 118k-row load;
    ``test_displacement_rule_agrees_across_implementations`` pins it to the scalar rule.
    """
    is_electric = df["fuel"] == "Electric"
    return df[is_electric | (df["vol_engine"] > 0)].reset_index(drop=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning: canonicalise geography, deduplicate, derive age, drop outliers."""
    canonical = normalize_fuels(normalize_provinces(df))
    domestic = drop_duplicate_adverts(drop_missing_provinces(canonical))
    return drop_missing_displacement(filter_outliers(add_age(domestic)))


def load_clean(path=config.DATASET_CSV) -> pd.DataFrame:
    """Convenience: load the raw CSV and return the cleaned frame."""
    return clean(load_raw(path))
