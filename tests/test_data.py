"""Tests for data loading and cleaning (pure transforms)."""

import pandas as pd
import pytest

from car_price_ml import config, data


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Mazowieckie", "Mazowieckie"),
        ("Kujawsko-pomorskie", "Kujawsko-pomorskie"),
        ("Kujawsko-Pomorskie", "Kujawsko-pomorskie"),   # the spelling the web form used to send
        ("warmińsko-mazurskie", "Warmińsko-mazurskie"),
        ("Warminsko-Mazurskie", "Warmińsko-mazurskie"),  # typed without Polish characters
        ("kujawsko pomorskie", "Kujawsko-pomorskie"),    # space instead of a hyphen
        ("  Śląskie  ", "Śląskie"),
        ("Lodzkie", "Łódzkie"),                          # ł has no NFKD decomposition
    ],
)
def test_canonical_province_accepts_spelling_variants(raw, expected):
    assert data.canonical_province(raw) == expected


@pytest.mark.parametrize(
    "raw", ["Berlin", "Moravian-Silesian Region", "(", "", None, float("nan"), 7]
)
def test_canonical_province_rejects_everything_outside_poland(raw):
    assert data.canonical_province(raw) is None


def test_every_declared_province_is_its_own_canonical_form():
    # Guards the vocabulary itself: a typo in config.PROVINCES would make a province
    # unreachable through normalisation.
    assert [data.canonical_province(p) for p in config.PROVINCES] == list(config.PROVINCES)


def test_normalize_provinces_rewrites_without_mutating_input():
    df = pd.DataFrame({"province": ["Kujawsko-Pomorskie", "Berlin"]})
    out = data.normalize_provinces(df)
    assert out["province"].tolist()[0] == "Kujawsko-pomorskie"
    assert pd.isna(out["province"].iloc[1])
    assert df["province"].tolist() == ["Kujawsko-Pomorskie", "Berlin"]  # input untouched


def test_normalize_provinces_requires_the_column():
    with pytest.raises(KeyError):
        data.normalize_provinces(pd.DataFrame({"price": [1000]}))


def test_drop_missing_provinces_removes_foreign_adverts_once_normalized():
    df = data.normalize_provinces(pd.DataFrame({"province": ["Mazowieckie", "Wiedeń", "Śląskie"]}))
    assert data.drop_missing_provinces(df)["province"].tolist() == ["Mazowieckie", "Śląskie"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("Diesel", "Diesel"), ("diesel", "Diesel"), ("  LPG ", "LPG"), ("gasoline", "Gasoline")],
)
def test_canonical_fuel_is_case_insensitive(raw, expected):
    assert data.canonical_fuel(raw) == expected


def test_normalize_fuels_fixes_casing_and_leaves_unknowns_to_fail_loudly():
    df = pd.DataFrame({"fuel": ["electric", "DIESEL", "Petrol"]})
    out = data.normalize_fuels(df)
    # "electric" must become "Electric", or drop_missing_displacement would delete every EV
    # as a missing-displacement row. "Petrol" is left alone for the encoder to reject.
    assert out["fuel"].tolist() == ["Electric", "Diesel", "Petrol"]


@pytest.mark.parametrize("raw", ["Petrol", "Benzyna", "banana", "", None, 7])
def test_canonical_fuel_rejects_other_vocabularies(raw):
    # "Petrol" is a different word for the same thing, not a spelling variant — guessing
    # would be the same silent-substitution failure the province work exists to remove.
    assert data.canonical_fuel(raw) is None


def test_add_age_is_reference_minus_year():
    df = pd.DataFrame({"year": [2014, 2020]})
    out = data.add_age(df, reference_year=2024)
    assert list(out["age"]) == [10, 4]
    assert "year" in df.columns  # input not mutated (age not added in place)
    assert "age" not in df.columns


def test_filter_outliers_drops_implausible_rows():
    df = pd.DataFrame({
        "price": [50000, 100, 50000, 50000, 50000],       # 100 below PRICE_MIN
        "mileage": [100000, 100000, 5_000_000, 100000, 100000],  # 5M above MILEAGE_MAX
        "vol_engine": [1600, 1600, 1600, 20000, 1600],    # 20000 above VOL_ENGINE_MAX
        "age": [10, 10, 10, 10, 50],                       # 50 above AGE_MAX
    })
    out = data.filter_outliers(df)
    assert len(out) == 1  # only the first row passes every rule


def test_clean_applies_every_rule():
    keep = {
        "price": 50000, "mileage": 100000, "vol_engine": 1600, "year": 2015,
        "province": "Kujawsko-Pomorskie", "fuel": "Diesel",
    }
    df = pd.DataFrame([
        keep,                                        # survives, and is canonicalised
        keep,                                        # exact duplicate
        {**keep, "price": 100},                      # below PRICE_MIN
        {**keep, "province": "Berlin"},              # outside Poland
        {**keep, "vol_engine": 0},                   # combustion car with missing displacement
        {**keep, "vol_engine": 0, "fuel": "electric", "mileage": 5000},  # a genuine EV, lowercased
    ])

    out = data.clean(df)

    assert "age" in out.columns
    assert out["province"].tolist() == ["Kujawsko-pomorskie", "Kujawsko-pomorskie"]
    assert out["fuel"].tolist() == ["Diesel", "Electric"]


def test_filter_outliers_allows_ev_zero_engine():
    df = pd.DataFrame({
        "price": [80000], "mileage": [50000], "vol_engine": [0], "age": [3],
    })
    assert len(data.filter_outliers(df)) == 1  # EVs (vol_engine == 0) are kept


@pytest.mark.parametrize(
    ("fuel", "vol_engine", "plausible"),
    [("Electric", 0, True), ("Electric", 1600, True), ("Diesel", 0, False),
     ("Gasoline", 0, False), ("Hybrid", 0, False), ("Diesel", 1600, True)],
)
def test_displacement_rule_agrees_across_implementations(fuel, vol_engine, plausible):
    """The scalar rule (used by the API) and the vectorised filter must not drift apart.

    Training enforced this rule and serving did not, so a diesel with no engine was priced
    11 % above the same car with one.
    """
    assert data.has_plausible_displacement(fuel, vol_engine) is plausible

    frame = pd.DataFrame([{"fuel": fuel, "vol_engine": vol_engine}])
    assert data.drop_missing_displacement(frame).empty is not plausible


def test_drop_missing_displacement_keeps_evs_and_drops_combustion_zeros():
    df = pd.DataFrame({
        "fuel": ["Electric", "Diesel", "Gasoline", "Hybrid"],
        "vol_engine": [0, 0, 1600, 0],
    })
    out = data.drop_missing_displacement(df)
    # Zero displacement is a fact for an EV and a missing value for anything that burns fuel.
    assert out["fuel"].tolist() == ["Electric", "Gasoline"]


def test_drop_duplicate_adverts_keeps_one_row_per_advert():
    advert = {"mark": "opel", "model": "combo", "year": 2015, "price": 35900}
    df = pd.DataFrame([advert, advert, {**advert, "price": 36900}, advert])
    assert len(data.drop_duplicate_adverts(df)) == 2


def test_drop_duplicate_adverts_ignores_derived_age():
    # `age` is derived from `year`, so it must not make two copies of one advert look distinct.
    advert = {"mark": "opel", "model": "combo", "year": 2015}
    df = pd.DataFrame([{**advert, "age": 9}, {**advert, "age": 7}])
    assert len(data.drop_duplicate_adverts(df)) == 1
