# Data & Methodology — car-price-ml

Date: 2026-07-17
Status: accepted
Author: P0w3r223 + Claude
Related to: project A3 (used-car price model)

---

Synthesis of two research passes: where the data comes from (and why not scraping) and
how to build the model correctly.

## Data source — an open dataset, not scraping

For a portfolio project the safest and cleanest path is a **ready, openly published
dataset** rather than scraping. Scraping otomoto/OLX stacks up risk: the EU *sui generis*
database right (a full 100k+ base is a "substantial part"), contractual ToS bans
(Ryanair v. PR Aviation), and GDPR (seller data). Open Kaggle datasets — all sourced from
otomoto.pl — cover almost every feature we need.

| Dataset | Rows | Notes |
|---------|------|-------|
| **Car Prices Poland** (aleksandrglotov) — **primary** | ~117k | Clean, popular, no personal data. Columns: `mark`, `model`, `generation_name`, `year`, `mileage`, `vol_engine`, `fuel`, `city`, `province`, `price`. Missing power/gearbox. |
| Poland cars for sale (bartoszpieniak) | ~208k | Richer (has **power HP**, engine displacement) but rawer; scraper code public. |

**Decision:** start with *Car Prices Poland* (clean, benchmarkable). Cite the author +
license in the README. **Verify the exact Kaggle license on the dataset page before
publishing.** Prices are from ~2021–2023 — the model is historically biased; document it.

## Methodology (the load-bearing part)

**Target.** Car prices are strongly right-skewed → train on **`log1p(price)`**; invert
with `expm1` **before** computing any metric in PLN. Reporting metrics on the log scale
is a common silent mistake.

**Features.**
- Derive **`age = reference_year − year`**, not raw `year` — raw year leaks/drifts when
  the model is used in a later calendar year.
- Depreciation is **non-linear** (front-loaded) in age and mileage — linear models need
  transforms; tree ensembles capture it natively.
- **High-cardinality `mark`/`model`** is the central design choice: one-hot explodes.
  Use **out-of-fold target encoding** (k-fold, with smoothing for rare categories) — the
  #1 leakage source if fit on the full data. Consider hierarchical grouping (mark, then
  mark+model) and bucketing rare models.
- Low-cardinality categoricals (fuel, gearbox, body) → one-hot.
- **Outliers:** drop implausible rows by domain rules (price≈0, mileage in millions,
  age > ~40) — document every rule.

**Models (bake-off).** Linear/Ridge on log-price (honest baseline) → RandomForest →
**LightGBM** (expected winner). Trees beat linear here because the target is irregular and
interaction-heavy (Grinsztajn et al., NeurIPS 2022, on tabular data).

**Validation & metrics.** **k-fold CV** (mean ± std), not a single split. Report **MAE**
(primary, PLN), **RMSE**, **MAPE** (business-intuitive %), **R²** — all in **PLN** (invert
log first).

**Interpretability.** **SHAP TreeExplainer** (beeswarm + a few waterfalls), not
impurity-based `feature_importances_` (biased toward high-cardinality mark/model).

**Junior mistakes to avoid.** Leakage via preprocessing before the split (wrap in a
`Pipeline`/`ColumnTransformer`, out-of-fold target encoding); ignoring price skew;
metrics on the log scale; single split; raw `year`; trusting impurity importance;
unsmoothed rare-category encoding.

## Sources

- Kaggle: aleksandrglotov/car-prices-poland · bartoszpieniak/poland-cars-for-sale-dataset
- Grinsztajn, Oyallon, Varoquaux, *Why do tree-based models still outperform deep learning
  on tabular data?* NeurIPS 2022
- Target encoding leakage (H2O / Train in Data); log-target transform (Wilhelm);
  SHAP TreeExplainer; EU database right, Ryanair v. PR Aviation, GDPR art. 14
