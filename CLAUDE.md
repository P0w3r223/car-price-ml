# CLAUDE.md — car-price-ml

Guidance for Claude Code (and any contributor) working in this repository.

## What this project is

A used-car price model for the Polish market: clean an open Kaggle dataset, engineer
features, compare models, and serve predictions via FastAPI. Portfolio project A3 — the
full ML cycle end-to-end.

Its value is being defensible in a technical interview, so the worst outcome available is a
confidently-presented wrong number — worse than a crash, worse than a gap in the feature
set. Most of the bugs found here have been instances of one shape: an input the model could
not handle was quietly substituted with something plausible instead of refused.

## Architecture

```
src/car_price_ml/
  config.py     # dataset, paths, feature groups, outlier rules, vocabularies, model settings
  data.py       # load + clean (pure transforms; only load_raw touches disk)
  features.py   # preprocessing: target encoding (OOF), one-hot, declared domains
  model.py      # bake-off, k-fold CV, metrics, SHAP, artifact contract, persistence
  train.py      # entrypoint: vintage check → bake-off → train the winner → save
  figures.py    # report figures, regenerated from the current data and model
  report.py     # generates the mini site from the artifact
api/            # FastAPI /predict + /vocabulary service
tests/          # pytest
docs/adr/       # design decisions with their measurements
docs/research/  # data sources, methodology, temporal recalibration
```

## Methodology rules

**Target and features**

- Train on `log1p(price)`; invert with `expm1` before every metric. Reporting metrics on the
  log scale is the classic silent error here.
- Derive `age = REFERENCE_YEAR - year` rather than using raw `year`, which drifts.
  `REFERENCE_YEAR` is the source snapshot's vintage (2022 — a single January 2022 scrape),
  not the current year. `train.py` checks it against the **raw** frame: cleaning drops
  everything outside `age ∈ [0, AGE_MAX]`, so a newer dataset would be silently truncated if
  the check ran later.
- Out-of-fold target encoding for `mark`/`model`. Fitting the encoder on the full dataset is
  the primary leakage source — keep it inside a Pipeline, with a seeded splitter.

**Unknown input fails loudly**

- Both closed one-hot domains (`fuel`, `province`) are **declared** in `config.py`, not
  learned from whatever a training sample happened to contain, and the encoder is set to
  raise on anything outside them. Under `handle_unknown="ignore"` an unseen category becomes
  an all-zero row — a combination present in no training row, priced by extrapolation and
  returned with a 200.
- `mark`/`model` have the same problem in a different disguise: `TargetEncoder` answers an
  unseen category with the global target mean, so an unknown car returns a confident,
  fictional price. The vocabulary is read off the fitted encoder, stamped into the artifact,
  served at `GET /vocabulary`, and enforced at the API boundary.
- Spelling variants are normalised (case, diacritics, separators); different vocabularies are
  rejected. "Kujawsko-Pomorskie" is a variant of a known province; "Petrol" is a different
  word for a known fuel and is refused rather than guessed.

**Measuring**

- k-fold cross-validation with pooled out-of-fold predictions, reported with the
  fold-to-fold spread. A gap smaller than the spread is not a better model.
- The served model is whichever one wins the bake-off; `train.py` selects it from the CV
  rather than hardcoding a name, and logs loudly if a pin overrides the measurement.
- SHAP (TreeExplainer) over impurity importance, which is biased toward high-cardinality
  `mark`/`model`. Note the explainer runs on the regressor *inside* the
  `TransformedTargetRegressor`, so its values are log-price contributions summing to
  `log1p(prediction)` — they are not złoty and do not stay additive under `expm1`.
- The artifact carries its own contract — feature spec, age anchor, and vocabularies — and
  `load_model` refuses a bundle that disagrees with the code. A stale artifact fails at
  load, not at the first valuation.

**Cleaning** (documented domain rules; see `data.py`)

- Deduplicate: ~9.8 % of the raw rows are repeated adverts, and with shuffled k-fold the
  same advert lands in train and test.
- Keep EVs (`vol_engine == 0` is a fact for them) and drop combustion rows with zero
  displacement (there it means "missing").
- Drop adverts from outside the 16 Polish provinces.

## Conventions

- English for code, comments, README, commits. Conventional Commits.
- Constants and configurable values live in `config.py`; paths there are overridable by
  environment variable, because the package is installed non-editable in the container.
- Interpreter: `.venv/Scripts/python.exe` (Python 3.12).

## How to run

```bash
.venv/Scripts/python -m pip install -r requirements.txt
kaggle datasets download -d aleksandrglotov/car-prices-poland -p data/raw --unzip
pytest
python -m car_price_ml.train      # bake-off, then train and save the winner
python -m car_price_ml.figures    # redraw report figures from the current data/model
```

## Code intelligence

Two indexes exist over this repo, and which one is reachable depends on the session:

- `.codegraph/` — queried with the `codegraph_explore` MCP tool, or `codegraph explore
  "<question>"` from PowerShell. Returns verbatim source plus call paths; usually answers a
  "how does X work" or "what calls Y" question in one call, so reach for it before a
  Grep/Read loop. The CLI ships as `codegraph.cmd`, so from Git Bash it needs the extension
  (`codegraph.cmd explore ...`) — bare `codegraph` resolves only where PATHEXT applies.
- `.code-review-graph/` — its MCP server is declared in `.mcp.json`, but it does not always
  load; when its tools are absent, the CLI still works as `uvx code-review-graph <command>`.
  No hooks are installed, so run `uvx code-review-graph update` after changing code.

Grep, Glob and Read remain correct whenever the question is about text rather than structure,
or when neither index is available.

---

When something reaches the pipeline that it cannot price — an unknown make, a province it was
not trained on, an artifact from a different feature set — the answer is an error, never a
substituted default. Every wrong number this project has shipped came from a silent fallback
that looked like an answer.
