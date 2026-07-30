# CLAUDE.md — car-price-ml

Guidance for Claude Code (and any contributor) working in this repository.

## What this project is

A used-car price model for the Polish market: clean an open Kaggle dataset, engineer
features, compare models, and serve predictions via FastAPI. Portfolio project A3 — the
full ML cycle end-to-end.

## Architecture

```
src/car_price_ml/
  config.py     # dataset, paths, feature groups, outlier rules, model settings
  data.py       # load + clean (pure transforms; only load_raw touches disk)
  features.py   # preprocessing: age, target encoding (OOF), one-hot, log target
  model.py      # bake-off (linear/RF/LightGBM), k-fold CV, metrics, SHAP, persistence
api/            # FastAPI /predict service
notebooks/      # EDA + feature engineering
tests/          # pytest
docs/research/  # data + methodology
```

## Methodology rules (do not violate)

- **Log-price target.** Train on `log1p(price)`; **invert with `expm1` before every
  metric** (reporting metrics on the log scale is a silent, common mistake).
- **`age`, not raw `year`.** Derive `age = REFERENCE_YEAR - year`; raw year leaks/drifts.
- **Out-of-fold target encoding** for make/model (high cardinality). Fitting the encoder
  on the full dataset is the #1 leakage source — do it inside CV folds / a Pipeline.
- **k-fold cross-validation**, not a single split, for model selection (pooled
  out-of-fold predictions across folds).
- **SHAP (TreeExplainer)** for importance, not impurity `feature_importances_` (biased
  toward high-cardinality make/model).
- Filter outliers by **documented domain rules**; keep EVs (`vol_engine == 0`).

## Conventions

- English for code, comments, README, commits. Conventional Commits.
- No hardcoded values — configurable things live in `config.py`.
- Separate I/O from logic; pure functions are unit-tested.
- Interpreter: `.venv/Scripts/python.exe` (Python 3.12).

## How to run

```bash
.venv/Scripts/python -m pip install -r requirements.txt
kaggle datasets download -d aleksandrglotov/car-prices-poland -p data/raw --unzip
pytest
```

<!-- code-review-graph MCP tools -->
## MCP Tools: code-review-graph

**IMPORTANT: This project has a knowledge graph. ALWAYS use the
code-review-graph MCP tools BEFORE using Grep/Glob/Read to explore
the codebase.** The graph is faster, cheaper (fewer tokens), and gives
you structural context (callers, dependents, test coverage) that file
scanning cannot.

### When to use graph tools FIRST

- **Exploring code**: `semantic_search_nodes_tool` or `query_graph_tool` instead of Grep
- **Understanding impact**: `get_impact_radius_tool` instead of manually tracing imports
- **Code review**: `detect_changes_tool` + `get_review_context_tool` instead of reading entire files
- **Finding relationships**: `query_graph_tool` with callers_of/callees_of/imports_of/tests_for
- **Architecture questions**: `get_architecture_overview_tool` + `list_communities_tool`

Fall back to Grep/Glob/Read **only** when the graph doesn't cover what you need.

### Key Tools

| Tool | Use when |
| ------ | ---------- |
| `detect_changes_tool` | Reviewing code changes — gives risk-scored analysis |
| `get_review_context_tool` | Need source snippets for review — token-efficient |
| `get_impact_radius_tool` | Understanding blast radius of a change |
| `get_affected_flows_tool` | Finding which execution paths are impacted |
| `query_graph_tool` | Tracing callers, callees, imports, tests, dependencies |
| `semantic_search_nodes_tool` | Finding functions/classes by name or keyword |
| `get_architecture_overview_tool` | Understanding high-level codebase structure |
| `refactor_tool` | Planning renames, finding dead code |

### Workflow

1. No hooks installed — run `code-review-graph update` after code changes.
2. Use `detect_changes_tool` for code review.
3. Use `get_affected_flows_tool` to understand impact.
4. Use `query_graph_tool` pattern="tests_for" to check coverage.
