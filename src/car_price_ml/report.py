"""Generate a self-contained mini site (GitHub Pages) describing the project + results.

Explains what car-price-ml is, shows the model bake-off, an example valuation, and the
SHAP drivers. Charts (SHAP, depreciation) are embedded as base64 so the page is standalone.
"""

from __future__ import annotations

import base64
from datetime import datetime
from pathlib import Path

import pandas as pd

from car_price_ml import config
from car_price_ml import model as model_module

_DEFAULT_REPORT_PATH = config.PROJECT_ROOT / "reports" / "site" / "index.html"

# There is deliberately no fallback metrics table. The previous one held the pre-deduplication
# numbers, so an artifact missing its metadata produced a page showing better-than-real
# accuracy, rendered identically to a genuine one. A page that says "unavailable" is worth
# more than a page that quietly makes the model look good.


def _img_tag(path: Path, alt: str) -> str:
    if not path.exists():
        return f"<p><em>{alt} — chart not generated.</em></p>"
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f'<img src="data:image/png;base64,{b64}" alt="{alt}">'


def _example_prediction(model) -> str:
    if model is None:
        return ""
    row = pd.DataFrame([{
        "mark": "opel", "model": "combo", "fuel": "Diesel", "province": "Mazowieckie",
        "age": config.REFERENCE_YEAR - 2015, "mileage": 139568, "vol_engine": 1248,
    }])
    price = float(model.predict(row)[0])
    return (
        f"<p>Example — a 2015 Opel Combo (1.2 diesel, 140k km): "
        f"<strong>{price:,.0f} PLN</strong> (actual advert: 35,900 PLN).</p>"
    )


def _bakeoff_table(bakeoff: dict | None) -> str:
    """The model comparison, or an honest gap where it would be."""
    if not bakeoff:
        return ("<p><em>Metrics unavailable — the model artifact carries no cross-validation "
                "metadata. Retrain to populate this table.</em></p>")
    # Missing MAE sorts last, not first: an absent metric must never look like the winner.
    rows = sorted(bakeoff.items(), key=lambda kv: kv[1].get("mae", float("inf")))
    body = "".join(
        f"<tr><td>{name}</td><td>{_mae_cell(m)}</td>"
        f"<td>{m.get('mape', '—')}%</td><td>{m.get('r2', '—')}</td></tr>"
        for name, m in rows
    )
    return ("<table><tr><th>Model</th><th>MAE</th><th>MAPE</th><th>R²</th></tr>"
            f"{body}</table>")


def _mae_cell(metrics: dict) -> str:
    """MAE with its fold-to-fold spread, so the ranking carries an error bar."""
    mae = metrics.get("mae")
    if mae is None:
        return "—"
    spread = metrics.get("mae_fold_std")
    return f"{mae:,.0f} ± {spread:,.0f}" if spread else f"{mae:,.0f}"


def generate_report(output_path: Path | None = None) -> Path:
    """Build the HTML mini site and write it to ``output_path``."""
    output_path = output_path or _DEFAULT_REPORT_PATH
    try:
        bundle = model_module.load_model()
        loaded_model = bundle["model"]
        bakeoff = bundle["metadata"].get("cv_all")
        n_train = bundle["metadata"].get("n_train")
    except (FileNotFoundError, ValueError):
        loaded_model, bakeoff, n_train = None, None, None
    # Read from the artifact rather than hardcoded: the cleaning rules (deduplication,
    # out-of-domain geography, missing displacement) move this number, and a stale count in
    # the report is the kind of small dishonesty that costs credibility.
    trained_on = f"{n_train:,} cleaned Polish adverts" if n_train else "cleaned Polish adverts"
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    shap_img = _img_tag(config.FIGURES_DIR / "fig3_shap.png", "SHAP feature importance")
    depr_img = _img_tag(config.FIGURES_DIR / "fig2_depreciation.png", "Price vs age and mileage")

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>car-price-ml — Polish used-car price model</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  body {{ font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
         -webkit-font-smoothing: antialiased; text-rendering: optimizeLegibility;
         max-width: 860px; margin: 2rem auto; padding: 0 1rem; color: #1c2430; line-height: 1.5; }}
  h1 {{ margin-bottom: 0.2rem; font-weight: 700; letter-spacing: -0.01em; }}
  .sub {{ color: #667085; margin-top: 0; }}
  .card {{ background: #f6f8fa; border-radius: 10px; padding: 1rem 1.2rem; margin: 1.2rem 0; }}
  img {{ max-width: 100%; height: auto; }}
  table {{ border-collapse: collapse; margin: 1rem 0; }}
  td, th {{ border: 1px solid #ddd; padding: 0.4rem 0.8rem; text-align: right; }}
  th:first-child, td:first-child {{ text-align: left; }}
  code {{ background: #eef; padding: 0.1rem 0.3rem; border-radius: 4px; }}
  footer {{ color: #888; font-size: 0.85rem; margin-top: 2rem; }}
</style>
</head>
<body>
<h1>car-price-ml</h1>
<p class="sub">Used-car price prediction for the Polish market</p>

<div class="card">
  <strong>Overview.</strong> A complete ML pipeline that predicts used-car prices, trained on
  {trained_on} from an open dataset (CC0): EDA → feature engineering (log-price,
  <code>age</code>, out-of-fold target encoding) → a model comparison → a FastAPI
  <code>/predict</code> service. The pipeline is designed to be defensible end-to-end.
  {_example_prediction(loaded_model)}
</div>

<h2>Model comparison (5-fold CV, PLN)</h2>
{_bakeoff_table(bakeoff)}
<p class="sub">MAE is shown with its fold-to-fold standard deviation: a gap smaller than the
spread is not evidence of a better model.</p>
<p>Tree ensembles reduce the linear baseline's error by approximately half; the depreciation
curve is non-linear:</p>
{depr_img}

<h2>Valuation drivers (SHAP)</h2>
{shap_img}
<p>Age, model, engine size, make and mileage are the dominant features; the model captures
depreciation and the brand/model premium.</p>

<div class="card">
  <strong>Methodology and limitations.</strong> Log-price target (inverted before
  metrics), <code>age</code> not raw year, out-of-fold target encoding (no leakage),
  k-fold CV, SHAP over impurity importance. The data is a single January 2022 snapshot, so
  every valuation is a January 2022 valuation and is labelled as one rather than rescaled to
  today; the dataset lacks power/gearbox. See the repo's research docs.
</div>

<footer>
  Generated {generated} ·
  <a href="https://github.com/P0w3r223/car-price-ml">source on GitHub</a> ·
  Data © the Kaggle dataset author (CC0-1.0)
</footer>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


if __name__ == "__main__":
    print("wrote", generate_report())
