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

# Fallback CV results (PLN) used only if the model bundle carries no cv_all metadata.
_BAKEOFF_FALLBACK = {
    "RandomForest": {"mae": 8616, "mape": 14.3, "r2": 0.944},
    "LightGBM": {"mae": 9244, "mape": 14.8, "r2": 0.941},
    "Ridge": {"mae": 15612, "mape": 23.1, "r2": 0.843},
}


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


def _bakeoff_rows(bakeoff: dict) -> str:
    rows = sorted(bakeoff.items(), key=lambda kv: kv[1].get("mae", 0))
    return "".join(
        f"<tr><td>{name}</td><td>{m.get('mae', 0):,.0f}</td>"
        f"<td>{m.get('mape', 0)}%</td><td>{m.get('r2', 0)}</td></tr>"
        for name, m in rows
    )


def generate_report(output_path: Path | None = None) -> Path:
    """Build the HTML mini site and write it to ``output_path``."""
    output_path = output_path or _DEFAULT_REPORT_PATH
    try:
        bundle = model_module.load_model()
        loaded_model = bundle["model"]
        bakeoff = bundle["metadata"].get("cv_all") or _BAKEOFF_FALLBACK
    except FileNotFoundError:
        loaded_model, bakeoff = None, _BAKEOFF_FALLBACK
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
  <strong>What this is.</strong> A full ML pipeline that predicts used-car prices from an
  open dataset of ~118k Polish adverts (CC0): EDA → feature engineering (log-price,
  <code>age</code>, out-of-fold target encoding) → a model bake-off → a FastAPI
  <code>/predict</code> service. Built to be defensible end-to-end.
  {_example_prediction(loaded_model)}
</div>

<h2>Model bake-off (5-fold CV, PLN)</h2>
<table>
  <tr><th>Model</th><th>MAE</th><th>MAPE</th><th>R²</th></tr>
  {_bakeoff_rows(bakeoff)}
</table>
<p>Tree ensembles roughly halve the linear baseline's error — the depreciation curve is
non-linear:</p>
{depr_img}

<h2>What drives a valuation? (SHAP)</h2>
{shap_img}
<p>Age, model, engine size, make and mileage dominate — the model learns depreciation and
the brand/model premium.</p>

<div class="card">
  <strong>Methodology &amp; limitations.</strong> Log-price target (inverted before
  metrics), <code>age</code> not raw year, out-of-fold target encoding (no leakage),
  k-fold CV, SHAP over impurity importance. Prices are ~2021–2023, so the model is
  historically biased; the dataset lacks power/gearbox. See the repo's research doc.
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
