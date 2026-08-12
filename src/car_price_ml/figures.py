"""Regenerate the report figures from the current data and model.

    python -m car_price_ml.figures

The figures used to exist only inside the notebook, which meant the published page could
carry charts drawn from data the pipeline no longer produces — the cleaning rules moved the
row count by 6 909 without the charts noticing. Generating them from the same modules the
model uses keeps the page and the pipeline in step.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display in a script or CI run

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from car_price_ml import config, data, features, model  # noqa: E402

SAMPLE_SIZE = 4_000
SHAP_SAMPLE_SIZE = 1_000


def price_distribution(df: pd.DataFrame, out_dir: Path) -> Path:
    """Raw price against log1p(price) — the reason the model trains on the log target."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df[config.TARGET], bins=60, ax=axes[0], color="#4575b4")
    axes[0].set(title="Price (PLN) — right-skewed", xlabel="PLN")
    sns.histplot(np.log1p(df[config.TARGET]), bins=60, ax=axes[1], color="#5aae61")
    axes[1].set(title="log1p(price) — symmetric", xlabel="log1p(PLN)")
    fig.tight_layout()
    path = out_dir / "fig1_price_dist.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def depreciation(df: pd.DataFrame, out_dir: Path) -> Path:
    """Price against age and mileage — the non-linearity that beats the linear baseline."""
    sample = df.sample(min(SAMPLE_SIZE, len(df)), random_state=config.RANDOM_STATE)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.scatterplot(data=sample, x="age", y=config.TARGET, ax=axes[0], s=8, alpha=0.3)
    axes[0].set(title="Price vs age", ylabel="PLN")
    sns.scatterplot(data=sample, x="mileage", y=config.TARGET, ax=axes[1], s=8, alpha=0.3)
    axes[1].set(title="Price vs mileage", ylabel="PLN")
    fig.tight_layout()
    path = out_dir / "fig2_depreciation.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def shap_summary(df: pd.DataFrame, out_dir: Path) -> Path:
    """SHAP beeswarm over a LightGBM fit.

    The axis is relabelled deliberately. The explainer runs on the regressor *inside* the
    ``TransformedTargetRegressor``, which is fit on ``log1p(price)``, so the contributions
    are log-price contributions — they sum to ``log1p(prediction)``, not to a number of
    złoty. `shap`'s default label ("impact on model output") is true but invites a reader to
    add the bars up in PLN, and under ``expm1`` those contributions are multiplicative
    rather than additive.
    """
    import shap

    x, y = features.prepare(df)
    # Explain the model that is actually served, read from the artifact — a hardcoded name
    # here would keep drawing the old chart the first time the bake-off picks differently,
    # and the page would explain a model nobody is running.
    served = model.load_model()["metadata"]["model"]
    fitted = model.train(x, y, name=served)
    x_sample = x.sample(min(SHAP_SAMPLE_SIZE, len(x)), random_state=config.RANDOM_STATE)
    shap_values, x_trans, names = model.shap_explanation(fitted, x_sample)

    shap.summary_plot(shap_values, x_trans, feature_names=names, show=False)
    figure = plt.gcf()
    figure.set_size_inches(9, 5)
    figure.axes[0].set_xlabel("SHAP value — contribution to log-price (not PLN)")
    plt.tight_layout()
    path = out_dir / "fig3_shap.png"
    figure.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return path


def generate_figures(out_dir: Path = config.FIGURES_DIR) -> list[Path]:
    """Write every report figure; returns the paths written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    df = data.load_clean()
    return [
        price_distribution(df, out_dir),
        depreciation(df, out_dir),
        shap_summary(df, out_dir),
    ]


if __name__ == "__main__":
    for written in generate_figures():
        print("wrote", written)
