"""Generate the two files under ``docs/app/`` that the valuation form must not write by hand.

    python -m car_price_ml.site.form

The form is static JavaScript, so every bound it validates against used to be a second copy
of a value ``config.py`` owns. One of those copies drifted: the form offered
"Kujawsko-Pomorskie" while the model was trained on "Kujawsko-pomorskie", and one-hot
encoding answers a near-miss with an all-zero row rather than an error — around 7 % of the
market was priced as if the car had no location at all, with a 200 and no indication that
anything had been ignored. A test pinning the copies came after that; this replaces the
copies.

``config.json`` is emitted beside the form rather than into ``docs/data/`` on purpose. The
API mounts ``docs/app`` at the site root, so a sibling file is fetchable both there and on
GitHub Pages, while ``../data/`` exists only in the Pages layout — the form would have
loaded its own vocabulary in one deployment and refused to start in the other.

Nothing here needs the artifact or the dataset: both outputs are functions of the package's
own source, which is what lets CI regenerate them and fail on any diff.
"""

from __future__ import annotations

import json
from pathlib import Path

from car_price_ml import config
from car_price_ml.site import stylesheet

# Bump when the shape of config.json changes. The form refuses a payload stamped with
# anything else instead of validating against the fields it happens to recognise — a form
# that silently stops checking a bound is the failure this file exists to prevent.
FORM_CONFIG_SCHEMA = 1

# The form's own stylesheet: the shared palette and page frame, then what only a form needs.
# The report is built from the same first two parts, so following its "Try the valuation
# form" link no longer lands the reader on a differently-coloured site with no dark mode.
FORM_STYLESHEET_PARTS = ("tokens.css", "base.css", "form.css")


def config_payload() -> dict:
    """Everything the form validates against, read out of ``config.py``.

    Keys keep the Python names rather than being renamed to JavaScript conventions: the test
    that pins this file to the config compares them directly, and a translation table between
    two spellings of the same constant is one more place for a copy to drift.
    """
    return {
        "schema": FORM_CONFIG_SCHEMA,
        "fuels": list(config.KNOWN_FUELS),
        "provinces": list(config.PROVINCES),
        # The dataset is a single January 2022 scrape, so this is the year the form's
        # valuations are valuations *of* — and the only anchor from which `age` may be
        # derived, because training derived it the same way.
        "reference_year": config.REFERENCE_YEAR,
        "year_min": config.REFERENCE_YEAR - config.AGE_MAX,
        "year_max": config.REFERENCE_YEAR,
        "mileage_max": int(config.MILEAGE_MAX),
        "vol_engine_max": int(config.VOL_ENGINE_MAX),
        "mark_max_length": config.MARK_MAX_LENGTH,
        "model_max_length": config.MODEL_MAX_LENGTH,
        # The form applies `data.has_plausible_displacement` client-side, and that rule names
        # one fuel. Sent rather than spelled in JavaScript, so the two cannot disagree about
        # which fuel is allowed to have no engine.
        "electric_fuel": config.ELECTRIC_FUEL,
    }


def write(out_dir: Path | None = None) -> list[Path]:
    """Write ``config.json`` and ``styles.css`` where the form is served from."""
    out_dir = Path(out_dir or config.SITE_APP_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    config_path = out_dir / "config.json"
    # Explicit LF and a trailing newline, as for every other generated file here: these are
    # committed and CI regenerates them on another OS, so their bytes have to match there.
    with open(config_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(config_payload(), handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    styles_path = out_dir / "styles.css"
    with open(styles_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(stylesheet(*FORM_STYLESHEET_PARTS))

    return [config_path, styles_path]


if __name__ == "__main__":
    for path in write():
        print("wrote", path)
