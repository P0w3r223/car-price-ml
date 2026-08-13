"""The published mini site: aggregates exported from the model, then rendered to HTML.

``export`` writes ``docs/data/*.json`` and ``build`` renders the page from them. ``form``
writes the two files under ``docs/app/`` that the valuation form must not restate by hand.
The version below is the contract between the exporter and the renderer, kept here so the
renderer can check it without importing the exporter's model stack.
"""

from __future__ import annotations

from pathlib import Path

# Bump when the shape of any exported aggregate changes. The page build refuses a file
# stamped with anything else rather than rendering whatever fields it recognises — the same
# reason `model.load_model` refuses an artifact fit on a different feature set.
AGGREGATE_SCHEMA = 1

ASSET_DIR = Path(__file__).parent / "assets"


def stylesheet(*parts: str) -> str:
    """Concatenate stylesheet parts, in the cascade order given.

    Both published pages are built from ``tokens.css`` plus what only they need, so the
    palette is defined once and neither page can drift into its own colour scheme. The parts
    are read with universal newlines, so a CRLF checkout still produces the LF bytes the
    committed outputs are compared against.
    """
    blocks = [(ASSET_DIR / part).read_text(encoding="utf-8").rstrip() for part in parts]
    return "\n".join(blocks) + "\n"
