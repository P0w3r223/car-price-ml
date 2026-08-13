"""Charts as inline SVG — pure functions from rows to markup.

SVG rather than an embedded PNG for three reasons that matter here: it inherits the
reader's colour scheme, so dark mode is not a second rendering; it stays sharp at any size;
and its text is real text, so a screen reader can read the figures. The previous page
carried two base64 matplotlib images and was 233 KB of HTML, none of which a reader could
select, search or restyle.

Every chart carries the number it plots. The comparison chart additionally refuses to draw
a MAE without its fold-to-fold spread: this project decides which model to serve on gaps of
a few hundred złoty, and a bar without an error bar is exactly the picture that makes such a
gap look decisive.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape

_WIDTH = 720
_ROW_HEIGHT = 30
_LABEL_WIDTH = 130
_PAD = 12
_VALUE_ROOM = 210  # the value label sits outside the bar, and it is a long one here

# Geometry of the depreciation curve, which is plotted rather than laid out in rows.
_CURVE_HEIGHT = 250
_CURVE_LEFT = 74
_CURVE_RIGHT = 24
_CURVE_TOP = 24
_CURVE_BASELINE = 200


class IncompleteFigure(RuntimeError):
    """Raised when a figure would be published without the uncertainty it rests on."""


@dataclass(frozen=True)
class SpreadBar:
    """One model in the bake-off: its pooled MAE, and how much that moved between folds."""

    label: str
    value: float
    # Optional in the type only. A missing spread is not a chart with one bar undrawn — it
    # stops the build, below.
    spread: float | None
    note: str  # e.g. the artifact size — the other half of the decision
    served: bool = False


@dataclass(frozen=True)
class Bar:
    label: str
    value: float
    note: str | None = None


@dataclass(frozen=True)
class Point:
    """One age bucket: what the median advert cost, and how many adverts stand behind it."""

    x: float
    y: float
    n: int


def _text(value) -> str:
    return escape(str(value), quote=True)


def _thousands(value: float) -> str:
    return f"{value:,.0f}".replace(",", " ")


def _svg(width: int, height: int, title: str, body: str) -> str:
    return (
        f'<svg class="chart" viewBox="0 0 {width} {height}" width="{width}" '
        f'height="{height}" role="img" aria-label="{_text(title)}" '
        f'xmlns="http://www.w3.org/2000/svg">'
        f"<title>{_text(title)}</title>{body}</svg>"
    )


def bakeoff_chart(bars: list[SpreadBar], title: str = "Cross-validated error by model") -> str:
    """Pooled out-of-fold MAE per model, each with its fold-to-fold spread.

    The spread is drawn as a whisker on the bar end rather than printed only in text, because
    the comparison this chart exists to support turns on whether two bars differ by more than
    their own wobble — and bars alone always look decisive. Whether they do differ is a
    question for the data, so the sentence beside the chart is derived from it
    (``build._accuracy_verdict``) rather than written here.
    """
    # No "no data" fallback here, unlike the other two charts. This one carries the decision
    # the page is built around; a page that renders without it is not a lesser page, it is a
    # different claim. The build stops instead.
    if not bars:
        raise IncompleteFigure("no cross-validated metrics to compare")
    missing = [bar.label for bar in bars if bar.spread is None]
    if missing:
        raise IncompleteFigure(f"MAE without a fold spread for {missing}")

    plot_width = _WIDTH - _LABEL_WIDTH - _VALUE_ROOM
    largest = max(bar.value + bar.spread for bar in bars) or 1.0
    height = len(bars) * _ROW_HEIGHT + _PAD * 2

    def x_of(value: float) -> float:
        return _LABEL_WIDTH + max(0.0, min(1.0, value / largest)) * plot_width

    parts = []
    for index, bar in enumerate(bars):
        y = _PAD + index * _ROW_HEIGHT
        mid = y + _ROW_HEIGHT / 2 - 3
        classes = "bar served" if bar.served else "bar"
        end = x_of(bar.value)
        parts.append(
            f'<text class="bar-label" x="{_LABEL_WIDTH - 8}" y="{mid + 4}" '
            f'text-anchor="end">{_text(bar.label)}</text>'
            f'<rect class="{classes}" x="{_LABEL_WIDTH}" y="{y + 5}" '
            f'width="{end - _LABEL_WIDTH:.1f}" height="{_ROW_HEIGHT - 14}" rx="2"></rect>'
            # End caps, not a bare line. At this scale ±72 PLN is under four pixels, most of
            # it hidden under the bar it sits on — a reader told to read the whiskers before
            # the ranking could not see them at all.
            f'<line class="spread" x1="{x_of(bar.value - bar.spread):.1f}" '
            f'x2="{x_of(bar.value + bar.spread):.1f}" y1="{mid}" y2="{mid}"></line>'
            f'<line class="spread-cap" x1="{x_of(bar.value - bar.spread):.1f}" '
            f'x2="{x_of(bar.value - bar.spread):.1f}" y1="{mid - 5}" y2="{mid + 5}"></line>'
            f'<line class="spread-cap" x1="{x_of(bar.value + bar.spread):.1f}" '
            f'x2="{x_of(bar.value + bar.spread):.1f}" y1="{mid - 5}" y2="{mid + 5}"></line>'
            f'<text class="bar-value" x="{x_of(bar.value + bar.spread) + 8:.1f}" '
            f'y="{mid + 4}">{_thousands(bar.value)} ± {_thousands(bar.spread)}'
            f" · {_text(bar.note)}</text>"
        )
    return _svg(_WIDTH, height, f"{title} (PLN)", "".join(parts))


def driver_chart(bars: list[Bar], title: str) -> str:
    """Mean |SHAP| per feature — how much each feature moves a valuation, on average.

    Deliberately not a signed waterfall in złoty. The explainer runs on the regressor inside
    the ``TransformedTargetRegressor``, which is fit on ``log1p(price)``, so these are
    log-price contributions: they sum to ``log1p(prediction)`` and become multiplicative
    under ``expm1``. A bar reading "+8 400 PLN for age" would be false in the specific way
    this project exists to avoid.
    """
    if not bars:
        return '<p class="empty">No SHAP aggregate exported.</p>'

    plot_width = _WIDTH - _LABEL_WIDTH - 110
    largest = max(bar.value for bar in bars) or 1.0
    height = len(bars) * _ROW_HEIGHT + _PAD * 2

    parts = []
    for index, bar in enumerate(bars):
        y = _PAD + index * _ROW_HEIGHT
        mid = y + _ROW_HEIGHT / 2 - 3
        width = max(1.0, bar.value / largest * plot_width)
        note = f" · {bar.note}" if bar.note else ""
        parts.append(
            f'<text class="bar-label" x="{_LABEL_WIDTH - 8}" y="{mid + 4}" '
            f'text-anchor="end">{_text(bar.label)}</text>'
            f'<rect class="bar" x="{_LABEL_WIDTH}" y="{y + 5}" width="{width:.1f}" '
            f'height="{_ROW_HEIGHT - 14}" rx="2"></rect>'
            f'<text class="bar-value" x="{_LABEL_WIDTH + width + 8:.1f}" y="{mid + 4}">'
            f"{bar.value:.3f}{_text(note)}</text>"
        )
    return _svg(_WIDTH, height, f"{title} (mean |SHAP|, log-price)", "".join(parts))


def curve_chart(points: list[Point], title: str, x_label: str, y_label: str) -> str:
    """Median price against age, drawn from the cleaned adverts themselves.

    A line of medians rather than a scatter of individual adverts: the scatter was 4 000
    points of PNG that said nothing the medians do not, and could not be read at all on a
    phone. Buckets are dropped by the caller before they get here, so every point on the
    line rests on enough adverts to mean something — and the thinnest bucket's `n` is
    published in the caption.
    """
    if not points:
        return '<p class="empty">No cleaned adverts to summarise.</p>'

    plot_width = _WIDTH - _CURVE_LEFT - _CURVE_RIGHT
    x_max = max(point.x for point in points) or 1.0
    y_max = max(point.y for point in points) or 1.0

    def x_of(value: float) -> float:
        return _CURVE_LEFT + (value / x_max) * plot_width

    def y_of(value: float) -> float:
        span = _CURVE_BASELINE - _CURVE_TOP
        return _CURVE_BASELINE - max(0.0, min(1.0, value / y_max)) * span

    path = " ".join(f"{x_of(p.x):.1f},{y_of(p.y):.1f}" for p in points)
    parts = [
        f'<line class="axis-line" x1="{_CURVE_LEFT}" x2="{_CURVE_LEFT + plot_width}" '
        f'y1="{_CURVE_BASELINE}" y2="{_CURVE_BASELINE}"></line>',
        f'<line class="axis-line" x1="{_CURVE_LEFT}" x2="{_CURVE_LEFT}" '
        f'y1="{_CURVE_TOP}" y2="{_CURVE_BASELINE}"></line>',
        f'<polyline class="series" points="{path}"></polyline>',
        f'<text class="axis" x="{_CURVE_LEFT - 8}" y="{_CURVE_TOP + 4}" '
        f'text-anchor="end">{_thousands(y_max)}</text>',
        f'<text class="axis" x="{_CURVE_LEFT - 8}" y="{_CURVE_BASELINE + 4}" '
        f'text-anchor="end">0</text>',
        f'<text class="axis" x="{_CURVE_LEFT}" y="{_CURVE_BASELINE + 20}">'
        f"{_text(x_label)}</text>",
        f'<text class="axis" x="{_CURVE_LEFT + plot_width}" y="{_CURVE_BASELINE + 20}" '
        f'text-anchor="end">{_thousands(x_max)}</text>',
        f'<text class="axis" x="{_CURVE_LEFT - 8}" y="{_CURVE_TOP - 10}" '
        f'text-anchor="end">{_text(y_label)}</text>',
    ]
    for point in points:
        parts.append(
            f'<circle class="series-dot" cx="{x_of(point.x):.1f}" '
            f'cy="{y_of(point.y):.1f}" r="3"></circle>'
        )
    # Only the ends are labelled. A number on every marker would be a table drawn as a
    # chart, and the numbers that matter are the two the curve runs between.
    first, last = points[0], points[-1]
    parts.append(
        f'<text class="bar-value" x="{x_of(first.x) + 8:.1f}" y="{y_of(first.y) - 8:.1f}">'
        f"{_thousands(first.y)} PLN (n={_thousands(first.n)})</text>"
        f'<text class="bar-value" x="{x_of(last.x):.1f}" y="{y_of(last.y) - 12:.1f}" '
        f'text-anchor="end">{_thousands(last.y)} PLN (n={_thousands(last.n)})</text>'
    )
    return _svg(_WIDTH, _CURVE_HEIGHT, title, "".join(parts))
