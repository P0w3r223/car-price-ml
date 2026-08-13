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

import math
from dataclasses import dataclass
from html import escape

_WIDTH = 720
_ROW_HEIGHT = 30
_LABEL_WIDTH = 130
_PAD = 12
_VALUE_ROOM = 210  # the value label sits outside the bar, and it is a long one here

# Geometry of the depreciation curve, which is plotted rather than laid out in rows. The
# baseline leaves room under it for the age ticks and the axis label.
_CURVE_HEIGHT = 266
_CURVE_LEFT = 86
_CURVE_RIGHT = 30
_CURVE_TOP = 30
_CURVE_BASELINE = 226
_CURVE_TICK_EVERY = 5  # years between labelled points, and between x-axis ticks
_CURVE_Y_TICKS = 4


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


def _nice_step(span: float, ticks: int) -> float:
    """A round interval close to ``span / ticks`` — 1, 2, 2.5 or 5 times a power of ten.

    Axis labels are read, not measured off, so they have to be numbers a person recognises.
    Dividing the range evenly produces gridlines at 41 574 PLN, which is worse than none.
    """
    if span <= 0:
        return 1.0
    raw = span / max(1, ticks)
    magnitude = 10 ** math.floor(math.log10(raw))
    for multiple in (1, 2, 2.5, 5, 10):
        if magnitude * multiple >= raw:
            return magnitude * multiple
    return magnitude * 10


def curve_chart(
    points: list[Point], title: str, x_label: str, y_label: str,
    tick_every: int = _CURVE_TICK_EVERY,
) -> str:
    """Median price against age, drawn from the cleaned adverts themselves.

    A line of medians rather than a scatter of individual adverts: the scatter was 4 000
    points of PNG that said nothing the medians do not, and could not be read at all on a
    phone.

    Both axes carry a scale, and every ``tick_every`` years the point carries its exact
    median. Labelling only the two ends — which is how this started — left the whole middle
    of the curve unreadable: a reader could see that a car loses value quickly and could not
    tell what a ten-year-old one costs, which is the question the chart is on the page to
    answer. Not every point is labelled either; that would be a table drawn as a chart, and
    the table would be 26 rows long.

    The bucket sizes are deliberately not drawn. They were, under each tick, and the row of
    ``n=`` cluttered the axis for a number nobody reads off a curve. The size rule the points
    obey is stated once in the caption instead: buckets under the threshold are dropped, and
    the caller refuses to publish a curve that would bridge a dropped one — so no point on
    this line rests on too few adverts, which is the guarantee a per-point count was there to
    give.
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

    parts = [
        f'<text class="axis" x="{_CURVE_LEFT - 8}" y="{_CURVE_TOP - 12}" '
        f'text-anchor="end">{_text(y_label)}</text>',
    ]

    # Horizontal gridlines at round prices. They stop at the largest round value the data
    # actually reaches, so the frame never implies headroom that was never measured.
    y_step = _nice_step(y_max, _CURVE_Y_TICKS)
    level = 0.0
    while level <= y_max:
        y = y_of(level)
        parts.append(
            f'<line class="grid" x1="{_CURVE_LEFT}" x2="{_CURVE_LEFT + plot_width}" '
            f'y1="{y:.1f}" y2="{y:.1f}"></line>'
            f'<text class="axis" x="{_CURVE_LEFT - 8}" y="{y + 4:.1f}" '
            f'text-anchor="end">{_thousands(level)}</text>'
        )
        level += y_step

    parts.append(
        f'<line class="axis-line" x1="{_CURVE_LEFT}" x2="{_CURVE_LEFT + plot_width}" '
        f'y1="{_CURVE_BASELINE}" y2="{_CURVE_BASELINE}"></line>'
        f'<line class="axis-line" x1="{_CURVE_LEFT}" x2="{_CURVE_LEFT}" '
        f'y1="{_CURVE_TOP}" y2="{_CURVE_BASELINE}"></line>'
    )

    path = " ".join(f"{x_of(p.x):.1f},{y_of(p.y):.1f}" for p in points)
    parts.append(f'<polyline class="series" points="{path}"></polyline>')

    # A labelled point is drawn larger, so the markers carrying a number are distinguishable
    # from the ones that only shape the line.
    labelled = {point.x for point in points if point.x % tick_every == 0}
    for point in points:
        marked = point.x in labelled
        classes = "series-dot" if marked else "series-dot minor"
        radius = 4 if marked else 2.5
        parts.append(
            f'<circle class="{classes}" cx="{x_of(point.x):.1f}" '
            f'cy="{y_of(point.y):.1f}" r="{radius}"></circle>'
        )

    for point in points:
        if point.x not in labelled:
            continue
        x, y = x_of(point.x), y_of(point.y)
        # The first and last labels are anchored inward so they cannot run off the frame;
        # the rest sit centred over their own marker.
        anchor = "start" if point.x == 0 else ("end" if point.x == x_max else "middle")
        parts.append(
            f'<text class="bar-value" x="{x:.1f}" y="{y - 12:.1f}" text-anchor="{anchor}">'
            f"{_thousands(point.y)} PLN</text>"
            f'<text class="axis" x="{x:.1f}" y="{_CURVE_BASELINE + 18:.1f}" '
            f'text-anchor="middle">{_thousands(point.x)}</text>'
        )

    parts.append(
        f'<text class="axis" x="{_CURVE_LEFT + plot_width}" y="{_CURVE_HEIGHT - 6}" '
        f'text-anchor="end">{_text(x_label)}</text>'
    )
    return _svg(_WIDTH, _CURVE_HEIGHT, title, "".join(parts))
