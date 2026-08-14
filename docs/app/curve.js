"use strict";

// The what-if curve: the car currently in the form, priced by the local model at every age
// the model was trained on. Drawn as SVG with the same class names `charts.py` uses on the
// report, so both halves of the site look like one system and the stylesheet is shared rather
// than restated.
//
// It draws only what it was given. Where the report's charts refuse to publish a value
// without its spread, this one refuses to draw a curve it cannot label: an empty or
// single-point series returns an explicit "nothing to draw" rather than an axis implying a
// measurement.

const WIDTH = 640;
const HEIGHT = 260;
const PADDING = { top: 16, right: 16, bottom: 34, left: 64 };

/** Round numbers a reader recognises — the same 1/2/2.5/5 progression charts.py uses. */
function niceStep(span, target) {
  // A zero span would take log10(0) to -Infinity, make `magnitude` zero and leave the axis
  // with no ticks at all. It cannot happen for real prices, which is exactly why it would be
  // found by a reader rather than by us.
  if (!(span > 0)) return 1;
  const rough = span / Math.max(1, target);
  const magnitude = Math.pow(10, Math.floor(Math.log10(rough)));
  for (const multiple of [1, 2, 2.5, 5, 10]) {
    if (magnitude * multiple >= rough) return magnitude * multiple;
  }
  return magnitude * 10;
}

function ticks(low, high, count) {
  const step = niceStep(high - low, count);
  const first = Math.ceil(low / step) * step;
  const out = [];
  for (let value = first; value <= high + step * 1e-9; value += step) out.push(value);
  return out;
}

function svgElement(name, attributes, text) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", name);
  for (const [key, value] of Object.entries(attributes)) {
    node.setAttribute(key, String(value));
  }
  if (text !== undefined) node.textContent = text;
  return node;
}

/**
 * Draw `points` ([{x, y}]) into `container`, marking the point at `highlight.x`.
 * `format` turns a y value into the label the axis and the marker show.
 */
function drawCurve(container, points, { highlightX, xLabel, format }) {
  container.replaceChildren();
  if (points.length < 2) {
    container.appendChild(Object.assign(document.createElement("p"), {
      className: "note",
      textContent: "Nothing to draw yet — fill the form in.",
    }));
    return;
  }

  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  // The y axis starts at zero: a curve about what a car *loses* would exaggerate the loss on
  // a truncated axis, and this one is read by someone deciding what their car is worth.
  const yMax = Math.max(...ys);
  const plotWidth = WIDTH - PADDING.left - PADDING.right;
  const plotHeight = HEIGHT - PADDING.top - PADDING.bottom;
  const toX = (x) => PADDING.left + (xMax === xMin ? 0 : ((x - xMin) / (xMax - xMin)) * plotWidth);
  const toY = (y) => PADDING.top + plotHeight - (yMax === 0 ? 0 : (y / yMax) * plotHeight);

  const svg = svgElement("svg", {
    class: "chart", viewBox: `0 0 ${WIDTH} ${HEIGHT}`, role: "img",
    "aria-label": `Modelled price by ${xLabel} for the car in the form`,
  });

  for (const value of ticks(0, yMax, 4)) {
    svg.appendChild(svgElement("line", {
      class: "grid", x1: PADDING.left, x2: WIDTH - PADDING.right, y1: toY(value), y2: toY(value),
    }));
    svg.appendChild(svgElement("text", {
      class: "axis", x: PADDING.left - 8, y: toY(value) + 3.5, "text-anchor": "end",
    }, format(value)));
  }
  for (const value of ticks(xMin, xMax, 6)) {
    svg.appendChild(svgElement("text", {
      class: "axis", x: toX(value), y: HEIGHT - PADDING.bottom + 18, "text-anchor": "middle",
    }, String(value)));
  }
  svg.appendChild(svgElement("line", {
    class: "axis-line", x1: PADDING.left, x2: WIDTH - PADDING.right,
    y1: toY(0), y2: toY(0),
  }));
  svg.appendChild(svgElement("text", {
    class: "axis", x: PADDING.left + plotWidth / 2, y: HEIGHT - 4, "text-anchor": "middle",
  }, xLabel));

  svg.appendChild(svgElement("path", {
    class: "series",
    d: points.map((point, index) => `${index ? "L" : "M"}${toX(point.x)},${toY(point.y)}`).join(" "),
  }));

  const marked = points.find((point) => point.x === highlightX);
  if (marked) {
    svg.appendChild(svgElement("line", {
      class: "grid", x1: toX(marked.x), x2: toX(marked.x), y1: PADDING.top, y2: toY(0),
    }));
    svg.appendChild(svgElement("circle", {
      class: "series-dot", cx: toX(marked.x), cy: toY(marked.y), r: 4.5,
    }));
    const anchor = toX(marked.x) > PADDING.left + plotWidth * 0.6 ? "end" : "start";
    svg.appendChild(svgElement("text", {
      class: "bar-value", x: toX(marked.x) + (anchor === "end" ? -8 : 8),
      y: toY(marked.y) - 10, "text-anchor": anchor,
    }, format(marked.y)));
  }
  container.appendChild(svg);
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = { drawCurve, ticks, niceStep };
}
