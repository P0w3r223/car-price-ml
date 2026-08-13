"use strict";

// The served model, run here. `car_price_ml.site.browser_model` exports 1 200 trees as
// parallel arrays plus the preprocessing as an ordered plan, and refuses to write the file at
// all unless re-pricing 200 real adverts through it agrees with the Python pipeline to within
// a grosz (measured: 4e-06 PLN). This file is the other half of that claim — the interpreter
// the parity fixture is run against under Node in CI.
//
// It deliberately implements no fallback of any kind. An unknown make, an unknown province, a
// plan step it does not recognise: each throws. The one thing this project has shipped wrong
// every time is a substituted default that looked like an answer, and a model running on the
// reader's own machine is the easiest place in the system to hide one.

const MODEL_SCHEMA = 1; // must match browser_model.BROWSER_MODEL_SCHEMA
const LEAF = -1; // feature[i] === LEAF marks a leaf, whose answer is value[i]
const STEP_KINDS = ["target_encode", "one_hot", "numeric"];

class ModelError extends Error {}

/** A value outside what the model was trained on — the API answers these with a 422. */
class UnknownValue extends Error {
  constructor(field, value) {
    super(`unknown ${field}: ${JSON.stringify(value)}`);
    this.field = field;
    this.value = value;
  }
}

function checkPayload(payload) {
  if (!payload || typeof payload !== "object") throw new ModelError("model.json is not an object");
  if (payload.schema !== MODEL_SCHEMA) {
    throw new ModelError(`model.json is schema ${payload.schema}, this runtime reads ${MODEL_SCHEMA}`);
  }
  // The inverse of the target transform is named in the payload rather than assumed, because
  // a model trained on raw price would otherwise be exponentiated into nonsense that still
  // looks like a price.
  if (payload.inverse !== "expm1") {
    throw new ModelError(`model.json asks for inverse ${JSON.stringify(payload.inverse)}, which this runtime does not implement`);
  }
  if (!Array.isArray(payload.plan) || payload.plan.length === 0) {
    throw new ModelError("model.json carries no preprocessing plan");
  }
  for (const step of payload.plan) {
    if (!STEP_KINDS.includes(step.kind)) {
      throw new ModelError(`model.json plan contains an unknown step kind: ${JSON.stringify(step.kind)}`);
    }
  }
  const trees = payload.trees;
  for (const key of ["feature", "threshold", "left", "right", "value", "roots"]) {
    if (!Array.isArray(trees && trees[key])) throw new ModelError(`model.json carries no trees.${key}`);
  }
  const n = trees.feature.length;
  if (trees.threshold.length !== n || trees.left.length !== n || trees.right.length !== n
      || trees.value.length !== n) {
    throw new ModelError("model.json's tree arrays have different lengths");
  }
}

/**
 * Turn the exported payload into something that can price a car.
 * Throws ModelError on a payload this runtime cannot faithfully run.
 */
function createModel(payload) {
  checkPayload(payload);
  const source = payload.trees;
  // Typed arrays: faster to walk, and Float64 rather than Float32 on purpose — the leaf
  // values carry twelve decimals precisely so the browser and the API agree, and Float32
  // would throw that away at load.
  const trees = {
    feature: Int16Array.from(source.feature),
    threshold: Float64Array.from(source.threshold),
    left: Int32Array.from(source.left),
    right: Int32Array.from(source.right),
    value: Float64Array.from(source.value),
    roots: Int32Array.from(source.roots),
  };

  const plan = payload.plan;

  /** The car as the model's feature row, in the order the plan gives. */
  function encode(car) {
    const row = [];
    for (const step of plan) {
      const value = car[step.field];
      if (value === undefined || value === null || value === "") {
        throw new UnknownValue(step.field, value);
      }
      if (step.kind === "target_encode") {
        // Object.prototype keys ("constructor", "toString") would otherwise resolve to a
        // function and be pushed into the feature row as a non-number.
        const encoded = Object.prototype.hasOwnProperty.call(step.map, value)
          ? step.map[value] : undefined;
        if (typeof encoded !== "number") throw new UnknownValue(step.field, value);
        row.push(encoded);
      } else if (step.kind === "one_hot") {
        if (!step.categories.includes(value)) throw new UnknownValue(step.field, value);
        for (const category of step.categories) row.push(category === value ? 1 : 0);
      } else {
        const numeric = Number(value);
        if (!Number.isFinite(numeric)) throw new UnknownValue(step.field, value);
        row.push(numeric);
      }
    }
    return row;
  }

  function predict(car) {
    const row = encode(car);
    let total = 0;
    for (let t = 0; t < trees.roots.length; t += 1) {
      let node = trees.roots[t];
      while (trees.feature[node] !== LEAF) {
        node = row[trees.feature[node]] <= trees.threshold[node]
          ? trees.left[node] : trees.right[node];
      }
      total += trees.value[node];
    }
    return Math.expm1(total);
  }

  // What the model can price, read off the plan rather than restated: the same lists the
  // export took from the fitted encoders.
  const vocabulary = {};
  for (const step of plan) {
    if (step.kind === "target_encode") vocabulary[step.field] = Object.keys(step.map);
    else if (step.kind === "one_hot") vocabulary[step.field] = step.categories.slice();
  }

  return {
    predict,
    encode,
    vocabulary,
    servedModel: payload.served_model,
    trainedAt: payload.trained_at,
    trainedOn: payload.n_train,
    referenceYear: payload.reference_year,
    treeCount: trees.roots.length,
  };
}

// Loaded as a plain <script> in the browser and required by the parity test under Node. The
// shim is three lines rather than a build step, because this file has to be the *same* file
// in both places for the parity test to mean anything.
if (typeof module !== "undefined" && module.exports) {
  module.exports = { createModel, ModelError, UnknownValue, MODEL_SCHEMA };
}
