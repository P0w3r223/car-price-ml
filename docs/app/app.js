"use strict";

// Every bound this form validates against is fetched from config.json, which
// `car_price_ml.site.form` generates from config.py. Nothing is restated here: the one value
// this file used to spell for itself drifted from the Python by a single capital letter
// ("Kujawsko-Pomorskie"), and one-hot encoding answers a near-miss with an all-zero row —
// around 7 % of the market was priced as if the car had no location, and returned with a 200.
const CONFIG_URL = "config.json";
// Must match site/form.py FORM_CONFIG_SCHEMA. A payload of any other shape is refused rather
// than read for the fields that still happen to match.
const CONFIG_SCHEMA = 1;
const REQUIRED_LISTS = ["fuels", "provinces"];
const REQUIRED_NUMBERS = [
  "reference_year", "year_min", "year_max", "mileage_max", "vol_engine_max",
  "mark_max_length", "model_max_length",
];

// Same-origin absolute paths: real when the API serves this directory at its root, absent on
// the static GitHub Pages copy — which the banner reports before the first submit rather than
// after it.
const PREDICT_PATH = "/predict";
const VOCABULARY_PATH = "/vocabulary";
const HEALTH_PATH = "/health";

const pln = new Intl.NumberFormat("pl-PL", {
  style: "currency",
  currency: "PLN",
  maximumFractionDigits: 0,
});

let CONFIG = null;
// What will answer this form: "model" (a trained artifact is loaded), "no-model" (the service
// is up but unservable) or "absent" (no API at this origin). Probed at load, and corrected if
// a submit turns out to disagree with the probe.
let backend = "absent";

// --- Configuration ----------------------------------------------------------

class ConfigError extends Error {}

async function loadConfig() {
  let response;
  try {
    response = await fetch(CONFIG_URL);
  } catch (error) {
    throw new ConfigError(`config.json could not be fetched (${error.message})`);
  }
  if (!response.ok) throw new ConfigError(`config.json answered ${response.status}`);

  let payload;
  try {
    payload = await response.json();
  } catch {
    throw new ConfigError("config.json is not valid JSON");
  }
  if (payload.schema !== CONFIG_SCHEMA) {
    throw new ConfigError(
      `config.json is schema ${payload.schema}, this form reads ${CONFIG_SCHEMA}`
    );
  }
  // Checked field by field: a payload missing one list would leave that dropdown empty and
  // its membership test vacuously false, which reads as "nothing is a valid province".
  for (const key of REQUIRED_LISTS) {
    if (!Array.isArray(payload[key]) || payload[key].length === 0) {
      throw new ConfigError(`config.json carries no ${key}`);
    }
  }
  for (const key of REQUIRED_NUMBERS) {
    if (!Number.isFinite(payload[key])) {
      throw new ConfigError(`config.json carries no numeric ${key}`);
    }
  }
  // Membership rather than truthiness: a value outside the fuel list would make the
  // displacement rule below reject every zero-displacement car, EVs included, while
  // explaining itself with whatever that value happened to be.
  if (!payload.fuels.includes(payload.electric_fuel)) {
    throw new ConfigError("config.json's electric_fuel is not one of its fuels");
  }
  return payload;
}

// --- Rendering helpers ------------------------------------------------------

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

// Built as nodes rather than assembled as HTML: `detail` comes from the service, and a
// valuation form has no business interpreting anything it is told as markup.
function renderCard(target, className, label, children) {
  target.hidden = false;
  target.className = className;
  target.replaceChildren(element("span", label.className, label.text), ...children);
}

function renderStatus(kind) {
  const status = document.getElementById("status");
  const states = {
    model: {
      className: "card status live",
      label: "Live model",
      note: `A trained model is answering this form. Prices are ${CONFIG.reference_year} ` +
            `market prices — the dataset is a single January ${CONFIG.reference_year} ` +
            `snapshot, so they are not today's.`,
    },
    "no-model": {
      className: "card status offline",
      label: "No model loaded — answers will be guesses",
      note: "The prediction API is running but could not load an artifact, so anything you " +
            "submit is priced by a rough formula in app.js rather than by the model. Train " +
            "one (python -m car_price_ml.train) and restart the service.",
    },
    absent: {
      className: "card status offline",
      label: "Static demo — answers will be guesses",
      note: "No prediction API is reachable from this page, so anything you submit is " +
            "priced by a rough formula in app.js rather than by the model. Run the service " +
            "(uvicorn api.main:app) for a real valuation.",
    },
  };
  const state = states[kind];
  renderCard(status, state.className, { className: "status-label", text: state.label },
             [element("p", null, state.note)]);
}

function renderUnavailable(reason) {
  const status = document.getElementById("status");
  renderCard(status, "card status broken",
             { className: "status-label", text: "This form cannot run" },
             [element("p", null,
                      `${reason}. It validates against the same vocabularies and bounds the ` +
                      `model was trained on, and will not fall back to a guess at them.`)]);
}

function renderProblems(messages) {
  const list = element("ul", "problems");
  for (const message of messages) list.appendChild(element("li", null, message));
  renderCard(document.getElementById("result"), "card result refused",
             { className: "result-label", text: "Not sent — the input is outside what the "
                                                + "model can price" },
             [list]);
}

function renderPrediction(price, asOf) {
  renderCard(document.getElementById("result"), "card result prediction",
             { className: "result-label", text: "Model prediction" },
             [element("p", "price", pln.format(price)),
              element("p", "note", `At ${asOf} market prices, which is the vintage of the `
                                   + `data this model was trained on.`)]);
}

function renderEstimate(price, note) {
  renderCard(document.getElementById("result"), "card result estimate",
             { className: "result-label", text: "Offline estimate — not a model prediction" },
             [element("p", "price", `≈ ${pln.format(price)}`),
              element("p", "note", note)]);
}

// --- The form ---------------------------------------------------------------

function fillSelect(id, values) {
  const select = document.getElementById(id);
  // A disabled, empty placeholder so nothing is pre-selected — the user must choose, and the
  // "Pick a …" validation becomes reachable.
  const placeholder = element("option", null, "Select…");
  placeholder.value = "";
  placeholder.disabled = true;
  placeholder.selected = true;
  const options = values.map((value) => {
    const option = element("option", null, value);
    option.value = value;
    return option;
  });
  select.replaceChildren(placeholder, ...options);
}

// The same bounds the validation below uses, put on the inputs themselves so the browser's
// own spinners and keyboards stop at them too.
function applyBounds() {
  const bound = (id, attributes) => {
    const input = document.getElementById(id);
    for (const [name, value] of Object.entries(attributes)) input.setAttribute(name, value);
  };
  bound("mark", { maxlength: CONFIG.mark_max_length });
  bound("model", { maxlength: CONFIG.model_max_length });
  bound("year", { min: CONFIG.year_min, max: CONFIG.year_max });
  bound("mileage", { min: 0, max: CONFIG.mileage_max });
  bound("vol_engine", { min: 0, max: CONFIG.vol_engine_max });
}

function readForm() {
  const value = (id) => document.getElementById(id).value.trim();
  // Empty -> NaN (not 0): a blank number field must fail validation, not silently mean "0".
  const num = (id) => {
    const raw = value(id);
    return raw === "" ? NaN : Number(raw);
  };
  return {
    // The dataset spells every make and model in lower case, and the API compares exactly
    // after casefolding — "Opel" used to be priced as an unknown make, 33 % high.
    mark: value("mark").toLowerCase(),
    model: value("model").toLowerCase(),
    fuel: value("fuel"),
    province: value("province"),
    year: num("year"),
    mileage: num("mileage"),
    vol_engine: num("vol_engine"),
  };
}

// Returns a list of human-readable problems; empty means the input is valid.
function validate(car) {
  const problems = [];
  const isInt = (n) => Number.isInteger(n);

  if (!car.mark) problems.push("Make is required.");
  else if (car.mark.length > CONFIG.mark_max_length) {
    problems.push(`Make is too long (max ${CONFIG.mark_max_length} characters).`);
  }
  if (!car.model) problems.push("Model is required.");
  else if (car.model.length > CONFIG.model_max_length) {
    problems.push(`Model is too long (max ${CONFIG.model_max_length} characters).`);
  }
  if (!CONFIG.fuels.includes(car.fuel)) problems.push("Pick a fuel type.");
  // Membership, not just emptiness: the API answers an unknown province with a 422, so the
  // client-side message has to be checking the same thing the server decides on.
  if (!CONFIG.provinces.includes(car.province)) problems.push("Pick a province.");
  if (!isInt(car.year) || car.year < CONFIG.year_min || car.year > CONFIG.year_max) {
    problems.push(`Year must be a whole number between ${CONFIG.year_min} and `
                  + `${CONFIG.year_max}.`);
  }
  if (!isInt(car.mileage) || car.mileage < 0 || car.mileage > CONFIG.mileage_max) {
    problems.push(`Mileage must be between 0 and `
                  + `${CONFIG.mileage_max.toLocaleString("en")} km.`);
  }
  if (!isInt(car.vol_engine) || car.vol_engine < 0 || car.vol_engine > CONFIG.vol_engine_max) {
    problems.push(`Engine capacity must be between 0 and ${CONFIG.vol_engine_max} cm³.`);
  } else if (car.vol_engine === 0 && car.fuel !== CONFIG.electric_fuel) {
    // Mirrors the cleaning rule (data.has_plausible_displacement): combustion cars with no
    // displacement were dropped from training, so the model has never seen one and the API
    // refuses the request.
    problems.push(`Engine capacity of 0 is only valid for ${CONFIG.electric_fuel}.`);
  }
  return problems;
}

// A transparent, rough fallback used only when no model is answering. It is NOT the trained
// model — just a plausible-looking heuristic, labelled as one everywhere it appears.
function heuristicEstimate(car) {
  const fuelFactor = {
    Gasoline: 1.0, Diesel: 1.15, Hybrid: 1.35, Electric: 1.5, LPG: 0.9, CNG: 0.85,
  };
  let price = 6000 + car.vol_engine * 22;
  price *= fuelFactor[car.fuel] ?? 1.0;
  const age = CONFIG.reference_year - car.year;
  price *= Math.pow(0.92, Math.max(0, age)); // ~8% per year
  price *= Math.max(0.25, 1 - car.mileage / 400000); // mileage wear
  return Math.max(2000, Math.round(price));
}

// --- The service ------------------------------------------------------------

class ValidationError extends Error {}

/** What is answering this form, asked once at load instead of discovered by submitting. */
async function probeBackend() {
  let response;
  try {
    response = await fetch(HEALTH_PATH);
  } catch {
    return "absent";
  }
  if (!response.ok) return "absent";
  const body = await response.json().catch(() => null);
  // A 200 that is not this API's health payload (a static host answering with a page, a proxy
  // interstitial) is not evidence of a service — reporting it as one would promise a model.
  if (!body || typeof body.model_loaded !== "boolean") return "absent";
  return body.model_loaded ? "model" : "no-model";
}

/**
 * Ask the API for a prediction. Resolves to {price, asOf} on success, or {unavailable} when
 * no model answered. Throws ValidationError on a 422 — the input was refused, and that is an
 * answer rather than a failure.
 */
async function predictViaApi(car) {
  let response;
  try {
    response = await fetch(PREDICT_PATH, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(car),
    });
  } catch {
    return { unavailable: "absent" }; // network error — no API here (e.g. the Pages demo)
  }

  if (response.ok) {
    const body = await response.json().catch(() => null);
    // Both fields or neither: an undated price is a number the reader would date themselves,
    // and this market moved after the snapshot the model was trained on.
    if (!body || typeof body.predicted_price_pln !== "number"
        || typeof body.valuation_as_of !== "number") {
      return { unavailable: "absent" };
    }
    return { price: body.predicted_price_pln, asOf: body.valuation_as_of };
  }
  if (response.status === 422) {
    const body = await response.json().catch(() => null);
    // FastAPI's own validation errors arrive as a list; ours (unknown make/model) as a string.
    const detail = body && Array.isArray(body.detail)
      ? body.detail.map((problem) => problem.msg).join("; ")
      : (body && typeof body.detail === "string" ? body.detail : "the API rejected the input");
    throw new ValidationError(detail);
  }
  // 503 means the service answered and has no model — a deployment fault, not an absent API.
  // Reporting it as "unreachable" once hid a container that could never load its artifact.
  if (response.status === 503) return { unavailable: "no-model" };
  return { unavailable: "absent" }; // 404, 5xx, … -> no model answered
}

/** Offer what the model can actually price, when something knows. */
async function fillVocabularyHints() {
  let vocabulary;
  try {
    const response = await fetch(VOCABULARY_PATH);
    if (!response.ok) return;
    vocabulary = await response.json();
  } catch {
    return; // no API here
  }
  for (const field of ["mark", "model"]) {
    const list = document.getElementById(`${field}-options`);
    if (!list || !Array.isArray(vocabulary[field])) continue;
    list.replaceChildren(...vocabulary[field].map((value) => {
      const option = document.createElement("option");
      option.value = value;
      return option;
    }));
  }
}

// --- Wiring -----------------------------------------------------------------

const HEURISTIC_NOTE = {
  "no-model": "The API is running but has no model loaded, so this number comes from a rough "
              + "formula in app.js. It carries no accuracy claim.",
  absent: "No prediction API answered, so this number comes from a rough formula in app.js. "
          + "It carries no accuracy claim.",
};

async function onSubmit(event) {
  event.preventDefault();
  const car = readForm();

  const problems = validate(car);
  if (problems.length) {
    renderProblems(problems);
    return;
  }

  const button = document.getElementById("submit");
  button.disabled = true;
  button.textContent = "Estimating…";
  try {
    const prediction = await predictViaApi(car);
    if (typeof prediction.price === "number") {
      renderPrediction(prediction.price, prediction.asOf);
      if (backend !== "model") {
        backend = "model";
        renderStatus(backend);
      }
    } else {
      // What actually happened outranks what the probe found at load: a service that has gone
      // away since then must change the banner, not just this one answer.
      if (backend !== prediction.unavailable) {
        backend = prediction.unavailable;
        renderStatus(backend);
      }
      renderEstimate(heuristicEstimate(car), HEURISTIC_NOTE[backend]);
    }
  } catch (error) {
    if (error instanceof ValidationError) renderProblems([error.message]);
    else renderProblems([`Unexpected error: ${error.message}`]);
  } finally {
    button.disabled = false;
    button.textContent = "Estimate price";
  }
}

async function init() {
  try {
    CONFIG = await loadConfig();
  } catch (error) {
    // No fallback constants. A form that guessed its own bounds is how the province bug
    // shipped, and the guess would look exactly like a working form.
    renderUnavailable(error.message);
    return;
  }

  fillSelect("fuel", CONFIG.fuels);
  fillSelect("province", CONFIG.provinces);
  applyBounds();
  document.getElementById("valuation").addEventListener("submit", onSubmit);

  backend = await probeBackend();
  renderStatus(backend);
  // Only now: between the button opening and the banner arriving, a fast submit would have
  // been answered by the heuristic while the banner still said "Checking…" — which is the
  // order this change exists to reverse.
  document.getElementById("submit").disabled = false;
  if (backend === "model") await fillVocabularyHints();
}

document.addEventListener("DOMContentLoaded", init);
