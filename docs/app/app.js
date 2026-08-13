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

// The served model itself, exported by `car_price_ml.site.browser_model` and interpreted by
// predict.js. This replaced an "offline heuristic" — a formula in this file, labelled as a
// guess, which is what the published demo used to answer with. There is no guess left here:
// the page runs the same 1 200 trees the API runs, and a fixture holds all three
// implementations to the same prices.
const MODEL_URL = "model.json";

// Same-origin absolute paths: real when the API serves this directory at its root, absent on
// the static GitHub Pages copy — which the banner reports before the first submit rather than
// after it.
const PREDICT_PATH = "/predict";
const HEALTH_PATH = "/health";

const pln = new Intl.NumberFormat("pl-PL", {
  style: "currency",
  currency: "PLN",
  maximumFractionDigits: 0,
});

let CONFIG = null;
let MODEL = null; // the exported model, once predict.js has it loaded
// Which of the two paths answers a submission: "api" (the service, when it has a model) or
// "browser" (the exported model, running here). Both are the same 1 200 trees; the banner says
// which one, because "where was this computed" is a question the reader is entitled to ask.
let backend = "browser";
// Set when the form has discovered that nothing can answer it any more. Latching rather than
// re-checking: a form that reopened its own button after refusing would invite the reader to
// press it again for the same refusal.
let halted = false;

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

// --- The model ---------------------------------------------------------------

// 2.3 MB over the wire and around 8.5 MB parsed, so it is fetched once at load and the submit
// button waits for it. Failing to load is not fatal on its own: if the API is there with a
// model, it can answer instead. What is fatal is *nothing* being able to answer, which is the
// state `init` refuses in rather than filling with a formula.
async function loadModel() {
  const response = await fetch(MODEL_URL);
  if (!response.ok) throw new Error(`model.json answered ${response.status}`);
  return createModel(await response.json());
}

// --- Rendering helpers ------------------------------------------------------

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

/** Announce this card: it is an answer to something the reader asked for, not a preview. */
function announce(target) {
  target.setAttribute("aria-live", "polite");
  return target;
}

// Built as nodes rather than assembled as HTML: `detail` comes from the service, and a
// valuation form has no business interpreting anything it is told as markup.
function renderCard(target, className, label, children) {
  target.hidden = false;
  target.className = className;
  target.replaceChildren(element("span", label.className, label.text), ...children);
}

function trainedDescription() {
  if (!MODEL) return "";
  return `${MODEL.servedModel}, ${MODEL.treeCount.toLocaleString("en")} trees trained on `
       + `${MODEL.trainedOn.toLocaleString("en")} adverts (${MODEL.trainedAt}).`;
}

function renderStatus(kind) {
  const status = document.getElementById("status");
  const dated = `Prices are ${CONFIG.reference_year} market prices — the data is a single `
              + `January ${CONFIG.reference_year} snapshot, so they are not today's.`;
  const states = {
    api: {
      className: "card status live",
      label: "Answered by the prediction API",
      note: `The service is running and has a model loaded, so it answers. ${dated}`,
    },
    browser: {
      className: "card status live",
      label: "The model runs in this page",
      note: `No prediction API here, so the model itself was downloaded and runs on your `
          + `machine — not an approximation of it: ${trainedDescription()} A fixture holds `
          + `this runtime, the service and the Python pipeline to the same prices. ${dated}`,
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
  renderCard(announce(document.getElementById("result")), "card result refused",
             { className: "result-label", text: "Not sent — the input is outside what the "
                                                + "model can price" },
             [list]);
}

/**
 * The measured spread for a price, said in the terms the measurement supports.
 * Null when there is no local model to read the bands off — the API can price a car without
 * this page holding the model, and a missing spread has to be an absent sentence rather than
 * a crash blamed on the reader's input.
 */
function bandNote(price) {
  if (!MODEL) return null;
  const band = MODEL.errorBand(price);
  const half = `half of cars valued near this land within ${pln.format(band.p50_abs_error)}`;
  const most = `nine in ten within ${pln.format(band.p90_abs_error)}`;
  // The band's own width, said out loud: the top one spans 150 796 to 898 640 PLN, and
  // "cars valued near this" would otherwise cover a sixfold range without admitting it.
  const range = `${pln.format(band.from_pln)}–${pln.format(band.to_pln)}`;
  const measured = band.measured
    ? ""
    : " — this valuation is outside the range the model's own errors were measured over, so "
      + "the figures are the nearest band's rather than this one's";
  return `Out-of-fold error for valuations of ${range}: ${half}, ${most}${measured}.`;
}

function renderLive(price) {
  // The live figure is not announced. The region was written for one answer per submit, and it
  // now re-renders on every pause in typing — a screen reader would read the whole card back
  // each time and never drain the queue. The submitted answer and every refusal still are.
  document.getElementById("result").setAttribute("aria-live", "off");
  renderCard(document.getElementById("result"), "card result prediction",
             { className: "result-label", text: "Valued here, as you type" },
             [element("p", "price", pln.format(price)),
              element("p", "note", bandNote(price)),
              element("p", "note", `At ${CONFIG.reference_year} market prices. `
                                   + trainedDescription())]);
}

/** What is still missing, said without the weight of a refusal — nothing was submitted yet. */
function renderIncomplete(problems) {
  const list = element("ul", "problems");
  for (const problem of problems) list.appendChild(element("li", null, problem));
  renderCard(document.getElementById("result"), "card result pending",
             { className: "result-label", text: "Not enough to value this car yet" }, [list]);
}

function renderPrediction(price, asOf, where) {
  const computed = where === "api"
    ? "Computed by the prediction API, which agrees with the copy running in this page to "
      + "within a grosz — a fixture holds them to it."
    : `Computed here, in your browser, by ${trainedDescription()}`;
  const spread = bandNote(price);
  renderCard(announce(document.getElementById("result")), "card result prediction",
             { className: "result-label", text: "Model prediction" },
             [element("p", "price", pln.format(price)),
              ...(spread ? [element("p", "note", spread)] : []),
              element("p", "note", `At ${asOf} market prices, which is the vintage of the `
                                   + `data this model was trained on. ${computed}`)]);
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
  // The two sliders are what-if controls over the same values, not extra fields: the model is
  // in this page, so dragging one costs a millisecond and no request. Bounds come from the
  // same config, and the mileage slider stops at a range real adverts occupy rather than at
  // the validation ceiling — a slider whose useful travel is the first 4 % of its length is a
  // control nobody can aim.
  bound("year-slider", { min: CONFIG.year_min, max: CONFIG.year_max, step: 1 });
  bound("mileage-slider", { min: 0, max: SLIDER_MILEAGE_MAX, step: 5000 });
}

// Beyond this the mileage slider stops; the number field still accepts anything the model was
// trained on. 400 000 km covers all but a fraction of a percent of the adverts.
const SLIDER_MILEAGE_MAX = 400_000;

/** Keep a number field and its slider showing one value, whichever the reader moved. */
function coupleSlider(fieldId) {
  const field = document.getElementById(fieldId);
  const slider = document.getElementById(`${fieldId}-slider`);
  // A range input with no value sits at its own midpoint, so an empty year field would ship
  // beside a slider pointing at 2002 — a year nobody entered, shown as if someone had. It
  // stays disabled until the field says what it should point at.
  const follow = () => {
    const value = Number(field.value);
    slider.disabled = field.value.trim() === "" || !Number.isFinite(value);
    if (!slider.disabled) slider.value = String(value);
  };
  follow();
  slider.addEventListener("input", () => {
    field.value = slider.value;
    onFormInput();
  });
  // Out-of-range values are left to validation rather than snapped: silently rewriting what
  // someone typed is how a form tells you it priced something other than what you asked.
  field.addEventListener("input", follow);
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

  // Make and model are checked against the model's own vocabulary when it is loaded here.
  // Before this file carried the model, the page had no way to know them and the caveat was
  // real: on GitHub Pages an unknown make still got a number. Now the page refuses it exactly
  // where the API answers 422, from the same list — read off the fitted encoder at export.
  const known = (field) => (MODEL ? MODEL.vocabulary[field] : null);
  for (const [field, label, limit] of [
    ["mark", "Make", CONFIG.mark_max_length], ["model", "Model", CONFIG.model_max_length],
  ]) {
    const value = car[field];
    const vocabulary = known(field);
    if (!value) problems.push(`${label} is required.`);
    else if (value.length > limit) {
      problems.push(`${label} is too long (max ${limit} characters).`);
    } else if (vocabulary && !vocabulary.includes(value)) {
      problems.push(`${label} "${value}" is not one of the ${vocabulary.length} the model was `
                    + `trained on — it cannot be priced, only guessed at.`);
    }
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

// --- Live valuation ---------------------------------------------------------

// Long enough that a burst of keystrokes costs one valuation, short enough to feel like the
// page is answering rather than catching up. A prediction is about a millisecond — the delay
// is for the reader's eyes, not for the model.
const LIVE_DELAY_MS = 120;
let livePending = null;

function onFormInput() {
  if (!MODEL) return; // the API path answers on submit; there is nothing to preview with
  clearTimeout(livePending);
  livePending = setTimeout(refreshLive, LIVE_DELAY_MS);
}

function refreshLive() {
  const car = readForm();
  const problems = validate(car);
  if (problems.length) {
    renderIncomplete(problems);
    document.getElementById("whatif").hidden = true;
    return;
  }
  const input = toModelInput(car);
  renderLive(MODEL.predict(input));
  drawWhatIf(input);
}

/** The same car at every age the data supports, priced now rather than looked up. */
function drawWhatIf(input) {
  // Stops where the report's depreciation curve stops. The model answers for any age, but past
  // the last bucket the site is willing to publish it is extrapolating: at 33 years and beyond
  // it returns one flat number, and between 29 and 33 it has the car *gaining* value. Drawn in
  // the same visual language as a measured chart, that would read as a finding.
  const oldest = MODEL.ageSupport.max_age;
  // Encoded once and swept along the age column: 40 re-encodings per keystroke is work the
  // reader's phone does not need to do for a value that never changes.
  const row = MODEL.encode(input);
  const ageColumn = MODEL.columnIndex("age");
  const points = [];
  for (let age = 0; age <= oldest; age += 1) {
    row[ageColumn] = age;
    points.push({ x: age, y: MODEL.predictRow(row) });
  }
  const panel = document.getElementById("whatif");
  panel.hidden = false;
  document.getElementById("whatif-limit").textContent =
    `Drawn to ${oldest} years, where the data stops supporting it: older buckets hold under `
    + `${MODEL.ageSupport.min_bucket_n} adverts each, so the report's own curve drops them too.`;
  drawCurve(document.getElementById("whatif-chart"), points, {
    highlightX: input.age,
    xLabel: "years old",
    // Thousands while the numbers are large enough for it to mean anything; below that the
    // rounding would print three axis labels reading "0k".
    format: (value) => (Math.max(...points.map((point) => point.y)) >= 20000
      ? `${Math.round(value / 1000)}k`
      : String(Math.round(value))),
  });
}

// --- The service ------------------------------------------------------------

class ValidationError extends Error {}

// How long the probe below may take before the form stops waiting for it. Undefined where
// AbortSignal.timeout is unavailable, which leaves the older behaviour rather than aborting
// the probe outright and reporting a live service as absent.
const PROBE_TIMEOUT_MS = 3000;

function probeTimeout() {
  return typeof AbortSignal?.timeout === "function"
    ? AbortSignal.timeout(PROBE_TIMEOUT_MS)
    : undefined;
}

/** What is answering this form, asked once at load instead of discovered by submitting. */
async function probeBackend() {
  let response;
  try {
    // Bounded: the submit button waits for this answer, and a host that accepts the
    // connection without ever replying would otherwise hold the form shut for good — when it
    // could have said "no API answered" and gone on validating.
    response = await fetch(HEALTH_PATH, { signal: probeTimeout() });
  } catch {
    return "absent";
  }
  if (!response.ok) return "absent";
  const body = await response.json().catch(() => null);
  // A 200 that is not this API's health payload (a static host answering with a page, a proxy
  // interstitial) is not evidence of a service — reporting it as one would promise a model.
  if (!body || typeof body.model_loaded !== "boolean") return "browser";
  // A service that is up but could not load its artifact answers 503, and the page can do
  // better than that: it has the model. So only a *serving* API takes precedence.
  return body.model_loaded ? "api" : "browser";
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
    return { unavailable: true }; // network error — no API here (e.g. the Pages demo)
  }

  if (response.ok) {
    const body = await response.json().catch(() => null);
    // Both fields or neither: an undated price is a number the reader would date themselves,
    // and this market moved after the snapshot the model was trained on.
    if (!body || typeof body.predicted_price_pln !== "number"
        || typeof body.valuation_as_of !== "number") {
      return { unavailable: true };
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
  // Everything else — 503 from a service with no artifact, a 404, a 5xx — means the API did
  // not answer this. The page can, with the same model, so it is not an error state.
  return { unavailable: true };
}

/** Offer what the model can price, read off the model itself. */
function fillVocabularyHints() {
  if (!MODEL) return;
  for (const field of ["mark", "model"]) {
    const list = document.getElementById(`${field}-options`);
    if (!list) continue;
    list.replaceChildren(...MODEL.vocabulary[field].map((value) => {
      const option = document.createElement("option");
      option.value = value;
      return option;
    }));
  }
}

// --- Wiring -----------------------------------------------------------------

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
  button.textContent = "Valuing…";
  try {
    // The service first when it has a model — it is the artifact of record, and its answer
    // carries the vintage it was fit on. Otherwise the same model, here.
    const served = backend === "api" ? await predictViaApi(car) : { unavailable: true };
    if (typeof served.price === "number") {
      renderPrediction(served.price, served.asOf, "api");
    } else {
      // The API answered at load and has stopped. Without a local model there is nothing left
      // to answer with — and saying "the model runs in this page" while it does not, then
      // blaming the reader's input for the TypeError that follows, would be two false claims
      // in place of one honest refusal.
      if (!MODEL) {
        halted = true; // the `finally` below leaves the button shut rather than reopening it
        renderUnavailable("the prediction API stopped answering and the model could not be "
                          + "loaded from this page");
        document.getElementById("result").hidden = true;
        return;
      }
      // What actually happened outranks what the probe found at load: a service that has gone
      // away since then must change the banner, not just this one answer.
      if (backend !== "browser") {
        backend = "browser";
        renderStatus(backend);
      }
      renderPrediction(MODEL.predict(toModelInput(car)), CONFIG.reference_year, "browser");
    }
  } catch (error) {
    if (error instanceof ValidationError) renderProblems([error.message]);
    else if (error instanceof UnknownValue) {
      // The runtime refusing something validation let through: a disagreement between the two
      // lists, which must surface as itself rather than as an "unexpected error".
      renderProblems([`The model cannot price this ${error.field}: ${error.value}.`]);
    } else renderProblems([`Unexpected error: ${error.message}`]);
  } finally {
    button.disabled = halted;
    button.textContent = "Value this car";
  }
}

/** The request shape uses `year`; the model was trained on `age` from a fixed anchor. */
function toModelInput(car) {
  const { year, ...rest } = car;
  return { ...rest, age: CONFIG.reference_year - year };
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

  // Together, not one after the other: a reader whose API is live would otherwise wait on a
  // 2.3 MB download they are not going to use before the button opens.
  let modelFailure = null;
  const [loaded, probed] = await Promise.all([
    loadModel().catch((error) => { modelFailure = error.message; return null; }),
    probeBackend(),
  ]);
  MODEL = loaded;
  backend = probed;
  // Neither path can answer: refuse, rather than reaching for something that can produce a
  // number anyway. This page had such a something until this change, and it was the only part
  // of the project that answered a question it could not answer.
  if (!MODEL && backend !== "api") {
    renderUnavailable(`the model could not be loaded (${modelFailure})`);
    return;
  }

  fillVocabularyHints();
  renderStatus(backend);
  // Only now: between the button opening and the banner arriving, a fast submit would have
  // been answered while the banner still said "Checking…" — which is the order this reverses.
  document.getElementById("submit").disabled = false;

  // Live valuation, only where there is a model in the page to do it with. Against the API
  // alone it would be a request per keystroke, so the form keeps its submit-and-wait shape
  // there instead of quietly becoming chatty.
  if (MODEL) {
    coupleSlider("year");
    coupleSlider("mileage");
    document.getElementById("valuation").addEventListener("input", onFormInput);
    refreshLive();
  }
}

document.addEventListener("DOMContentLoaded", init);
