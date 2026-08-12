"use strict";

// These bounds mirror the FastAPI service (api/main.py + config.py). The form is static,
// so it validates client-side too — but the API remains the source of truth.
const CONFIG = {
  fuels: ["CNG", "Diesel", "Electric", "Gasoline", "Hybrid", "LPG"],
  // Spelled exactly as in config.PROVINCES — correct Polish orthography capitalises only
  // the first element of a compound name ("Kujawsko-pomorskie"). A test asserts this list
  // matches the Python vocabulary: the earlier "Kujawsko-Pomorskie" spelling matched no
  // trained category, so the model priced those two provinces as if no location was given.
  provinces: [
    "Dolnośląskie", "Kujawsko-pomorskie", "Lubelskie", "Lubuskie", "Łódzkie",
    "Małopolskie", "Mazowieckie", "Opolskie", "Podkarpackie", "Podlaskie",
    "Pomorskie", "Śląskie", "Świętokrzyskie", "Warmińsko-mazurskie",
    "Wielkopolskie", "Zachodniopomorskie",
  ],
  // Mirrors of config.py, asserted equal by a test. The anchor is the source snapshot's
  // vintage (January 2022), so the form cannot offer model years the model never saw.
  referenceYear: 2022, // config.REFERENCE_YEAR
  yearMin: 1982, // REFERENCE_YEAR - AGE_MAX (2022 - 40)
  yearMax: 2022,
  mileageMax: 1_000_000,
  volEngineMax: 8_000,
  predictPath: "/predict", // same origin: real when served by the API, 404 on Pages -> heuristic
};

const pln = new Intl.NumberFormat("pl-PL", {
  style: "currency",
  currency: "PLN",
  maximumFractionDigits: 0,
});

function fillSelect(id, values) {
  const select = document.getElementById(id);
  // A disabled, empty placeholder so nothing is pre-selected — the user must choose, and the
  // "Pick a …" validation becomes reachable.
  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = "Select…";
  placeholder.disabled = true;
  placeholder.selected = true;
  select.appendChild(placeholder);
  for (const value of values) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    select.appendChild(option);
  }
}

function readForm() {
  const value = (id) => document.getElementById(id).value.trim();
  // Empty -> NaN (not 0): a blank number field must fail validation, not silently mean "0".
  const num = (id) => {
    const v = value(id);
    return v === "" ? NaN : Number(v);
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
  const errors = [];
  const isInt = (n) => Number.isInteger(n);

  if (!car.mark) errors.push("Make is required.");
  else if (car.mark.length > 40) errors.push("Make is too long (max 40 characters).");
  if (!car.model) errors.push("Model is required.");
  else if (car.model.length > 60) errors.push("Model is too long (max 60 characters).");
  if (!CONFIG.fuels.includes(car.fuel)) errors.push("Pick a fuel type.");
  // Membership, not just emptiness: the API answers an unknown province with a 422, so the
  // client-side message has to be checking the same thing the server decides on.
  if (!CONFIG.provinces.includes(car.province)) errors.push("Pick a province.");
  if (!isInt(car.year) || car.year < CONFIG.yearMin || car.year > CONFIG.yearMax) {
    errors.push(`Year must be a whole number between ${CONFIG.yearMin} and ${CONFIG.yearMax}.`);
  }
  if (!isInt(car.mileage) || car.mileage < 0 || car.mileage > CONFIG.mileageMax) {
    errors.push(`Mileage must be between 0 and ${CONFIG.mileageMax.toLocaleString("en")} km.`);
  }
  if (!isInt(car.vol_engine) || car.vol_engine < 0 || car.vol_engine > CONFIG.volEngineMax) {
    errors.push(`Engine capacity must be between 0 and ${CONFIG.volEngineMax} cm³.`);
  } else if (car.vol_engine === 0 && car.fuel !== "Electric") {
    // Mirrors the cleaning rule: combustion cars with no displacement were dropped from
    // training, so the model has never seen one and the API refuses the request.
    errors.push("Engine capacity of 0 is only valid for an electric car.");
  }
  return errors;
}

// A transparent, rough fallback used only when the ML API is unreachable (e.g. the static
// GitHub Pages demo). It is NOT the trained model — just a plausible-looking heuristic.
function heuristicEstimate(car) {
  const fuelFactor = { Gasoline: 1.0, Diesel: 1.15, Hybrid: 1.35, Electric: 1.5, LPG: 0.9, CNG: 0.85 };
  let price = 6000 + car.vol_engine * 22;
  price *= fuelFactor[car.fuel] ?? 1.0;
  const age = CONFIG.referenceYear - car.year;
  price *= Math.pow(0.92, Math.max(0, age)); // ~8% per year
  price *= Math.max(0.25, 1 - car.mileage / 400000); // mileage wear
  return Math.max(2000, Math.round(price));
}

class ValidationError extends Error {}

// Ask the API for a prediction. Resolves to {price} on success. Throws ValidationError on a
// 422 (server rejected the input); returns null when the API is unreachable or otherwise
// unavailable, signalling the caller to fall back to the heuristic.
async function predictViaApi(car) {
  let response;
  try {
    response = await fetch(CONFIG.predictPath, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(car),
    });
  } catch {
    return null; // network error — no API here (e.g. the Pages demo)
  }

  if (response.ok) {
    const data = await response.json().catch(() => null);
    if (!data || typeof data.predicted_price_pln !== "number") return null;  // malformed -> fall back
    return { price: data.predicted_price_pln, asOf: data.valuation_as_of };
  }
  if (response.status === 422) {
    const body = await response.json().catch(() => null);
    // FastAPI's own validation errors arrive as a list; ours (unknown make/model) as a string.
    const detail = body && Array.isArray(body.detail)
      ? body.detail.map((d) => d.msg).join("; ")
      : (body && typeof body.detail === "string" ? body.detail : "the API rejected the input");
    throw new ValidationError(detail);
  }
  // 503 means the service answered and has no model — a deployment fault, not an absent API.
  // Reporting it as "unreachable" once hid a container that could never load its artifact.
  if (response.status === 503) return { unavailable: "reachable-but-no-model" };
  return null; // 404, 5xx, ... -> fall back
}

function renderErrors(messages) {
  const result = document.getElementById("result");
  result.hidden = false;
  const items = messages.map((m) => `<li>${escapeHtml(m)}</li>`).join("");
  result.innerHTML = `<div class="errors"><strong>Please fix:</strong><ul>${items}</ul></div>`;
}

function renderPrice(price, source, reason, asOf) {
  const result = document.getElementById("result");
  result.hidden = false;
  const badge = source === "model"
    ? '<span class="badge model">model prediction</span>'
    : '<span class="badge offline">offline estimate</span>';
  // The training data is a single January 2022 snapshot, so the figure is a 2022 price.
  // Showing it undated would let the reader assume a currency the model cannot support.
  const dated = asOf ? `<p class="reason">Valued at ${asOf} market prices.</p>` : "";
  const note = reason ? `<p class="reason">${escapeHtml(reason)}</p>` : "";
  result.innerHTML = `${badge}<p class="price">${pln.format(price)}</p>${dated}${note}`;
}

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
  );
}

async function onSubmit(event) {
  event.preventDefault();
  const car = readForm();

  const problems = validate(car);
  if (problems.length) {
    renderErrors(problems);
    return;
  }

  const button = document.getElementById("submit");
  button.disabled = true;
  button.textContent = "Estimating…";
  try {
    const prediction = await predictViaApi(car);
    if (prediction && typeof prediction.price === "number") {
      renderPrice(prediction.price, "model", null, prediction.asOf);
    } else {
      const noModel = prediction && prediction.unavailable === "reachable-but-no-model";
      renderPrice(
        heuristicEstimate(car),
        "offline",
        noModel
          ? "The API is running but has no model loaded, so this is a rough offline heuristic — not a prediction. Train one (python -m car_price_ml.train) and restart the service."
          : "The ML API was not reachable, so this is a rough offline heuristic — not the trained model. Run the API (uvicorn api.main:app) for a real prediction."
      );
    }
  } catch (error) {
    if (error instanceof ValidationError) {
      renderErrors([error.message]);
    } else {
      renderErrors(["Unexpected error: " + error.message]);
    }
  } finally {
    button.disabled = false;
    button.textContent = "Estimate price";
  }
}

// Offer what the model can actually price. Make and model are free text because the domain
// lives in the trained artifact, not in this file — so the list is fetched when the API is
// there, and the inputs simply stay unsuggested when it is not (the Pages demo). The server
// rejects anything outside the list either way; this only makes the boundary visible before
// the user submits.
async function fillVocabularyHints() {
  let vocabulary;
  try {
    const response = await fetch("/vocabulary");
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

function init() {
  fillSelect("fuel", CONFIG.fuels);
  fillSelect("province", CONFIG.provinces);
  document.getElementById("valuation").addEventListener("submit", onSubmit);
  fillVocabularyHints();
}

document.addEventListener("DOMContentLoaded", init);
