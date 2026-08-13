"use strict";

// Runs the fixture through the same predict.js the browser loads, and prints the result as
// JSON for pytest to read. Deliberately dumb: every judgement about what counts as agreement
// stays on the Python side, so this file cannot quietly relax the test it feeds.
//
//   node tests/browser_parity_runner.js
//
// Requires nothing but Node itself — no package.json, no install step. That is the whole
// reason the runtime is a plain script with a three-line CommonJS shim rather than a module
// with a build.

const fs = require("fs");
const path = require("path");

const root = path.join(__dirname, "..");
const { createModel, UnknownValue } = require(path.join(root, "docs", "app", "predict.js"));

const payload = JSON.parse(fs.readFileSync(path.join(root, "docs", "app", "model.json"), "utf8"));
const fixture = JSON.parse(
  fs.readFileSync(path.join(root, "tests", "fixtures", "browser_parity.json"), "utf8")
);

const model = createModel(payload);
const results = fixture.cases.map((entry) => {
  try {
    const price = model.predict(entry.car);
    const band = model.errorBand(price);
    // Reported rather than judged here: which band a price falls in is compared against the
    // Python lookup on the other side, so this file states what it found and nothing more.
    return { case: entry.case, price, band: { from: band.from_pln, to: band.to_pln,
                                              p50: band.p50_abs_error, measured: band.measured } };
  } catch (error) {
    if (error instanceof UnknownValue) {
      return { case: entry.case, refused: error.field };
    }
    return { case: entry.case, error: `${error.name}: ${error.message}` };
  }
});

process.stdout.write(JSON.stringify({
  servedModel: model.servedModel,
  trainedAt: model.trainedAt,
  treeCount: model.treeCount,
  results,
}));
