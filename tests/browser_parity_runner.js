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
    return { case: entry.case, price: model.predict(entry.car) };
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
