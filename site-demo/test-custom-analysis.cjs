const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const path = require("node:path");

function stubElement() {
  return {
    value: "0.2",
    disabled: false,
    files: [],
    classList: { add() {}, remove() {}, toggle() {} },
    addEventListener() {},
    querySelectorAll() { return []; },
    replaceChildren() {},
    append() {},
    set textContent(_) {},
    get textContent() { return ""; }
  };
}

const documentStub = {
  querySelector() { return stubElement(); },
  querySelectorAll() { return []; },
  createElement() { return stubElement(); }
};
const windowStub = {
  matchMedia() { return { matches: true }; },
  addEventListener() {},
  setTimeout,
  devicePixelRatio: 1
};
const source = fs.readFileSync(path.join(__dirname, "dist", "custom-analysis.js"), "utf8");
vm.runInNewContext(source, { window: windowStub, document: documentStub, console, setTimeout, Map, Set, Math, Number });
const engine = windowStub.CustomRegressionEngine;
assert.ok(engine, "Regression engine should be exposed for testing");

const csv = fs.readFileSync(path.join(__dirname, "..", "StudentsPerformance_train.csv"), "utf8");
const parsed = engine.parseCSV(csv);
const headers = engine.uniqueHeaders(parsed[0]);
const rows = parsed.slice(1).map((row) => headers.map((_, index) => row[index] ?? ""));
assert.equal(rows.length, 16000);
const profiles = headers.map((name, index) => engine.profileColumn(rows, index, name));
const target = profiles.find((profile) => profile.name === "productivity_score");
const selected = ["gender", "study_hours_per_day", "sleep_hours", "phone_usage_hours", "attendance_percentage", "stress_level", "focus_score"]
  .map((name) => profiles.find((profile) => profile.name === name));
const shuffled = engine.seededShuffle(rows);
const testRows = shuffled.slice(0, 3200);
const trainRows = shuffled.slice(3200);
const { encoders, encodedNames } = engine.fitEncoders(trainRows, selected);
const xTrain = trainRows.map((row) => engine.transformRow(row, encoders));
const yTrain = trainRows.map((row) => engine.numericValue(row[target.index]));
const coefficients = engine.fitLinearRegression(xTrain, yTrain);
const actual = testRows.map((row) => engine.numericValue(row[target.index]));
const predicted = testRows.map((row) => engine.transformRow(row, encoders)
  .reduce((sum, value, index) => sum + value * coefficients[index], 0));
const trainMean = yTrain.reduce((sum, value) => sum + value, 0) / yTrain.length;
const result = engine.metrics(actual, predicted, trainMean);
assert.equal(encodedNames.length, 8);
assert.ok(result.r2 > 0.9999, `Expected R2 > 0.9999, received ${result.r2}`);
assert.ok(result.rmse < 0.01, `Expected RMSE < 0.01, received ${result.rmse}`);
console.log(JSON.stringify({ rows: rows.length, encodedFeatures: encodedNames.length, rmse: result.rmse, r2: result.r2 }, null, 2));
