import assert from "node:assert/strict";
import { test } from "node:test";
import {
  DATA_QUALITY,
  NUTRIENTS,
  filterFoods,
  formatNutrient,
  isValidNutritionCoverage,
  isValidNutritionResponse,
  nutritionRequestKey,
  nutritionView,
  nutritionWarnings,
  parseServingGrams,
  predictionOptions,
  qualityWarnings,
  verificationText,
} from "./nutrition.js";

// Shapes only; these numbers are test fixtures, not nutrition data.
const VALUES = { calories_kcal: 1, protein_g: 2, carbohydrates_g: 3, fat_g: 4 };
const AVAILABLE = {
  food_class: "idli", available: true, match_type: "DIRECT_MATCH", serving_size_g: 100,
  nutrition_per_100g: VALUES, estimated_nutrition: VALUES, notice: "n",
  source: { dataset: "USDA FoodData Central", food_description: "Idli", record_id: "1", record_label: "FDC ID" },
};
const UNAVAILABLE = {
  food_class: "dhokla", available: false, match_type: "UNAVAILABLE", serving_size_g: 100,
  nutrition_per_100g: null, estimated_nutrition: null, notice: "n", source: null,
};

test("parseServingGrams accepts positive amounts up to 2000 g", () => {
  assert.deepEqual(parseServingGrams("150"), { grams: 150 });
  assert.deepEqual(parseServingGrams(" 37.5 "), { grams: 37.5 });
  assert.deepEqual(parseServingGrams("2000"), { grams: 2000 });
  assert.deepEqual(parseServingGrams(".5"), { grams: 0.5 });
  assert.deepEqual(parseServingGrams(100), { grams: 100 });
});

test("parseServingGrams rejects missing, zero, negative, too large and non-numeric input", () => {
  for (const input of ["", "   ", null, undefined, "0", "0.0", "-5", "2000.1", "abc", "1e3", "NaN", "Infinity", "12g", "1,5"]) {
    assert.ok(parseServingGrams(input).error, `expected an error for ${JSON.stringify(input)}`);
  }
});

test("nutrition responses are validated, and unavailable foods carry no values", () => {
  assert.equal(isValidNutritionResponse(AVAILABLE), true);
  assert.equal(isValidNutritionResponse(UNAVAILABLE), true);
  assert.equal(isValidNutritionResponse({ ...UNAVAILABLE, nutrition_per_100g: { ...VALUES, calories_kcal: 0 } }), false);
  assert.equal(isValidNutritionResponse({ ...AVAILABLE, estimated_nutrition: { ...VALUES, fat_g: -1 } }), false);
  assert.equal(isValidNutritionResponse({ ...AVAILABLE, estimated_nutrition: { ...VALUES, fat_g: null } }), false);
  assert.equal(isValidNutritionResponse({ ...AVAILABLE, match_type: "UNAVAILABLE" }), false);
  assert.equal(isValidNutritionResponse({ ...AVAILABLE, source: null }), false);
  assert.equal(isValidNutritionResponse({ ...AVAILABLE, source: { ...AVAILABLE.source, record_id: undefined } }), false);
});

test("coverage must list every class with a known match type", () => {
  const coverage = { total_classes: 2, foods: [{ food_class: "idli", match_type: "DIRECT_MATCH" }, { food_class: "dhokla", match_type: "UNAVAILABLE" }] };
  assert.equal(isValidNutritionCoverage(coverage), true);
  assert.equal(isValidNutritionCoverage({ ...coverage, total_classes: 3 }), false);
  assert.equal(isValidNutritionCoverage({ ...coverage, foods: [{ food_class: "idli", match_type: "GUESS" }, coverage.foods[1]] }), false);
});

test("nutritionView never shows a response for a different food or serving", () => {
  const key = nutritionRequestKey("idli", { grams: 100 });
  assert.equal(nutritionView({ servingError: "bad", requestKey: null, response: null }), "invalid");
  assert.equal(nutritionView({ requestKey: key, response: null }), "loading");
  assert.equal(nutritionView({ requestKey: key, response: { key, status: "loading" } }), "loading");
  assert.equal(nutritionView({ requestKey: key, response: { key, status: "ready", data: AVAILABLE } }), "ready");
  assert.equal(nutritionView({ requestKey: key, response: { key, status: "ready", data: UNAVAILABLE } }), "unavailable");
  assert.equal(nutritionView({ requestKey: key, response: { key, status: "error", error: "x" } }), "error");

  // Changing the selected food or the serving size makes the old result stale.
  const oldResponse = { key, status: "ready", data: AVAILABLE };
  assert.equal(nutritionView({ requestKey: nutritionRequestKey("samosa", { grams: 100 }), response: oldResponse }), "loading");
  assert.equal(nutritionView({ requestKey: nutritionRequestKey("idli", { grams: 250 }), response: oldResponse }), "loading");
  assert.equal(nutritionRequestKey("idli", { error: "bad" }), null);
});

test("low-confidence predictions get a warning that values depend on the food being right", () => {
  const warnings = nutritionWarnings({ lowConfidence: true, food: "chicken_curry", predictedFood: "chicken_curry" });
  assert.equal(warnings.length, 1);
  assert.match(warnings[0], /unsure/);
  assert.match(warnings[0], /chicken curry/);
  assert.deepEqual(nutritionWarnings({ lowConfidence: false, food: "pizza", predictedFood: "pizza" }), []);
  const switched = nutritionWarnings({ lowConfidence: false, food: "samosa", predictedFood: "pizza" });
  assert.match(switched[0], /instead of the top match/);
  for (const message of [...warnings, ...switched]) assert.doesNotMatch(message, /\d/);
});

test("low-quality and flagged references get a caveat; unavailable foods get none", () => {
  assert.deepEqual(qualityWarnings({ ...AVAILABLE, data_quality: "HIGH", flagged_for_review: false }), []);
  assert.deepEqual(qualityWarnings({ ...AVAILABLE, data_quality: "MEDIUM", flagged_for_review: false }), []);
  const low = qualityWarnings({ ...AVAILABLE, data_quality: "LOW", flagged_for_review: false });
  assert.equal(low.length, 1);
  assert.match(low[0], /Low data quality/);
  const flagged = qualityWarnings({ ...AVAILABLE, data_quality: "MEDIUM", flagged_for_review: true });
  assert.match(flagged[0], /Flagged for review/);
  assert.deepEqual(qualityWarnings({ ...UNAVAILABLE, data_quality: "NONE", flagged_for_review: false }), []);
  for (const message of [...low, ...flagged]) assert.doesNotMatch(message, /\d/);
  assert.equal(DATA_QUALITY.NONE, undefined);
  assert.equal(DATA_QUALITY.LOW.style, "warning");
});

test("verification status is shown in plain language, or verbatim when unknown", () => {
  assert.match(verificationText("SOURCE_RECORD_READ_VIA_FDC_MIRROR_NOT_INSPECTED_ON_USDA_SITE"), /mirror, not on the USDA site/);
  assert.match(verificationText("SOURCE_RECORD_READ_VIA_FDC_MIRROR; SOME_VALUES_ARITHMETICALLY SCALED_FROM_DECLARED_SERVING"), /declared serving/);
  assert.match(verificationText("REQUIRES_RESEARCH"), /Further research/);
  assert.equal(verificationText("SOMETHING_NEW"), "SOMETHING_NEW");
  assert.equal(verificationText(undefined), "");
});

test("prediction options and food search", () => {
  assert.deepEqual(predictionOptions([{ food: "kadai_paneer", confidence: 41.234 }]), [{ value: "kadai_paneer", label: "kadai paneer (41.2%)" }]);
  const foods = ["Baked Potato", "apple_pie", "kadai_paneer", "pizza"];
  assert.deepEqual(filterFoods(foods, "paneer"), ["kadai_paneer"]);
  assert.deepEqual(filterFoods(foods, "APPLE pie"), ["apple_pie"]);
  assert.deepEqual(filterFoods(foods, "apple_pie"), ["apple_pie"]);
  assert.deepEqual(filterFoods(foods, ""), foods);
  assert.deepEqual(filterFoods(foods, "pasta"), []);
});

test("formatNutrient uses whole kcal and one-decimal grams", () => {
  const [calories, protein] = NUTRIENTS;
  assert.equal(formatNutrient(576, calories), "576 kcal");
  assert.equal(formatNutrient(35, protein), "35.0 g");
});
