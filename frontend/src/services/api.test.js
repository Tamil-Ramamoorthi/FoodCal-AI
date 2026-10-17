import assert from "node:assert/strict";
import { afterEach, test } from "node:test";
import {
  API_BASE_URL, ApiError, clearHistory, deleteHistoryRecord, getNutrition, getNutritionCoverage,
} from "./api.js";

// fetch is replaced by a stub that records the URL and returns `reply`.
// The bodies are shape fixtures, not nutrition data.
const realFetch = globalThis.fetch;
let requested = [];
let methods = [];

function reply(status, body) {
  globalThis.fetch = async (url, options) => {
    requested.push(url);
    methods.push(options?.method || "GET");
    return { ok: status >= 200 && status < 300, status, json: async () => body };
  };
}

afterEach(() => {
  globalThis.fetch = realFetch;
  requested = [];
  methods = [];
});

const VALUES = { calories_kcal: 1, protein_g: 1, carbohydrates_g: 1, fat_g: 1 };
const nutrition = (food) => ({
  food_class: food, available: true, match_type: "DIRECT_MATCH", serving_size_g: 150, notice: "n",
  nutrition_per_100g: VALUES, estimated_nutrition: VALUES,
  source: { dataset: "d", food_description: "f", record_id: "1", record_label: "FDC ID" },
});

test("nutrition requests go to the backend API with the class name encoded", async () => {
  assert.equal(API_BASE_URL, "http://127.0.0.1:8000");
  reply(200, nutrition("chapati"));
  await getNutrition("chapati", 150);
  reply(200, nutrition("Baked Potato"));
  await getNutrition("Baked Potato", 100);
  reply(200, nutrition("masala_dosa"));
  await getNutrition("masala_dosa", 37.5);
  assert.deepEqual(requested, [
    "http://127.0.0.1:8000/api/nutrition/chapati?grams=150",
    "http://127.0.0.1:8000/api/nutrition/Baked%20Potato?grams=100",
    "http://127.0.0.1:8000/api/nutrition/masala_dosa?grams=37.5",
  ]);
});

test("coverage is requested from /api/nutrition", async () => {
  reply(200, { total_classes: 1, foods: [{ food_class: "idli", match_type: "DIRECT_MATCH" }] });
  await getNutritionCoverage();
  assert.deepEqual(requested, ["http://127.0.0.1:8000/api/nutrition"]);
});

test("a backend without the nutrition routes is reported as such, not as a missing food", async () => {
  for (const call of [() => getNutrition("samosa", 100), () => getNutritionCoverage()]) {
    reply(404, { detail: "Not Found" });
    await assert.rejects(call, (error) => {
      assert.ok(error instanceof ApiError);
      assert.equal(error.status, 404);
      assert.match(error.message, /no nutrition endpoints/);
      assert.match(error.message, /Restart the FoodCal AI backend/);
      return true;
    });
  }
});

test("an unknown food keeps the backend's own message", async () => {
  reply(404, { detail: '"pasta" is not one of the 34 food classes.' });
  await assert.rejects(() => getNutrition("pasta", 100), (error) => {
    assert.match(error.message, /"pasta" is not one of the 34 food classes\. \(HTTP 404\)/);
    assert.doesNotMatch(error.message, /Restart/);
    return true;
  });
});

test("validation errors and mismatched responses are not shown as nutrition", async () => {
  reply(422, { detail: [{ msg: "Input should be greater than 0" }] });
  await assert.rejects(() => getNutrition("idli", 0), /greater than 0 \(HTTP 422\)/);
  reply(200, nutrition("samosa"));
  await assert.rejects(() => getNutrition("idli", 100), (error) => error.kind === "invalid-response");
  reply(200, { ...nutrition("idli"), estimated_nutrition: { ...VALUES, fat_g: null } });
  await assert.rejects(() => getNutrition("idli", 100), (error) => error.kind === "invalid-response");
});

// ---- History deletion ---------------------------------------------------

test("clear all history sends DELETE /api/history with the confirmation", async () => {
  reply(200, { deleted_records: 3, deleted_thumbnails: 2 });
  assert.deepEqual(await clearHistory(), { deleted_records: 3, deleted_thumbnails: 2 });
  reply(200, { deleted_records: 0, deleted_thumbnails: 0 });
  assert.deepEqual(await clearHistory(), { deleted_records: 0, deleted_thumbnails: 0 }, "empty history is not an error");
  assert.deepEqual(requested, Array(2).fill("http://127.0.0.1:8000/api/history?confirm=all"));
  assert.deepEqual(methods, ["DELETE", "DELETE"]);
});

test("a clear-all failure is reported, never treated as success", async () => {
  reply(500, { detail: "The history database could not be accessed." });
  await assert.rejects(clearHistory, (error) => error.kind === "http" && error.status === 500
    && /could not be accessed\. \(HTTP 500\)/.test(error.message));
  reply(200, { deleted: "x" });
  await assert.rejects(clearHistory, (error) => error.kind === "invalid-response");
  reply(405, { detail: "Method Not Allowed" });
  await assert.rejects(clearHistory, /cannot clear history \(HTTP 405\)\. Restart the FoodCal AI backend/);
});

test("deleting one record needs the backend to confirm that record", async () => {
  const id = "a".repeat(32);
  reply(200, { deleted: id });
  await deleteHistoryRecord(id);
  assert.deepEqual(requested, [`http://127.0.0.1:8000/api/history/${id}`]);
  assert.deepEqual(methods, ["DELETE"]);
  reply(200, { deleted: "b".repeat(32) });
  await assert.rejects(() => deleteHistoryRecord(id), (error) => error.kind === "invalid-response");
  reply(404, { detail: "Prediction not found." });
  await assert.rejects(() => deleteHistoryRecord(id), (error) => error.status === 404);
});
