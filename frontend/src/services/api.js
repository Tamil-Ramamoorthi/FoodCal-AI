// Explicit .js extensions let Node's test runner import this module too.
import { isValidRealWorldReport } from "../utils/evaluation.js";
import { isValidNutritionCoverage, isValidNutritionResponse } from "../utils/nutrition.js";

// VITE_API_BASE_URL lets a second dev server target a different backend
// (used by the browser tests with a temporary database).
export const API_BASE_URL = (
  import.meta.env?.VITE_API_BASE_URL || "http://127.0.0.1:8000"
).replace(/\/+$/, "");

// The backend's FOODCAL_ADMIN_TOKEN, entered in Settings. It is kept only
// in this browser (never in the bundle) and sent only with requests that
// change or delete saved history. Storage can be unavailable (private
// windows, blocked site data); the app then works without a token.
const ADMIN_TOKEN_KEY = "foodcal-admin-token";
let memoryAdminToken = "";

export function getAdminToken() {
  try {
    return globalThis.localStorage?.getItem(ADMIN_TOKEN_KEY) || memoryAdminToken;
  } catch {
    return memoryAdminToken;
  }
}

export function setAdminToken(token) {
  memoryAdminToken = token.trim();
  try {
    if (memoryAdminToken) globalThis.localStorage?.setItem(ADMIN_TOKEN_KEY, memoryAdminToken);
    else globalThis.localStorage?.removeItem(ADMIN_TOKEN_KEY);
  } catch {
    // Kept in memory for this page only.
  }
}

function adminHeaders(headers = {}) {
  const token = getAdminToken();
  return token ? { ...headers, "X-FoodCal-Admin-Token": token } : headers;
}

const HEALTH_TIMEOUT_MS = 5000;
const HISTORY_TIMEOUT_MS = 10000;
const PREDICT_TIMEOUT_MS = 60000;

// kind is one of: "network", "timeout", "http", "invalid-response".
// detail is the backend's own error detail, when it sent one.
export class ApiError extends Error {
  constructor(kind, message, status = null, detail = "") {
    super(message);
    this.name = "ApiError";
    this.kind = kind;
    this.status = status;
    this.detail = detail;
  }
}

// True when the backend could not be reached at all, as opposed to the
// backend answering with an error or an unexpected body.
export function isConnectionError(error) {
  return error instanceof ApiError
    && (error.kind === "network" || error.kind === "timeout");
}

function formatDetail(detail) {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail.map((item) => item?.msg).filter(Boolean).join("; ");
  }
  return "";
}

async function request(path, options, timeoutMs) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    let response;

    try {
      response = await fetch(`${API_BASE_URL}${path}`, {
        ...options,
        signal: controller.signal,
      });
    } catch (error) {
      if (controller.signal.aborted) {
        throw new ApiError(
          "timeout",
          `The FoodCal AI backend did not respond within ${timeoutMs / 1000} seconds.`,
        );
      }
      // Browsers report connection refused, DNS failures and CORS
      // rejections identically, so the message names both possibilities.
      throw new ApiError(
        "network",
        `Cannot reach the FoodCal AI backend at ${API_BASE_URL}. `
          + "Check that the server is running and that its CORS settings "
          + `allow ${window.location.origin}. (${error.message})`,
      );
    }

    let body = null;
    try {
      body = await response.json();
    } catch {
      if (controller.signal.aborted) {
        throw new ApiError(
          "timeout",
          `The FoodCal AI backend did not respond within ${timeoutMs / 1000} seconds.`,
        );
      }
    }

    if (!response.ok) {
      const detail = formatDetail(body?.detail);
      throw new ApiError(
        "http",
        detail
          ? `${detail} (HTTP ${response.status})`
          : `The backend returned HTTP ${response.status}.`,
        response.status,
        detail,
      );
    }

    if (body === null) {
      throw new ApiError(
        "invalid-response",
        "The backend returned a response that is not valid JSON.",
        response.status,
      );
    }

    return body;
  } finally {
    clearTimeout(timer);
  }
}

export async function checkApi() {
  const data = await request("/", {}, HEALTH_TIMEOUT_MS);

  if (typeof data.message !== "string" || !Number.isFinite(data.classes)) {
    throw new ApiError(
      "invalid-response",
      `The service at ${API_BASE_URL} is not the FoodCal AI backend.`,
    );
  }

  return data;
}

export async function getFoodClasses() {
  const data = await request("/api/classes", {}, HEALTH_TIMEOUT_MS);

  if (!Array.isArray(data.classes)) {
    throw new ApiError(
      "invalid-response",
      "The backend returned an invalid food class list.",
    );
  }

  return data;
}

export async function getEvaluationReport() {
  const data = await request("/api/evaluation", {}, HEALTH_TIMEOUT_MS);

  const valid = ["accuracy", "macro_f1", "weighted_f1", "test_images"]
    .every((key) => Number.isFinite(data[key]))
    && Array.isArray(data.per_class)
    && data.per_class.every(
      (item) => typeof item?.food === "string"
        && ["precision", "recall", "f1", "support"].every((key) => Number.isFinite(item[key])),
    );

  if (!valid) {
    throw new ApiError(
      "invalid-response",
      "The backend returned an incomplete evaluation report.",
    );
  }

  return data;
}

// Resolves to null when no real-world report has been generated yet.
export async function getRealWorldEvaluation() {
  let data;
  try {
    data = await request("/api/evaluation/real-world", {}, HISTORY_TIMEOUT_MS);
  } catch (error) {
    if (error instanceof ApiError && error.status === 404) return null;
    throw error;
  }

  if (!isValidRealWorldReport(data)) {
    throw new ApiError(
      "invalid-response",
      "The backend returned an incomplete real-world evaluation report.",
    );
  }

  return data;
}

function isValidPrediction(data) {
  return typeof data.predicted_food === "string"
    && Number.isFinite(data.confidence)
    && Array.isArray(data.top_predictions)
    && data.top_predictions.every(
      (item) => typeof item?.food === "string" && Number.isFinite(item?.confidence),
    );
}

export async function predictFood(file) {
  const formData = new FormData();
  formData.append("file", file);

  // No Content-Type header: the browser sets multipart/form-data with
  // the correct boundary for FormData bodies.
  const data = await request(
    "/api/predict",
    { method: "POST", body: formData },
    PREDICT_TIMEOUT_MS,
  );

  if (!isValidPrediction(data)) {
    throw new ApiError(
      "invalid-response",
      "The backend returned an incomplete prediction response.",
    );
  }

  return data;
}

// ---- Persisted prediction history ------------------------------------

const STATUSES = ["unverified", "correct", "incorrect"];

function isValidRecord(record) {
  return typeof record?.id === "string"
    && typeof record.predicted_food === "string"
    && Number.isFinite(record.confidence)
    && typeof record.created_at === "string"
    && STATUSES.includes(record.verification_status)
    && (record.actual_food === null || typeof record.actual_food === "string");
}

function validatedRecord(data) {
  if (!isValidRecord(data)) {
    throw new ApiError("invalid-response", "The backend returned an invalid history record.");
  }
  return data;
}

export function thumbnailSrc(record) {
  return record.thumbnail_url ? `${API_BASE_URL}${record.thumbnail_url}` : null;
}

export async function getHistory({ limit = 10, offset = 0, query = "" } = {}) {
  const params = new URLSearchParams({ limit: String(limit), offset: String(offset) });
  if (query.trim()) params.set("q", query.trim());

  const data = await request(`/api/history?${params}`, {}, HISTORY_TIMEOUT_MS);

  if (!Array.isArray(data.items) || !Number.isFinite(data.total) || !data.items.every(isValidRecord)) {
    throw new ApiError("invalid-response", "The backend returned an invalid history list.");
  }

  return data;
}

export async function saveFeedback(predictionId, actualFood) {
  const data = await request(
    `/api/history/${encodeURIComponent(predictionId)}/feedback`,
    {
      method: "PATCH",
      headers: adminHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify({ actual_food: actualFood }),
    },
    HISTORY_TIMEOUT_MS,
  );
  return validatedRecord(data);
}

export async function clearFeedback(predictionId) {
  const data = await request(
    `/api/history/${encodeURIComponent(predictionId)}/feedback`,
    { method: "DELETE", headers: adminHeaders() },
    HISTORY_TIMEOUT_MS,
  );
  return validatedRecord(data);
}

export async function deleteHistoryRecord(predictionId) {
  const data = await request(
    `/api/history/${encodeURIComponent(predictionId)}`,
    { method: "DELETE", headers: adminHeaders() },
    HISTORY_TIMEOUT_MS,
  );
  if (data.deleted !== predictionId) {
    throw new ApiError("invalid-response", "The backend did not confirm the deletion.");
  }
  return data;
}

// Deletes every saved prediction on the backend (there are no user
// accounts) and the thumbnails they used.
export async function clearHistory() {
  let data;
  try {
    data = await request("/api/history?confirm=all", { method: "DELETE", headers: adminHeaders() }, HISTORY_TIMEOUT_MS);
  } catch (error) {
    // A backend started before this route existed only allows GET here.
    if (error instanceof ApiError && error.status === 405) {
      throw new ApiError(
        "http",
        `The backend at ${API_BASE_URL} cannot clear history (HTTP 405). `
          + "Restart the FoodCal AI backend so it runs the current code.",
        405,
        error.detail,
      );
    }
    throw error;
  }

  if (!Number.isInteger(data.deleted_records) || !Number.isInteger(data.deleted_thumbnails)) {
    throw new ApiError("invalid-response", "The backend did not confirm that history was cleared.");
  }
  return data;
}

export async function getLiveEvaluation() {
  const data = await request("/api/evaluation/live", {}, HISTORY_TIMEOUT_MS);

  const valid = ["total_predictions", "verified", "correct", "incorrect"]
    .every((key) => Number.isInteger(data[key]))
    && (data.accuracy === null || Number.isFinite(data.accuracy))
    && Array.isArray(data.per_class);

  if (!valid) {
    throw new ApiError("invalid-response", "The backend returned invalid live evaluation data.");
  }

  return data;
}

// ---- Nutrition reference values -------------------------------------

// FastAPI answers "Not Found" for a path it has no route for. For the
// nutrition routes that means the backend process predates them (or the
// API URL points elsewhere), not that a food class is unknown, which the
// backend reports with its own message.
function explainMissingRoute(error) {
  if (error instanceof ApiError && error.status === 404 && error.detail === "Not Found") {
    return new ApiError(
      "http",
      `The backend at ${API_BASE_URL} has no nutrition endpoints (HTTP 404). `
        + "Restart the FoodCal AI backend so it runs the current code.",
      404,
      error.detail,
    );
  }
  return error;
}

export async function getNutritionCoverage() {
  let data;
  try {
    data = await request("/api/nutrition", {}, HISTORY_TIMEOUT_MS);
  } catch (error) {
    throw explainMissingRoute(error);
  }

  if (!isValidNutritionCoverage(data)) {
    throw new ApiError("invalid-response", "The backend returned an invalid nutrition coverage list.");
  }

  return data;
}

// Reference values for one food class scaled to a user-entered weight.
export async function getNutrition(foodClass, grams) {
  const params = new URLSearchParams({ grams: String(grams) });
  let data;
  try {
    data = await request(
      `/api/nutrition/${encodeURIComponent(foodClass)}?${params}`,
      {},
      HISTORY_TIMEOUT_MS,
    );
  } catch (error) {
    throw explainMissingRoute(error);
  }

  if (!isValidNutritionResponse(data) || data.food_class !== foodClass) {
    throw new ApiError("invalid-response", "The backend returned an incomplete nutrition response.");
  }

  return data;
}
