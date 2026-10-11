// Pure helpers for the nutrition estimate. Values come only from the
// backend's reference data (USDA FoodData Central records chosen by the
// FoodCal AI nutrition research); nothing here invents, defaults or
// zero-fills a nutrient.

import { formatFood } from "./evaluation.js";

export const DEFAULT_SERVING_G = 100;
export const MAX_SERVING_G = 2000;

export const NUTRIENTS = [
  { key: "calories_kcal", label: "Calories", unit: "kcal", decimals: 0 },
  { key: "protein_g", label: "Protein", unit: "g", decimals: 1 },
  { key: "carbohydrates_g", label: "Carbohydrates", unit: "g", decimals: 1 },
  { key: "fat_g", label: "Fat", unit: "g", decimals: 1 },
];

export const MATCH_TYPES = {
  DIRECT_MATCH: { label: "Direct match", style: "success" },
  APPROXIMATE_MATCH: { label: "Approximate match", style: "warning" },
  UNAVAILABLE: { label: "Unavailable", style: "muted" },
};

// The research's data-quality rating; NONE (unavailable) gets no pill.
export const DATA_QUALITY = {
  HIGH: { label: "High quality", style: "success" },
  MEDIUM: { label: "Medium quality", style: "muted" },
  LOW: { label: "Low quality", style: "warning" },
};

const VERIFICATION = {
  SOURCE_RECORD_READ_VIA_FDC_MIRROR_NOT_INSPECTED_ON_USDA_SITE:
    "The research read this record through a public FoodData Central mirror, not on the USDA site.",
  "SOURCE_RECORD_READ_VIA_FDC_MIRROR; SOME_VALUES_ARITHMETICALLY SCALED_FROM_DECLARED_SERVING":
    "The research read this record through a public FoodData Central mirror and scaled some values from the label's declared serving.",
  REQUIRES_RESEARCH: "Further research is needed to find a reliable reference for this food.",
};

// Plain-language form of the research's verification status (shown verbatim when unknown).
export function verificationText(status) {
  return VERIFICATION[status] || status || "";
}

// Caveats for values that are shown but should be treated with care.
// Approximate matches carry their own explanation in the limitations.
export function qualityWarnings(data) {
  const warnings = [];
  if (data?.flagged_for_review) {
    warnings.push("Flagged for review: the research questions whether this reference is typical of the dish. Read the limitations before relying on it.");
  }
  if (data?.available && data.data_quality === "LOW") {
    warnings.push("Low data quality: the research rates this reference as a weak fit for the dish, so treat the values as indicative only.");
  }
  return warnings;
}

// Returns { grams } for a valid serving size, otherwise { error }.
export function parseServingGrams(input) {
  const text = typeof input === "number" ? String(input) : String(input ?? "").trim();
  if (!text) return { error: "Enter the serving size in grams." };
  if (!/^\d*\.?\d+$|^\d+\.$/.test(text)) return { error: "Serving size must be a number of grams." };
  const grams = Number(text);
  if (!Number.isFinite(grams)) return { error: "Serving size must be a number of grams." };
  if (grams <= 0) return { error: "Serving size must be more than 0 g." };
  if (grams > MAX_SERVING_G) return { error: `Serving size must be at most ${MAX_SERVING_G} g.` };
  return { grams };
}

export function formatNutrient(value, nutrient) {
  return `${Number(value).toFixed(nutrient.decimals)} ${nutrient.unit}`;
}

export function formatGrams(grams) {
  return `${Number(grams).toLocaleString("en-US", { maximumFractionDigits: 1 })} g`;
}

function validNutrients(values) {
  return values !== null && typeof values === "object"
    && NUTRIENTS.every(({ key }) => Number.isFinite(values[key]) && values[key] >= 0);
}

export function isValidNutritionResponse(data) {
  if (!data || typeof data.food_class !== "string" || !(data.match_type in MATCH_TYPES)) return false;
  if (typeof data.available !== "boolean" || data.available !== (data.match_type !== "UNAVAILABLE")) return false;
  if (!Number.isFinite(data.serving_size_g) || typeof data.notice !== "string") return false;
  if (!data.available) return data.nutrition_per_100g === null && data.estimated_nutrition === null;
  return validNutrients(data.nutrition_per_100g)
    && validNutrients(data.estimated_nutrition)
    && typeof data.source?.dataset === "string"
    && typeof data.source?.food_description === "string"
    && typeof data.source?.record_id === "string"
    && typeof data.source?.record_label === "string";
}

export function isValidNutritionCoverage(data) {
  return Boolean(data)
    && Number.isInteger(data.total_classes)
    && Array.isArray(data.foods)
    && data.foods.length === data.total_classes
    && data.foods.every((food) => typeof food?.food_class === "string" && food.match_type in MATCH_TYPES);
}

// What the nutrition panel shows. A response is only shown when it
// belongs to the current food and serving size, so stale or in-flight
// results never appear as values.
export function nutritionView({ servingError, requestKey, response }) {
  if (servingError) return "invalid";
  if (!response || response.key !== requestKey || response.status === "loading") return "loading";
  if (response.status === "error") return "error";
  return response.data.available ? "ready" : "unavailable";
}

export function nutritionRequestKey(food, parsed) {
  return food && !parsed.error ? `${food}|${parsed.grams}` : null;
}

export function predictionOptions(topPredictions) {
  return (topPredictions || []).map((prediction) => ({
    value: prediction.food,
    label: `${formatFood(prediction.food)} (${Number(prediction.confidence).toFixed(1)}%)`,
  }));
}

// Messages shown above the values; none of them contain a number that
// could be mistaken for nutrition data.
export function nutritionWarnings({ lowConfidence, food, predictedFood }) {
  const warnings = [];
  if (lowConfidence) {
    warnings.push(`The model is unsure about this image. Nutrition for ${formatFood(food)} only applies if the food really is ${formatFood(food)}; check the selected food first.`);
  }
  if (predictedFood && food !== predictedFood) {
    warnings.push(`Showing ${formatFood(food)}, which you selected instead of the top match (${formatFood(predictedFood)}).`);
  }
  return warnings;
}

export function filterFoods(foods, query) {
  const words = String(query || "").toLowerCase().replace(/_/g, " ").split(/\s+/).filter(Boolean);
  if (!words.length) return foods;
  return foods.filter((food) => {
    const name = formatFood(food).toLowerCase();
    return words.every((word) => name.includes(word));
  });
}
