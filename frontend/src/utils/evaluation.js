// Pure helpers for user-verified evaluation. Correctness is decided only
// by comparing a user-verified actual label with the predicted label;
// model confidence is never used.

// Per-class rates are shown only once a class has at least this many
// verified samples; below that only raw counts are reported.
export const MIN_CLASS_SAMPLES = 5;

export function formatFood(name) {
  return name.replaceAll("_", " ");
}

function normalizeLabel(label) {
  return label.trim().toLowerCase().replace(/[\s_]+/g, " ");
}

// Returns the canonical class name for user input, ignoring case and
// spaces vs underscores, or null when it is not exactly one known class.
export function resolveClassLabel(input, classes) {
  if (typeof input !== "string" || !input.trim()) return null;
  const wanted = normalizeLabel(input);
  const matches = classes.filter((name) => normalizeLabel(name) === wanted);
  return matches.length === 1 ? matches[0] : null;
}

// verdict is "correct" or "incorrect". Returns { verification } or { error }.
export function createVerification(predictedFood, verdict, actualInput, classes) {
  if (verdict === "correct") {
    return { verification: { verdict: "correct", actualLabel: predictedFood } };
  }

  if (!classes.length) {
    return {
      error: "The food class list has not loaded from the backend, so the actual label cannot be validated yet.",
    };
  }

  if (!actualInput.trim()) {
    return { error: `Select the actual food from the ${classes.length} known classes before reporting an incorrect prediction.` };
  }

  const actualLabel = resolveClassLabel(actualInput, classes);
  if (!actualLabel) {
    return {
      error: `"${actualInput.trim()}" is not one of the ${classes.length} food classes this model recognizes. Choose a class from the list.`,
    };
  }

  if (actualLabel === predictedFood) {
    return { error: "The actual food you entered is the same as the prediction. Use \"Correct\" instead." };
  }

  return { verification: { verdict: "incorrect", actualLabel } };
}

// Per-class counts come from GET /api/evaluation/live, which aggregates
// persisted user-verified labels only. Rates are withheld until a class
// has enough verified samples.
export function withClassRates(perClass) {
  return perClass.map((item) => ({
    ...item,
    // Share of verified images of this food that the model got right.
    recall: item.actual >= MIN_CLASS_SAMPLES ? item.correct / item.actual : null,
    // Share of verified predictions of this food that were right.
    precision: item.predicted >= MIN_CLASS_SAMPLES ? item.correct / item.predicted : null,
  }));
}

// ---- Real-world evaluation report ------------------------------------
// The report is produced by evaluation/evaluate_real_world.py and served
// unchanged by GET /api/evaluation/real-world. Undefined rates (a class
// with no images, or never predicted) are null, never 0 or 1.

const isCount = (value) => Number.isInteger(value) && value >= 0;
const isRate = (value) => Number.isFinite(value) && value >= 0 && value <= 1;
const isRateOrNull = (value) => value === null || isRate(value);

function isValidMetrics(metrics) {
  if (!metrics || typeof metrics !== "object") return false;
  const { per_class: perClass, confusion_matrix: cm } = metrics;
  const size = Array.isArray(cm?.labels) ? cm.labels.length : -1;
  return ["total", "correct", "incorrect", "macro_f1_classes"].every((key) => isCount(metrics[key]))
    && metrics.total > 0
    && metrics.correct + metrics.incorrect === metrics.total
    && ["accuracy", "macro_f1", "weighted_f1"].every((key) => isRate(metrics[key]))
    && Array.isArray(perClass)
    && perClass.length === size
    && perClass.every((row) => typeof row?.food === "string"
      && isCount(row.support) && isCount(row.predicted)
      && ["precision", "recall", "f1"].every((key) => isRateOrNull(row[key])))
    && Array.isArray(cm.matrix)
    && cm.matrix.length === size
    && cm.matrix.every((row) => Array.isArray(row) && row.length === size && row.every(isCount));
}

export function isValidRealWorldReport(data) {
  if (data?.report_type !== "real_world_evaluation") return false;
  if (!isCount(data.dataset?.valid_images) || !Array.isArray(data.dataset?.invalid_images)) return false;
  if (!Array.isArray(data.quality?.warnings)) return false;
  if (data.status === "no_valid_images") return data.metrics === null && data.dataset.valid_images === 0;
  return data.status === "ok"
    && isValidMetrics(data.metrics)
    && data.metrics.total === data.dataset.valid_images;
}

// Keeps only classes that occur as an actual or a predicted label, so a
// small real-world set does not render a mostly empty 34 x 34 grid.
export function activeConfusionMatrix({ labels, matrix }) {
  const active = labels
    .map((_, index) => index)
    .filter((index) => matrix[index].some(Boolean) || matrix.some((row) => row[index]));
  return {
    labels: active.map((index) => labels[index]),
    matrix: active.map((i) => active.map((j) => matrix[i][j])),
    rowTotals: active.map((i) => matrix[i].reduce((sum, value) => sum + value, 0)),
  };
}

// ---- Prediction confidence ---------------------------------------------
// The threshold (percent) comes from the backend, which chose it on the
// validation split. Unknown threshold or confidence never flags a result.
export function isLowConfidence(confidence, threshold) {
  return Number.isFinite(confidence) && Number.isFinite(threshold) && confidence < threshold;
}
