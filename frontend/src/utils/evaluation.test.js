import assert from "node:assert/strict";
import { test } from "node:test";
import {
  MIN_CLASS_SAMPLES,
  activeConfusionMatrix,
  createVerification,
  isLowConfidence,
  isValidRealWorldReport,
  resolveClassLabel,
  withClassRates,
} from "./evaluation.js";

// Live metric counting is done by the backend from persisted labels and
// is covered by tests/test_history_store.py; these tests cover the
// client-side label validation and per-class rate thresholds.

const CLASSES = ["Baked Potato", "Hot Dog", "kadai_paneer", "pizza", "samosa"];

test("resolveClassLabel accepts known classes regardless of case and spacing", () => {
  assert.equal(resolveClassLabel("pizza", CLASSES), "pizza");
  assert.equal(resolveClassLabel("  Kadai Paneer ", CLASSES), "kadai_paneer");
  assert.equal(resolveClassLabel("hot_dog", CLASSES), "Hot Dog");
  assert.equal(resolveClassLabel("BAKED POTATO", CLASSES), "Baked Potato");
});

test("resolveClassLabel rejects unknown, partial and empty labels", () => {
  assert.equal(resolveClassLabel("pasta", CLASSES), null);
  assert.equal(resolveClassLabel("piz", CLASSES), null);
  assert.equal(resolveClassLabel("", CLASSES), null);
  assert.equal(resolveClassLabel("   ", CLASSES), null);
  assert.equal(resolveClassLabel(undefined, CLASSES), null);
});

test("createVerification: correct uses the predicted label", () => {
  assert.deepEqual(createVerification("pizza", "correct", "", CLASSES), {
    verification: { verdict: "correct", actualLabel: "pizza" },
  });
});

test("createVerification: incorrect requires a valid, different class", () => {
  assert.deepEqual(createVerification("pizza", "incorrect", "samosa", CLASSES), {
    verification: { verdict: "incorrect", actualLabel: "samosa" },
  });
  assert.deepEqual(createVerification("pizza", "incorrect", "kadai paneer", CLASSES), {
    verification: { verdict: "incorrect", actualLabel: "kadai_paneer" },
  });
  assert.match(createVerification("pizza", "incorrect", "", CLASSES).error, /Select the actual food/);
  assert.match(createVerification("pizza", "incorrect", "pasta", CLASSES).error, /not one of the 5 food classes/);
  assert.match(createVerification("pizza", "incorrect", "Pizza", CLASSES).error, /same as the prediction/);
  assert.match(createVerification("pizza", "incorrect", "samosa", []).error, /has not loaded/);
});

test("withClassRates: empty input stays empty", () => {
  assert.deepEqual(withClassRates([]), []);
});

test("withClassRates: rates withheld below the sample threshold", () => {
  assert.equal(MIN_CLASS_SAMPLES, 5);
  const [pizza] = withClassRates([{ food: "pizza", actual: 4, predicted: 3, correct: 2 }]);
  assert.equal(pizza.recall, null);
  assert.equal(pizza.precision, null);
  assert.equal(pizza.correct, 2, "raw counts are kept");
});

test("withClassRates: known rates once enough samples exist", () => {
  const rows = withClassRates([
    { food: "pizza", actual: 5, predicted: 8, correct: 4 },
    { food: "samosa", actual: 10, predicted: 4, correct: 4 },
  ]);
  assert.equal(rows[0].recall, 4 / 5);
  assert.equal(rows[0].precision, 4 / 8);
  assert.equal(rows[1].recall, 4 / 10);
  assert.equal(rows[1].precision, null, "only 4 predicted, below threshold");
});

test("withClassRates: zero correct gives 0, not null, when samples suffice", () => {
  const [row] = withClassRates([{ food: "samosa", actual: 6, predicted: 5, correct: 0 }]);
  assert.equal(row.recall, 0);
  assert.equal(row.precision, 0);
});

// ---- Real-world evaluation report ------------------------------------

function realWorldReport() {
  // 3 classes; a: 2 images (1 right, 1 -> b), b: 1 image (right), c: none.
  return {
    report_type: "real_world_evaluation",
    status: "ok",
    dataset: { valid_images: 3, invalid_images: [] },
    quality: { warnings: [] },
    metrics: {
      total: 3, correct: 2, incorrect: 1, accuracy: 2 / 3,
      macro_f1: 2 / 3, macro_f1_classes: 2, weighted_f1: 2 / 3,
      per_class: [
        { food: "a", support: 2, predicted: 1, precision: 1, recall: 0.5, f1: 2 / 3 },
        { food: "b", support: 1, predicted: 2, precision: 0.5, recall: 1, f1: 2 / 3 },
        { food: "c", support: 0, predicted: 0, precision: null, recall: null, f1: null },
      ],
      confusion_matrix: { labels: ["a", "b", "c"], matrix: [[1, 1, 0], [0, 1, 0], [0, 0, 0]] },
    },
  };
}

test("isValidRealWorldReport accepts a complete report with null rates for absent classes", () => {
  assert.equal(isValidRealWorldReport(realWorldReport()), true);
});

test("isValidRealWorldReport accepts an honest empty report only without metrics", () => {
  const empty = {
    report_type: "real_world_evaluation", status: "no_valid_images", metrics: null,
    dataset: { valid_images: 0, invalid_images: [] }, quality: { warnings: ["No valid real-world images were found."] },
  };
  assert.equal(isValidRealWorldReport(empty), true);
  assert.equal(isValidRealWorldReport({ ...empty, metrics: realWorldReport().metrics }), false);
  assert.equal(isValidRealWorldReport({ ...empty, dataset: { valid_images: 3, invalid_images: [] } }), false);
});

test("isValidRealWorldReport rejects inconsistent or incomplete reports", () => {
  const broken = [
    (r) => { r.report_type = "offline"; },
    (r) => { r.status = "estimated"; },
    (r) => { r.metrics.accuracy = undefined; },
    (r) => { r.metrics.accuracy = 1.2; },
    (r) => { r.metrics.correct = 3; }, // correct + incorrect != total
    (r) => { r.dataset.valid_images = 4; }, // metrics total != valid images
    (r) => { r.metrics.per_class[2].f1 = "n/a"; },
    (r) => { r.metrics.per_class.pop(); },
    (r) => { r.metrics.confusion_matrix.matrix[0].pop(); },
    (r) => { delete r.quality; },
  ];
  for (const breakIt of broken) {
    const report = realWorldReport();
    breakIt(report);
    assert.equal(isValidRealWorldReport(report), false, breakIt.toString());
  }
});

test("activeConfusionMatrix keeps classes that appear as actual or predicted labels", () => {
  const cm = activeConfusionMatrix({
    labels: ["a", "b", "c", "d"],
    // a: 2 images, 1 predicted as d; d never occurs as actual; b, c absent.
    matrix: [[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
  });
  assert.deepEqual(cm.labels, ["a", "d"]);
  assert.deepEqual(cm.matrix, [[1, 1], [0, 0]]);
  assert.deepEqual(cm.rowTotals, [2, 0]);
});

test("activeConfusionMatrix of an all-zero matrix is empty", () => {
  assert.deepEqual(activeConfusionMatrix({ labels: ["a", "b"], matrix: [[0, 0], [0, 0]] }), {
    labels: [], matrix: [], rowTotals: [],
  });
});

test("isLowConfidence flags only confidences strictly below a known threshold", () => {
  assert.equal(isLowConfidence(49.99, 50), true);
  assert.equal(isLowConfidence(50, 50), false);
  assert.equal(isLowConfidence(77.57, 50), false);
  assert.equal(isLowConfidence(56.81, 60), true, "threshold is configurable");
  assert.equal(isLowConfidence(10, null), false, "threshold not loaded yet");
  assert.equal(isLowConfidence(undefined, 50), false);
  assert.equal(isLowConfidence(Number.NaN, 50), false);
});
