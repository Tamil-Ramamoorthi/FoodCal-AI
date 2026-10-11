"""Compares two checkpoints on the same leakage-free test manifest.

    .venv-gpu\\Scripts\\python.exe -m evaluation.compare_models

Defaults: production checkpoint vs models/efficientnet_b0_v2, on
data/manifests/v2/test_clean.csv (3,304 images with no exact or near copy
in train or validation). Both models see identical images through the
API's own preprocessing (backend/app/models/classifier.py). Read-only.

Writes reports/model_comparison/comparison.json, comparison.md and the two
34x34 confusion matrices as CSV.

The report includes a promotion gate (see promotion_gate): a candidate is
only eligible to replace the old model if the measured improvement on this
test set is clear and no well-supported class gets clearly worse. The gate
never deploys anything, and evaluation on independent real-world photos is
still required before deployment.

With --feedback-summary (summary.json written by
training/build_feedback_manifest.py) the report also shows, for the classes
the approved corrections touch, whether those confusions changed.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from backend.app.models.classifier import load_classifier
from training.build_manifests import read_manifest

from .dataset_audit import ROOT, rel, utc_now
from .diagnose_reliability import predict
from .metrics import classification_metrics
from .reports import pct, write_json, write_text

DEFAULT_OLD = ROOT / "models" / "efficientnet_b0" / "food_efficientnet_b0_best.pth"
DEFAULT_NEW = ROOT / "models" / "efficientnet_b0_v2" / "food_efficientnet_b0_v2_best.pth"
DEFAULT_MANIFEST = ROOT / "data" / "manifests" / "v2" / "test_clean.csv"
DEFAULT_OUTPUT = ROOT / "reports" / "model_comparison"
WEAKEST = 8
BOOTSTRAP = 2000
# Promotion gate. Classes with fewer test images are too noisy to judge
# (one image is >5 F1 points), so the per-class rule only covers these.
GATE = {"max_p_value": 0.05, "min_class_support": 20, "max_class_f1_drop": 0.05}


def summarize(y_true, y_pred, class_names):
    m = classification_metrics(y_true, y_pred, class_names)
    precisions = [r["precision"] for r in m["per_class"]]
    undefined = sum(p is None for p in precisions)
    m["macro_precision"] = sum(p or 0.0 for p in precisions) / len(precisions)
    m["macro_precision_undefined_classes"] = undefined  # counted as 0 if any
    m["macro_recall"] = sum(r["recall"] for r in m["per_class"] if r["recall"] is not None) / m["macro_f1_classes"]
    return m


def mcnemar_exact(old_correct, new_correct):
    """Two-sided exact McNemar test on paired correctness."""
    b = int(np.sum(old_correct & ~new_correct))  # old right, new wrong
    c = int(np.sum(~old_correct & new_correct))  # old wrong, new right
    n, k = b + c, min(b, c)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return {"old_right_new_wrong": b, "old_wrong_new_right": c, "p_value": p}


def bootstrap_diff(y_true, old_pred, new_pred, class_names, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    acc, f1 = [], []
    for _ in range(BOOTSTRAP):
        idx = rng.integers(0, n, n)
        t, o, w = y_true[idx], old_pred[idx], new_pred[idx]
        acc.append(np.mean(w == t) - np.mean(o == t))
        f1.append(macro_f1_fast(t, w, len(class_names)) - macro_f1_fast(t, o, len(class_names)))
    ci = lambda values: [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]  # noqa: E731
    return {"accuracy_difference_95ci": ci(acc), "macro_f1_difference_95ci": ci(f1), "resamples": BOOTSTRAP}


def macro_f1_fast(t, p, k):
    cm = np.bincount(t * k + p, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    support, predicted = cm.sum(1), cm.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(support + predicted > 0, 2 * tp / (support + predicted), 0.0)
    return f1[support > 0].mean()


def promotion_gate(report, gate=GATE):
    """Offline evidence that `new` is better than `old`. Returns passed + reasons."""
    o, n = report["overall"]["old"], report["overall"]["new"]
    pt, ci = report["paired_test"], report["bootstrap"]["macro_f1_difference_95ci"]
    failures = []
    if not n["macro_f1"] > o["macro_f1"]:
        failures.append(f"macro-F1 did not improve ({o['macro_f1']:.4f} -> {n['macro_f1']:.4f})")
    if not ci[0] > 0:
        failures.append(f"95% bootstrap CI of the macro-F1 difference includes 0 or less ({ci[0]:+.4f} to {ci[1]:+.4f})")
    if not (pt["p_value"] < gate["max_p_value"] and pt["old_wrong_new_right"] > pt["old_right_new_wrong"]):
        failures.append(f"paired McNemar test does not favour the new model (p = {pt['p_value']:.3g}, "
                        f"{pt['old_wrong_new_right']} fixed vs {pt['old_right_new_wrong']} broken)")
    for c in report["per_class"]:
        if c["support"] >= gate["min_class_support"] and c["old"]["f1"] - c["new"]["f1"] > gate["max_class_f1_drop"]:
            failures.append(f"{c['food']} F1 dropped {(c['old']['f1'] - c['new']['f1']) * 100:.1f} pts "
                            f"({c['support']} test images)")
    return {
        "passed": not failures,
        "criteria": gate,
        "failures": failures,
        "note": "Offline gate only. It deploys nothing; real-world photos (data/real_world_test) must also be "
                "evaluated before the production checkpoint is changed.",
    }


def feedback_focus(report, summary_path):
    """Per-class metrics and confusion counts for the classes approved corrections touch."""
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    names = report["confusion_matrices"]["labels"]
    old_mat, new_mat = report["confusion_matrices"]["old"], report["confusion_matrices"]["new"]
    by_food = {c["food"]: c for c in report["per_class"]}
    pairs = []
    for actual, predicted in summary.get("correction_pairs", []):
        i, j = names.index(actual), names.index(predicted)
        pairs.append({"actual": actual, "predicted": predicted, "actual_images": sum(old_mat[i]),
                      "old": old_mat[i][j], "new": new_mat[i][j]})
    return {
        "summary": str(summary_path),
        "train_added": summary.get("train_added"),
        "classes": [by_food[f] for f in summary.get("affected_classes", []) if f in by_food],
        "correction_pairs": pairs,
        "note": "Measured on the independent test set, not on the corrected images (those are training data).",
    }


def write_matrix(path, labels, matrix):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual \\ predicted", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *row])


def run(args):
    auto = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(auto if args.device == "auto" else args.device)
    rows = read_manifest(args.manifest)
    old = load_classifier(Path(args.old), device)
    new = load_classifier(Path(args.new), device)
    if old.class_names != new.class_names:
        raise ValueError("The two checkpoints use different class lists or orders.")
    names = old.class_names
    if old.image_size != new.image_size:
        raise ValueError("The two checkpoints use different input sizes.")
    for row in rows:
        if names[row["class_index"]] != row["label"] or Path(row["path"]).parent.name != row["label"]:
            raise ValueError(f"Label / class-index / folder mismatch: {row['path']}")

    paths = [ROOT / row["path"] for row in rows]
    y_true = np.array([row["class_index"] for row in rows])
    print(f"Evaluating {len(rows)} images on {device} ...")
    results = {}
    for tag, classifier in (("old", old), ("new", new)):
        probs = predict(classifier, paths)
        pred = probs.argmax(1)
        results[tag] = {"pred": pred, "conf": probs.max(1), "metrics": summarize(y_true.tolist(), pred.tolist(), names)}

    i_idli, i_kulfi = names.index("idli"), names.index("kulfi")

    def kulfi_stats(m):
        mat = m["confusion_matrix"]["matrix"]
        idli_support = sum(mat[i_idli])
        return {
            "idli_images": idli_support,
            "idli_predicted_as_kulfi": mat[i_idli][i_kulfi],
            "idli_to_kulfi_rate": mat[i_idli][i_kulfi] / idli_support if idli_support else None,
            "non_kulfi_images_predicted_as_kulfi": sum(mat[i][i_kulfi] for i in range(len(names)) if i != i_kulfi),
            "kulfi_precision": m["per_class"][i_kulfi]["precision"],
            "kulfi_recall": m["per_class"][i_kulfi]["recall"],
        }

    old_m, new_m = results["old"]["metrics"], results["new"]["metrics"]
    weakest = [r["food"] for r in sorted(old_m["per_class"], key=lambda r: r["f1"])[:WEAKEST]]
    old_mat, new_mat = np.array(old_m["confusion_matrix"]["matrix"]), np.array(new_m["confusion_matrix"]["matrix"])
    off = old_mat.copy()
    np.fill_diagonal(off, 0)
    top_pairs = [divmod(int(k), len(names)) for k in np.argsort(-off, axis=None)[:15]]

    overall_keys = ("total", "correct", "accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1",
                    "macro_precision_undefined_classes")
    report = {
        "report_type": "model_comparison",
        "generated_at": utc_now(),
        "test_manifest": rel(args.manifest),
        "test_images": len(rows),
        "preprocessing": "backend/app/models/classifier.py (same as the API)",
        "checks": {"class_names_identical": True, "labels_match_class_index_and_folder": True},
        "models": {
            tag: {"checkpoint": rel(path), "epoch": c.epoch, "val_accuracy": c.val_accuracy}
            for tag, path, c in (("old", args.old, old), ("new", args.new, new))
        },
        "overall": {tag: {k: results[tag]["metrics"][k] for k in overall_keys} for tag in ("old", "new")},
        "paired_test": mcnemar_exact(results["old"]["pred"] == y_true, results["new"]["pred"] == y_true),
        "bootstrap": bootstrap_diff(y_true, results["old"]["pred"], results["new"]["pred"], names),
        "idli_kulfi": {tag: kulfi_stats(results[tag]["metrics"]) for tag in ("old", "new")},
        "per_class": [
            {"food": name,
             "support": old_m["per_class"][i]["support"],
             "old": {k: old_m["per_class"][i][k] for k in ("precision", "recall", "f1")},
             "new": {k: new_m["per_class"][i][k] for k in ("precision", "recall", "f1")}}
            for i, name in enumerate(names)
        ],
        "weakest_classes_of_old_model": weakest,
        "frequent_confusions_of_old_model": [
            {"actual": names[i], "predicted": names[j], "old": int(old_mat[i][j]), "new": int(new_mat[i][j]),
             "actual_images": int(old_mat[i].sum())}
            for i, j in top_pairs
        ],
        "confusion_matrices": {"labels": names, "old": old_mat.tolist(), "new": new_mat.tolist()},
    }
    report["promotion_gate"] = promotion_gate(report)
    if getattr(args, "feedback_summary", None):
        report["feedback_focus"] = feedback_focus(report, args.feedback_summary)
    out = Path(args.output_dir)
    write_json(out / "comparison.json", report)
    write_matrix(out / "confusion_matrix_old.csv", names, old_mat.tolist())
    write_matrix(out / "confusion_matrix_new.csv", names, new_mat.tolist())
    write_text(out / "comparison.md", markdown(report))
    print(f"Wrote {rel(out / 'comparison.md')}")
    return report


def markdown(r):
    o, n = r["overall"]["old"], r["overall"]["new"]
    pt, bs = r["paired_test"], r["bootstrap"]
    row = lambda label, key: f"| {label} | {pct(o[key])} | {pct(n[key])} | {(n[key] - o[key]) * 100:+.1f} pts |"  # noqa: E731
    lines = [
        "# Old vs new model on the leakage-free test set", "",
        f"Generated {r['generated_at']}. {r['test_images']} images from `{r['test_manifest']}`; "
        f"preprocessing: {r['preprocessing']}. Class lists identical; every label matches its folder and class index.", "",
        f"- Old: `{r['models']['old']['checkpoint']}` (epoch {r['models']['old']['epoch']})",
        f"- New: `{r['models']['new']['checkpoint']}` (epoch {r['models']['new']['epoch']})", "",
        "## Overall", "| Metric | Old | New | Change |", "|---|---|---|---|",
        row("Accuracy", "accuracy"), row("Macro precision", "macro_precision"), row("Macro recall", "macro_recall"),
        row("Macro F1", "macro_f1"), row("Weighted F1", "weighted_f1"), "",
        f"Correct: old {o['correct']} / new {n['correct']} of {o['total']}.",
        f"Paired exact McNemar test: old right & new wrong {pt['old_right_new_wrong']}, old wrong & new right "
        f"{pt['old_wrong_new_right']}, p = {pt['p_value']:.2g}.",
        f"Bootstrap 95% CI ({bs['resamples']} resamples): accuracy difference "
        f"{bs['accuracy_difference_95ci'][0] * 100:+.1f} to {bs['accuracy_difference_95ci'][1] * 100:+.1f} pts; macro-F1 difference "
        f"{bs['macro_f1_difference_95ci'][0] * 100:+.1f} to {bs['macro_f1_difference_95ci'][1] * 100:+.1f} pts.", "",
    ]
    g = r["promotion_gate"]
    lines += ["## Promotion gate", f"**{'PASSED' if g['passed'] else 'NOT PASSED'}** (offline test set only). {g['note']}"]
    lines += [f"- {f}" for f in g["failures"]] + [""]
    ff = r.get("feedback_focus")
    if ff:
        lines += [f"## Classes touched by approved feedback ({ff['train_added']} added training images)", ff["note"], "",
                  "| Actual → previously predicted | Actual images | Old | New |", "|---|---|---|---|"]
        lines += [f"| {p['actual']} → {p['predicted']} | {p['actual_images']} | {p['old']} | {p['new']} |"
                  for p in ff["correction_pairs"]]
        lines += ["", "| Class | Images | Old F1 | New F1 |", "|---|---|---|---|"]
        lines += [f"| {c['food']} | {c['support']} | {pct(c['old']['f1'])} | {pct(c['new']['f1'])} |" for c in ff["classes"]]
        lines.append("")
    lines += ["## Idli → kulfi", "| | Old | New |", "|---|---|---|"]
    ko, kn = r["idli_kulfi"]["old"], r["idli_kulfi"]["new"]
    lines += [
        f"| Idli images predicted as kulfi | {ko['idli_predicted_as_kulfi']} of {ko['idli_images']} ({pct(ko['idli_to_kulfi_rate'])}) | "
        f"{kn['idli_predicted_as_kulfi']} of {kn['idli_images']} ({pct(kn['idli_to_kulfi_rate'])}) |",
        f"| Non-kulfi images predicted as kulfi | {ko['non_kulfi_images_predicted_as_kulfi']} | {kn['non_kulfi_images_predicted_as_kulfi']} |",
        f"| Kulfi precision | {pct(ko['kulfi_precision'])} | {pct(kn['kulfi_precision'])} |",
        f"| Kulfi recall | {pct(ko['kulfi_recall'])} | {pct(kn['kulfi_recall'])} |", "",
        "Note: the clean test set contains only dataset-style idli photos; it says nothing about differently presented real photos.", "",
        f"## Weakest {len(r['weakest_classes_of_old_model'])} classes of the old model (F1)",
        "| Class | Images | Old P / R / F1 | New P / R / F1 |", "|---|---|---|---|",
    ]
    by_food = {c["food"]: c for c in r["per_class"]}
    fmt = lambda m: f"{pct(m['precision'])} / {pct(m['recall'])} / {pct(m['f1'])}"  # noqa: E731
    lines += [f"| {f} | {by_food[f]['support']} | {fmt(by_food[f]['old'])} | {fmt(by_food[f]['new'])} |"
              for f in r["weakest_classes_of_old_model"]]
    lines += ["", "## Most frequent confusions of the old model", "| Actual → predicted | Actual images | Old | New |", "|---|---|---|---|"]
    lines += [f"| {c['actual']} → {c['predicted']} | {c['actual_images']} | {c['old']} | {c['new']} |"
              for c in r["frequent_confusions_of_old_model"]]
    lines += ["", "## All classes", "| Class | Images | Old P / R / F1 | New P / R / F1 | F1 change |", "|---|---|---|---|---|"]
    lines += [f"| {c['food']} | {c['support']} | {fmt(c['old'])} | {fmt(c['new'])} | {(c['new']['f1'] - c['old']['f1']) * 100:+.1f} |"
              for c in sorted(r["per_class"], key=lambda c: c["new"]["f1"] - c["old"]["f1"])]
    lines += ["", "Full 34×34 confusion matrices: `confusion_matrix_old.csv`, `confusion_matrix_new.csv` (rows actual, columns predicted)."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare two checkpoints on the leakage-free test manifest.")
    parser.add_argument("--old", type=Path, default=DEFAULT_OLD)
    parser.add_argument("--new", type=Path, default=DEFAULT_NEW)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--feedback-summary", type=Path, help="summary.json from training/build_feedback_manifest.py")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
