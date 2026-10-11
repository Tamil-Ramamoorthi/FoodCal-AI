"""Compares two real-world evaluation reports (e.g. production vs v2) made by
evaluation/evaluate_real_world.py on the SAME images. Read-only; reads only
the two JSON reports.

    .venv-gpu\\Scripts\\python.exe -m evaluation.compare_real_world

Defaults: reports/real_world/real_world_evaluation.json (production) vs
reports/real_world_v2/real_world_evaluation.json (v2). Writes
reports/real_world_comparison/comparison.json, comparison.md and both
confusion matrices as CSV.

No comparison is made unless both reports evaluated exactly the same
images (same paths, class-folder labels and file SHA-256) with the same
class mapping. If either report is missing, unreadable, has no valid images
or is incompatible with the other, a "pending" report explaining why is
written and no metric is claimed.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .compare_models import bootstrap_diff, mcnemar_exact, write_matrix
from .dataset_audit import ROOT, rel, utc_now
from .evaluate_real_world import REPORT_NAME, image_manifest, metrics_for
from .reports import _bullets, _table, food, pct, write_json, write_text

DEFAULT_OLD = ROOT / "reports" / "real_world" / f"{REPORT_NAME}.json"
DEFAULT_NEW = ROOT / "reports" / "real_world_v2" / f"{REPORT_NAME}.json"
DEFAULT_OUTPUT = ROOT / "reports" / "real_world_comparison"
OVERALL_KEYS = ("total", "correct", "accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1",
                "macro_f1_classes")


def load_report(path):
    path = Path(path)
    if not path.is_file():
        return None, f"`{rel(path)}` does not exist; run evaluation.evaluate_real_world for that checkpoint first."
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("report_type") != "real_world_evaluation":
        raise ValueError(f"{path} is not a real-world evaluation report.")
    return report, None


def manifest_rows(report):
    return sorted([p["path"], p["actual"], p["sha256"]] for p in report.get("predictions") or [])


def check_same_images(old, new):
    if [c["name"] for c in old["class_mapping"]] != [c["name"] for c in new["class_mapping"]]:
        raise ValueError("The two reports use different class mappings.")
    old_rows, new_rows = manifest_rows(old), manifest_rows(new)
    if old_rows != new_rows:
        only_old = {r[0] for r in old_rows} - {r[0] for r in new_rows}
        only_new = {r[0] for r in new_rows} - {r[0] for r in old_rows}
        raise ValueError(
            "The two reports did not evaluate the same images "
            f"({len(old_rows)} vs {len(new_rows)}; {len(only_old)} only in old, {len(only_new)} only in new, "
            "others differ in label or file content). Re-run both evaluations on the same folder.")
    for report in (old, new):  # the stored digest must describe the stored predictions
        stored = report.get("image_manifest")
        if stored and stored != image_manifest(report["predictions"]):
            raise ValueError(f"The image manifest of {report['checkpoint']['path']} does not match its predictions.")
    return old_rows


def pending(old_path, new_path, reasons):
    return {
        "report_type": "real_world_comparison",
        "status": "pending",
        "generated_at": utc_now(),
        "reports": {"old": rel(old_path), "new": rel(new_path)},
        "reasons": reasons,
        "note": "No real-world metric is claimed for either model until both are evaluated on the same valid, "
                "independent photos in data/real_world_test/.",
    }


def compare(old_path=DEFAULT_OLD, new_path=DEFAULT_NEW):
    """Returns an `ok` comparison, or a `pending` one that says why it cannot be made."""
    reasons = []
    try:
        old, old_missing = load_report(old_path)
        new, new_missing = load_report(new_path)
    except ValueError as error:
        return pending(old_path, new_path, [str(error)])
    reasons = [r for r in (old_missing, new_missing) if r]
    for tag, report in (("old", old), ("new", new)):
        if report is not None and report["status"] != "ok":
            reasons.append(f"The {tag} report ({report['checkpoint']['path']}) has no valid real-world images "
                           f"(status `{report['status']}`).")
    if reasons:
        return pending(old_path, new_path, reasons)

    try:
        rows = check_same_images(old, new)
    except ValueError as error:
        return pending(old_path, new_path, [f"The reports are incompatible: {error}"])
    return compare_loaded(old, new, old_path, new_path, rows)


def compare_loaded(old, new, old_path, new_path, rows):
    """The `ok` comparison of two loaded reports already checked by check_same_images."""
    names = [c["name"] for c in old["class_mapping"]]
    index = {n: i for i, n in enumerate(names)}
    old_by_path = {p["path"]: p for p in old["predictions"]}
    new_by_path = {p["path"]: p for p in new["predictions"]}
    paths = [r[0] for r in rows]
    old_preds, new_preds = [old_by_path[p] for p in paths], [new_by_path[p] for p in paths]
    y_true = np.array([index[p["actual"]] for p in old_preds])
    y_old = np.array([index[p["predicted"]] for p in old_preds])
    y_new = np.array([index[p["predicted"]] for p in new_preds])

    m = {"old": metrics_for(old_preds, names), "new": metrics_for(new_preds, names)}
    mats = {tag: np.array(m[tag]["confusion_matrix"]["matrix"]) for tag in m}
    off = mats["old"] + mats["new"]
    np.fill_diagonal(off, 0)
    pairs = [divmod(int(k), len(names)) for k in np.argsort(-off, axis=None, kind="stable")[:15] if off.flat[k]]

    quality = new["quality"]
    limitations = [
        f"{len(rows)} images in {m['old']['macro_f1_classes']} of {len(names)} classes; classes without images "
        "are not measured and are excluded from macro averages.",
    ]
    if quality.get("classes_below_minimum"):
        limitations.append(f"{len(quality['classes_below_minimum'])} class(es) have fewer than "
                           f"{quality['min_images_per_class']} images; their per-class metrics are unreliable.")
    if quality.get("exact_overlaps") or quality.get("near_duplicate_overlaps"):
        limitations.append(f"{len(quality['exact_overlaps'])} exact and {len(quality['near_duplicate_overlaps'])} "
                           "possible near-duplicate overlaps with training sources; those images are not independent.")
    limitations.append(f"Independence status: `{quality.get('independence_status')}`. {quality.get('independence_note', '')}")
    limitations.append("Confidence intervals below are wide for small samples; a difference inside them is not "
                       "evidence that either model is better.")

    return {
        "report_type": "real_world_comparison",
        "status": "ok",
        "generated_at": utc_now(),
        "reports": {"old": rel(old_path), "new": rel(new_path)},
        "models": {tag: r["checkpoint"] for tag, r in (("old", old), ("new", new))},
        "image_manifest": image_manifest(old["predictions"]),
        "checks": {"same_class_mapping": True, "same_images_labels_and_file_hashes": True},
        "overall": {tag: {k: m[tag][k] for k in OVERALL_KEYS} for tag in m},
        "paired_test": mcnemar_exact(y_old == y_true, y_new == y_true),
        "bootstrap": bootstrap_diff(y_true, y_old, y_new, names),
        "per_class": [
            {"food": n, "support": m["old"]["per_class"][i]["support"],
             "old": {k: m["old"]["per_class"][i][k] for k in ("precision", "recall", "f1")},
             "new": {k: m["new"]["per_class"][i][k] for k in ("precision", "recall", "f1")}}
            for i, n in enumerate(names) if m["old"]["per_class"][i]["support"]
        ],
        "frequent_confusions": [{"actual": names[i], "predicted": names[j], "old": int(mats["old"][i][j]),
                                 "new": int(mats["new"][i][j])} for i, j in pairs],
        "changed_predictions": [
            {"path": o["path"], "actual": o["actual"], "old": o["predicted"], "new": w["predicted"]}
            for o, w in zip(old_preds, new_preds) if o["predicted"] != w["predicted"]
        ],
        "confusion_matrices": {"labels": names, "old": mats["old"].tolist(), "new": mats["new"].tolist()},
        "limitations": limitations,
    }


def markdown(r):
    lines = ["# Real-world comparison", "", f"Generated {r['generated_at']}.",
             f"- Old: `{r['reports']['old']}`", f"- New: `{r['reports']['new']}`", ""]
    if r["status"] != "ok":
        return "\n".join(lines + ["**Status: pending.**", "", _bullets(r["reasons"]), "", r["note"]]) + "\n"
    o, n = r["overall"]["old"], r["overall"]["new"]
    pt, bs = r["paired_test"], r["bootstrap"]
    row = lambda label, key: (label, pct(o[key]), pct(n[key]), f"{(n[key] - o[key]) * 100:+.1f} pts")  # noqa: E731
    lines += [
        f"- Old checkpoint: `{r['models']['old']['path']}` (SHA-256 `{r['models']['old']['sha256'][:16]}...`)",
        f"- New checkpoint: `{r['models']['new']['path']}` (SHA-256 `{r['models']['new']['sha256'][:16]}...`)",
        f"- Same {r['image_manifest']['images']} images for both (manifest SHA-256 `{r['image_manifest']['sha256'][:16]}...`); "
        "identical class mapping and preprocessing.", "",
        "## Overall",
        _table(["Metric", "Old", "New", "Change"], [
            row("Accuracy", "accuracy"), row("Macro precision", "macro_precision"), row("Macro recall", "macro_recall"),
            row("Macro F1", "macro_f1"), row("Weighted F1", "weighted_f1")]),
        "", f"Correct: old {o['correct']} / new {n['correct']} of {o['total']}. Macro averages cover the "
            f"{o['macro_f1_classes']} classes with images.",
        f"Paired exact McNemar test: old right & new wrong {pt['old_right_new_wrong']}, old wrong & new right "
        f"{pt['old_wrong_new_right']}, p = {pt['p_value']:.2g}.",
        f"Bootstrap 95% CI ({bs['resamples']} resamples): accuracy difference "
        f"{bs['accuracy_difference_95ci'][0] * 100:+.1f} to {bs['accuracy_difference_95ci'][1] * 100:+.1f} pts; macro-F1 "
        f"difference {bs['macro_f1_difference_95ci'][0] * 100:+.1f} to {bs['macro_f1_difference_95ci'][1] * 100:+.1f} pts.",
        "", "## Limitations of this sample", _bullets(r["limitations"]), "",
        "## Per-class changes (classes with images)",
    ]
    fmt = lambda m: f"{pct(m['precision'])} / {pct(m['recall'])} / {pct(m['f1'])}"  # noqa: E731
    lines.append(_table(["Class", "Images", "Old P / R / F1", "New P / R / F1", "F1 change"], [
        (food(c["food"]), c["support"], fmt(c["old"]), fmt(c["new"]), f"{(c['new']['f1'] - c['old']['f1']) * 100:+.1f}")
        for c in sorted(r["per_class"], key=lambda c: c["new"]["f1"] - c["old"]["f1"])]))
    lines += ["", "## Most frequent confusions (either model)", _table(
        ["Actual → predicted", "Old", "New"],
        [(f"{food(c['actual'])} → {food(c['predicted'])}", c["old"], c["new"]) for c in r["frequent_confusions"]],
    ) if r["frequent_confusions"] else "None.", "",
        f"## Images whose prediction changed ({len(r['changed_predictions'])})", _table(
        ["Image", "Actual", "Old", "New"],
        [(f"`{c['path']}`", food(c["actual"]), food(c["old"]), food(c["new"])) for c in r["changed_predictions"]],
    ) if r["changed_predictions"] else "None.", "",
        "Full confusion matrices: `confusion_matrix_old.csv`, `confusion_matrix_new.csv` (rows actual, columns predicted)."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare two real-world evaluation reports on the same images.")
    parser.add_argument("--old", type=Path, default=DEFAULT_OLD)
    parser.add_argument("--new", type=Path, default=DEFAULT_NEW)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    report = compare(args.old, args.new)
    out = Path(args.output_dir)
    write_json(out / "comparison.json", report)
    if report["status"] == "ok":
        names = report["confusion_matrices"]["labels"]
        write_matrix(out / "confusion_matrix_old.csv", names, report["confusion_matrices"]["old"])
        write_matrix(out / "confusion_matrix_new.csv", names, report["confusion_matrices"]["new"])
    write_text(out / "comparison.md", markdown(report))
    print(f"Status: {report['status']}. Wrote {rel(out / 'comparison.md')}")
    for reason in report.get("reasons", []):
        print(f"  - {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
