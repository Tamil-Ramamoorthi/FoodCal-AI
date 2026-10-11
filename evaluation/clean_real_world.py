"""Builds a cleaned real-world evaluation manifest and re-scores both models on
it, WITHOUT touching any image, label or existing report.

    .venv-gpu\\Scripts\\python.exe -m evaluation.clean_real_world

Reads reports/real_world/ (production) and reports/real_world_v2/ (v2)
real_world_evaluation.json plus the manual-review sheet
data/real_world_review/manual_review.csv, and writes only to
reports/real_world_clean/:

- clean_manifest.csv: the images kept (path, label = class folder, SHA-256)
- exclusions.csv: every image left out, with its reason and who decided
- clean_evaluation.json / .md and both confusion matrices: both models'
  metrics on the kept images, with the paired McNemar test and bootstrap CIs

The predictions are the ones already stored in the two reports (same
checkpoints, same preprocessing); nothing is re-run. Exclusion rules:

1. Exact copies with the same label: one copy is kept (the file name
   without a " (n)" suffix, else the first by name), the others excluded.
2. Exact copies in DIFFERENT class folders: all are excluded, pending,
   until a reviewer either marks exactly one of them `keep` or marks every
   copy `exclude` (e.g. a mixed dish no single label fits). The label is
   never chosen automatically.
3. Possible overlaps with training/validation/test data reported by the
   evaluation: excluded unless a reviewer marks the image `keep`.
4. Any other row in the review sheet: the reviewer's `decision`, or, while
   that is blank, the audit's `default_action`.

The result is `provisional` while any flagged image still lacks a reviewer
decision, and `reviewed` once every one has one.
"""

import argparse
import csv
import re
import sys
from pathlib import Path

from .compare_real_world import DEFAULT_NEW, DEFAULT_OLD, check_same_images, compare_loaded, load_report, markdown
from .compare_models import write_matrix
from .dataset_audit import ROOT, rel
from .evaluate_real_world import image_manifest
from .hashing import sha256_file
from .reports import _bullets, _table, food, write_json, write_text

DEFAULT_REVIEW = ROOT / "data" / "real_world_review" / "manual_review.csv"
DEFAULT_OUTPUT = ROOT / "reports" / "real_world_clean"
REVIEW_COLUMNS = ("path", "issue", "related", "audit_finding", "default_action", "decision", "reviewer", "notes")
ACTIONS = ("keep", "exclude")
COPY_SUFFIX = re.compile(r" \(\d+\)(?=\.[^.]+$)")


def read_review(path, known_paths):
    """Rows of the manual-review sheet, keyed by image path. Raises ValueError on bad input."""
    path = Path(path)
    if not path.is_file():
        return {}
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in ("path", "default_action", "decision", "reviewer") if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{rel(path)} is missing column(s): {', '.join(missing)}.")
        rows = {}
        for line, row in enumerate(reader, 2):
            row = {k: (v or "").strip() for k, v in row.items() if k}
            where = f"{rel(path)} line {line}"
            if row["path"] not in known_paths:
                raise ValueError(f"{where}: {row['path']!r} is not an evaluated image.")
            if row["path"] in rows:
                raise ValueError(f"{where}: {row['path']!r} is listed twice.")
            if row["default_action"] not in ACTIONS:
                raise ValueError(f"{where}: default_action must be keep or exclude.")
            if row["decision"] and row["decision"] not in ACTIONS:
                raise ValueError(f"{where}: decision must be blank, keep or exclude.")
            if row["decision"] and not row["reviewer"]:
                raise ValueError(f"{where}: a decision needs a reviewer.")
            rows[row["path"]] = row
    return rows


def check_files_unchanged(rows, root=ROOT):
    """Every image must still exist with the SHA-256 recorded in the reports."""
    problems = []
    for path, _, sha in rows:
        file = Path(root) / path
        if not file.is_file():
            problems.append(f"{path} no longer exists.")
        elif sha256_file(file) != sha:
            problems.append(f"{path} changed since it was evaluated.")
    if problems:
        raise ValueError("Re-run both evaluations first: " + " ".join(problems[:5]))


def build_manifest(report, review):
    """Returns (kept paths, exclusions, pending review notes)."""
    labels = {p["path"]: p["actual"] for p in report["predictions"]}
    quality = report["quality"]
    excluded, pending = {}, []

    def decision(path):
        return review.get(path, {}).get("decision", "")

    def exclude(path, reason, decided_by):
        excluded.setdefault(path, {"path": path, "label": labels[path], "reason": reason, "decided_by": decided_by})

    for group in quality.get("exact_duplicates_within", []):
        paths = sorted(group["paths"])
        if group.get("conflicting_labels"):
            kept = [p for p in paths if decision(p) == "keep"]
            if len(kept) == 1:
                for p in paths:
                    if p != kept[0]:
                        exclude(p, f"Same file as {kept[0]} in another class folder; reviewer kept that copy.", "reviewer")
            elif all(decision(p) == "exclude" for p in paths):
                for p in paths:
                    exclude(p, "Same file is in several class folders (" + ", ".join(labels[q] for q in paths)
                            + "); reviewer excluded every copy.", "reviewer")
            else:
                pending.append("Conflicting labels for one file (" + ", ".join(paths) + "): mark exactly one copy "
                               "`keep`, or every copy `exclude`.")
                for p in paths:
                    exclude(p, "Same file is in several class folders (" + ", ".join(labels[q] for q in paths)
                            + "); label not decided.", "rule: pending review")
            continue
        canonical = sorted(paths, key=lambda p: (bool(COPY_SUFFIX.search(p)), p))[0]
        for p in paths:
            if p != canonical:
                exclude(p, f"Exact copy of {canonical}.", "rule")

    overlaps = {}
    for entry in quality.get("exact_overlaps", []) + quality.get("near_duplicate_overlaps", []):
        overlaps.setdefault(entry["path"], []).extend(
            f"{m['source']} {m['path']} (distance {m.get('distance', 0)})" for m in entry["matches"])
    for path, matches in overlaps.items():
        if decision(path) == "keep":
            continue
        if decision(path) == "exclude":
            exclude(path, "Overlaps model data: " + "; ".join(matches), "reviewer")
        else:
            pending.append(f"{path}: possible overlap with model data, needs a reviewer decision.")
            exclude(path, "Possible overlap with model data: " + "; ".join(matches), "rule: pending review")

    for path, row in review.items():
        if path in excluded or path in overlaps:
            continue
        reason = row.get("audit_finding") or row.get("issue") or "Excluded in manual review."
        if row["decision"]:
            if row["decision"] == "exclude":
                exclude(path, reason, "reviewer")
        else:
            pending.append(f"{path}: {row.get('issue') or 'flagged'}; default `{row['default_action']}` until reviewed.")
            if row["default_action"] == "exclude":
                exclude(path, reason, "audit default: pending review")

    kept = sorted(p for p in labels if p not in excluded)
    return kept, sorted(excluded.values(), key=lambda e: e["path"]), pending


def filtered(report, kept):
    keep = set(kept)
    predictions = [p for p in report["predictions"] if p["path"] in keep]
    return {**report, "predictions": predictions, "image_manifest": image_manifest(predictions)}


def clean(old_path=DEFAULT_OLD, new_path=DEFAULT_NEW, review_path=DEFAULT_REVIEW, root=ROOT):
    old, old_missing = load_report(old_path)
    new, new_missing = load_report(new_path)
    if old_missing or new_missing:
        raise ValueError(old_missing or new_missing)
    rows = check_same_images(old, new)
    check_files_unchanged(rows, root)
    review = read_review(review_path, {r[0] for r in rows})
    kept, exclusions, pending = build_manifest(old, review)
    if not kept:
        raise ValueError("No images are left after cleaning.")
    old_c, new_c = filtered(old, kept), filtered(new, kept)
    clean_rows = check_same_images(old_c, new_c)
    result = compare_loaded(old_c, new_c, old_path, new_path, clean_rows)

    per_class = {}
    for p in old_c["predictions"]:
        per_class[p["actual"]] = per_class.get(p["actual"], 0) + 1
    names = [c["name"] for c in old["class_mapping"]]
    minimum = old["quality"].get("min_images_per_class", 10)
    below = [n for n in names if per_class.get(n, 0) < minimum]
    provenance = old["quality"].get("provenance", {})
    result["limitations"] = [
        f"{len(kept)} of {len(rows)} images kept; {len(exclusions)} excluded (see exclusions.csv).",
        f"{len(below)} of {len(names)} classes have fewer than {minimum} images; their per-class metrics are unreliable.",
        "Predictions are reused from the two original reports (same checkpoints and preprocessing); no model was re-run.",
        "Duplicate checks cannot find crops, re-edits or different photos of the same plate; several photos come "
        "from the same photographer and shoot, so images are not fully independent.",
        "Confidence intervals are wide for this sample; a difference inside them is not evidence that either model is better.",
    ]
    if not provenance.get("present") or provenance.get("undocumented_images"):
        result["limitations"].append(f"Provenance: {provenance.get('undocumented_images', '?')} image(s) have no "
                                     "documented source or licence (data/real_world_test/provenance.csv).")
    result["report_type"] = "real_world_clean_evaluation"
    result["manifest_status"] = "provisional" if pending else "reviewed"
    result["review_file"] = rel(review_path)
    result["source_manifest"] = image_manifest(old["predictions"])
    result["exclusions"] = exclusions
    result["pending_review"] = pending
    result["kept"] = [{"path": p["path"], "label": p["actual"], "sha256": p["sha256"]}
                      for p in sorted(old_c["predictions"], key=lambda p: p["path"])]
    return result


def clean_markdown(r):
    body = markdown(r).split("\n", 1)[1]  # drop the comparison's own title
    head = [
        "# Cleaned real-world evaluation", "",
        f"Manifest status: **{r['manifest_status']}**"
        + (" (some flagged images still need a reviewer decision; numbers may change)." if r["pending_review"] else "."),
        f"- Review sheet: `{r['review_file']}`",
        f"- Source: {r['source_manifest']['images']} images (manifest SHA-256 `{r['source_manifest']['sha256'][:16]}...`); "
        f"kept {r['image_manifest']['images']}, excluded {len(r['exclusions'])}.",
        "- No image, label, checkpoint or original report was changed.", "",
        "## Still needs manual review", _bullets(r["pending_review"]), "",
        f"## Excluded images ({len(r['exclusions'])})",
        _table(["Image", "Label", "Reason", "Decided by"],
               [(f"`{e['path']}`", food(e["label"]), e["reason"], e["decided_by"]) for e in r["exclusions"]]), "",
    ]
    return "\n".join(head) + body


def write_csv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Build a cleaned real-world manifest from the review sheet and score both models on it. "
                    "Read-only for images, labels and the original reports.")
    parser.add_argument("--old", type=Path, default=DEFAULT_OLD, help="production report (default: %(default)s)")
    parser.add_argument("--new", type=Path, default=DEFAULT_NEW, help="candidate report (default: %(default)s)")
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW, help="manual-review CSV (default: %(default)s)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="default: %(default)s")
    args = parser.parse_args(argv)
    out = Path(args.output_dir).resolve()
    protected = {Path(p).resolve().parent for p in (args.old, args.new)}
    if out in protected or out == (ROOT / "data").resolve() or (ROOT / "data").resolve() in out.parents:
        parser.error("--output-dir must be separate from the original reports and from data/.")
    try:
        result = clean(args.old, args.new, args.review)
    except (ValueError, FileNotFoundError) as error:
        print(f"Cannot build the cleaned evaluation: {error}")
        return 1
    write_csv(out / "clean_manifest.csv", ("path", "label", "sha256"), result["kept"])
    write_csv(out / "exclusions.csv", ("path", "label", "reason", "decided_by"), result["exclusions"])
    write_json(out / "clean_evaluation.json", result)
    names = result["confusion_matrices"]["labels"]
    write_matrix(out / "confusion_matrix_old.csv", names, result["confusion_matrices"]["old"])
    write_matrix(out / "confusion_matrix_new.csv", names, result["confusion_matrices"]["new"])
    write_text(out / "clean_evaluation.md", clean_markdown(result))
    o, n = result["overall"]["old"], result["overall"]["new"]
    print(f"Manifest {result['manifest_status']}: kept {o['total']}, excluded {len(result['exclusions'])}. "
          f"Accuracy {o['accuracy']:.1%} -> {n['accuracy']:.1%}; macro F1 {o['macro_f1']:.1%} -> {n['macro_f1']:.1%}.")
    for note in result["pending_review"]:
        print(f"  - review: {note}")
    print(f"Wrote {rel(out / 'clean_evaluation.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
