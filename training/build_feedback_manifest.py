"""Builds a candidate training manifest from the v2 manifests plus APPROVED
intake rows, e.g. reviewed user feedback (human-in-the-loop, step 3 of 3).

    .venv-gpu\\Scripts\\python.exe -m training.build_feedback_manifest --batch <batch_id> [--batch ...] --name <name>

Writes data/manifests/<name>/{train,validation,test_clean}.csv and
summary.json. Nothing is trained; a candidate is trained separately with

    .venv-gpu\\Scripts\\python.exe -m training.train_efficientnet_v2 --manifest-dir data/manifests/<name> \\
        --output-dir models/<new folder>

Rules:
- Every batch is re-validated here with evaluation.intake.validate_batch.
  Only rows with status `ready` (a reviewer wrote accept / relabel:<class>
  and there are no errors) are added. Pending, rejected and erroring rows
  are listed in summary.json and left out. Unreviewed feedback never
  becomes training data.
- Approved images are added to TRAIN only. validation.csv and
  test_clean.csv are copied unchanged from the base manifests, so a
  candidate is selected and compared on the same independent images as v2
  and production.
- Final leakage check: no added image may have an exact (SHA-256) or near
  (dHash) copy in the base validation or test_clean manifests or in
  data/real_world_test/, and no image may be added twice.
- An existing output folder is never overwritten.
"""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

from evaluation.dataset_audit import (
    DEFAULT_CHECKPOINT,
    DEFAULT_NEAR_DISTANCE,
    DEFAULT_PROCESSED_DIR,
    DEFAULT_REAL_WORLD_DIR,
    ROOT,
    load_class_names,
    rel,
    utc_now,
)
from evaluation.hashing import build_reference_index, hamming_distances
from evaluation.intake import DEFAULT_INTAKE_DIR, HASH_CACHE, INTAKE_FILE, hash_folder, read_intake, validate_batch

from .build_manifests import FIELDS, read_manifest, write_manifest

BASE_MANIFEST_DIR = ROOT / "data" / "manifests" / "v2"
MANIFESTS_ROOT = ROOT / "data" / "manifests"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def held_out_hashes(base_dir, processed_dir, real_world_dir, cache_path):
    """(sha256 set, dHash array) of every image a candidate is evaluated on."""
    held_paths = {row["path"] for name in ("validation.csv", "test_clean.csv") for row in read_manifest(base_dir / name)}
    entries = [e for e in build_reference_index(processed_dir, None, cache_path, progress=lambda *_: None).entries
               if e["path"] in held_paths]
    if len(entries) != len(held_paths):
        raise ValueError("Some validation / test_clean images could not be found under the processed folder.")
    if real_world_dir and Path(real_world_dir).is_dir():
        entries += hash_folder(real_world_dir, "real_world_test", "*/*")
    return ({e["sha256"] for e in entries},
            np.array([e["dhash"] for e in entries if e["dhash"] is not None], dtype=np.uint64))


def build(batches, name, class_names, base_dir=BASE_MANIFEST_DIR, manifests_root=MANIFESTS_ROOT,
          intake_dir=DEFAULT_INTAKE_DIR, processed_dir=DEFAULT_PROCESSED_DIR, real_world_dir=DEFAULT_REAL_WORLD_DIR,
          cache_path=HASH_CACHE, near_distance=DEFAULT_NEAR_DISTANCE):
    base_dir, out = Path(base_dir), Path(manifests_root) / name
    if out.exists():
        raise FileExistsError(f"{out} already exists; choose a new --name.")
    class_index = {label: i for i, label in enumerate(class_names)}
    base_train = read_manifest(base_dir / "train.csv")
    base_shas = {row["sha256"] for row in base_train}

    added, left_out, batch_reports = [], [], {}
    for batch_id in batches:
        batch_dir = Path(intake_dir) / batch_id
        if not (batch_dir / INTAKE_FILE).is_file():
            raise FileNotFoundError(f"{batch_dir / INTAKE_FILE} not found.")
        report = validate_batch(batch_dir, class_names, processed_dir=processed_dir, real_world_dir=real_world_dir,
                                cache_path=cache_path, near_distance=near_distance)
        batch_reports[batch_id] = report["summary"]
        extra = {r["file"]: r for r in read_intake(batch_dir / INTAKE_FILE)[0]}
        for record in report["images"]:
            if record["status"] != "ready":
                left_out.append({"batch": batch_id, "file": record["file"], "status": record["status"],
                                 "errors": record["errors"]})
                continue
            source = extra.get(record["file"], {})
            added.append({
                "path": rel(batch_dir / "images" / record["file"]), "label": record["label"],
                "class_index": class_index[record["label"]], "sha256": record["sha256"],
                "dhash": int(record["dhash"], 16) if record["dhash"] else None,
                "batch": batch_id, "proposed_label": record["proposed_label"], "decision": record["decision"],
                "prediction_id": source.get("prediction_id", ""), "predicted_food": source.get("predicted_food", ""),
            })

    # Final leakage assertions, independent of the validator's findings.
    held_shas, held_dhash = held_out_hashes(base_dir, processed_dir, real_world_dir, cache_path)
    seen = set()
    for row in added:
        if row["sha256"] in held_shas:
            raise AssertionError(f"{row['path']} is an exact copy of a validation / test / real-world image.")
        if row["dhash"] is not None and len(held_dhash) and (hamming_distances(held_dhash, row["dhash"]) <= near_distance).any():
            raise AssertionError(f"{row['path']} is a near copy of a validation / test / real-world image.")
        if row["sha256"] in base_shas or row["sha256"] in seen:
            raise AssertionError(f"{row['path']} would be added to train twice.")
        seen.add(row["sha256"])
    if not added:
        return {"written": False, "reason": "No approved (ready) rows in the given batches.",
                "batches": batch_reports, "left_out": left_out}

    out.mkdir(parents=True)
    write_manifest(out / "train.csv", base_train + [{k: row[k] for k in FIELDS} for row in added])
    for split in ("validation.csv", "test_clean.csv"):
        shutil.copyfile(base_dir / split, out / split)
    corrections = sorted({(r["label"], r["predicted_food"]) for r in added
                          if r["predicted_food"] and r["predicted_food"] != r["label"]})
    summary = {
        "generated_at": utc_now(),
        "base_manifests": rel(base_dir),
        "base_sha256": {split: sha256(base_dir / split) for split in ("train.csv", "validation.csv", "test_clean.csv")},
        "train_base": len(base_train),
        "train_added": len(added),
        "validation_and_test_clean": "copied unchanged from the base manifests",
        "batches": batch_reports,
        "added": [{k: v for k, v in row.items() if k != "dhash"} for row in added],
        "left_out": left_out,
        "affected_classes": sorted({r["label"] for r in added} | {p for _, p in corrections}),
        # (actual, previously predicted) pairs, for checking the confusion they correct.
        "correction_pairs": [list(pair) for pair in corrections],
        "near_duplicate_max_distance": near_distance,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"written": True, "output": rel(out), **summary}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build a candidate manifest = base manifests + approved intake rows.")
    parser.add_argument("--batch", action="append", required=True, help="Intake batch id (repeatable).")
    parser.add_argument("--name", required=True, help="Output folder under data/manifests/.")
    parser.add_argument("--base", type=Path, default=BASE_MANIFEST_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="Source of the class list.")
    args = parser.parse_args(argv)
    result = build(args.batch, args.name, load_class_names(args.checkpoint), base_dir=args.base)
    if not result["written"]:
        print(f"Nothing written: {result['reason']} {len(result['left_out'])} row(s) not ready.")
        return 1
    print(f"Wrote {result['output']}: {result['train_base']} base + {result['train_added']} approved train rows; "
          f"validation and test_clean unchanged. Affected classes: {', '.join(result['affected_classes'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
