"""Builds leakage-free dataset manifests for retraining (Phase B).

    .venv-gpu\\Scripts\\python.exe -m training.build_manifests

Writes data/manifests/v2/{train,validation,test_clean}.csv and summary.json.
Images are only listed, never moved, renamed, copied or modified.

Rules:
- train: every processed train image EXCEPT
  1. images with an exact (SHA-256) or near (dHash <= 3) copy in the
     validation or test split, so neither can leak into training;
  2. extra copies of an exact duplicate inside train (the first path in
     sorted order is kept);
  3. all copies of an exact duplicate inside train that carry different
     labels (contradictory supervision; listed for manual review, not
     relabelled).
- validation: the full validation split (clean with respect to the new
  train manifest by rule 1).
- test_clean: the 3,304 leakage-free test images identified by
  evaluation/class_audit.py (no exact or near copy in train or validation).

Labels are the folder names; class indices follow the production
checkpoint's class_names.
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from evaluation.dataset_audit import DEFAULT_CHECKPOINT, DEFAULT_NEAR_DISTANCE, DEFAULT_PROCESSED_DIR, ROOT
from evaluation.hashing import build_reference_index, hamming_distances

OUTPUT_DIR = ROOT / "data" / "manifests" / "v2"
AUDIT_REPORT = ROOT / "reports" / "phase1_audit" / "class_audit.json"
HASH_CACHE = ROOT / "reports" / "real_world" / "cache" / "reference_index.json"
FIELDS = ["path", "label", "class_index", "sha256"]


def write_manifest(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_manifest(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return [{**row, "class_index": int(row["class_index"])} for row in csv.DictReader(handle)]


def build(processed_dir=DEFAULT_PROCESSED_DIR, output_dir=OUTPUT_DIR, near_distance=DEFAULT_NEAR_DISTANCE,
          audit_report=AUDIT_REPORT, checkpoint=DEFAULT_CHECKPOINT, cache_path=HASH_CACHE):
    class_names = torch.load(checkpoint, map_location="cpu", weights_only=True)["class_names"]
    class_index = {name: i for i, name in enumerate(class_names)}
    reference = build_reference_index(processed_dir, None, cache_path, progress=lambda *_: None)
    entries = [e for e in reference.entries if e["source"].startswith("processed/")]
    for e in entries:
        e["split"] = e["source"].split("/", 1)[1]
        if Path(e["path"]).parent.name != e["label"]:
            raise ValueError(f"Label does not match folder: {e['path']}")
        if e["label"] not in class_index:
            raise ValueError(f"Unknown class folder: {e['path']}")

    train = sorted((e for e in entries if e["split"] == "train"), key=lambda e: e["path"])
    held_out = [e for e in entries if e["split"] in ("validation", "test")]
    held_out_shas = {e["sha256"] for e in held_out}
    held_out_hashes = np.array([e["dhash"] for e in held_out if e["dhash"] is not None], dtype=np.uint64)

    excluded = defaultdict(list)
    candidates = []
    for e in train:
        if e["sha256"] in held_out_shas:
            excluded["exact_copy_in_validation_or_test"].append(e["path"])
        elif e["dhash"] is not None and (hamming_distances(held_out_hashes, e["dhash"]) <= near_distance).any():
            excluded["near_copy_in_validation_or_test"].append(e["path"])
        else:
            candidates.append(e)

    by_sha = defaultdict(list)
    for e in candidates:
        by_sha[e["sha256"]].append(e)
    kept = []
    for members in by_sha.values():
        if len({m["label"] for m in members}) > 1:
            excluded["conflicting_labels_exact_duplicate_in_train"] += [m["path"] for m in members]
            continue
        kept.append(members[0])
        excluded["extra_exact_copy_in_train"] += [m["path"] for m in members[1:]]
    kept.sort(key=lambda e: e["path"])

    def rows(items):
        return [{"path": e["path"], "label": e["label"], "class_index": class_index[e["label"]], "sha256": e["sha256"]}
                for e in items]

    validation = sorted((e for e in entries if e["split"] == "validation"), key=lambda e: e["path"])
    clean_paths = set(json.loads(Path(audit_report).read_text(encoding="utf-8"))["leakage_free_test_paths"])
    test_clean = sorted((e for e in entries if e["split"] == "test" and e["path"] in clean_paths), key=lambda e: e["path"])
    if len(test_clean) != len(clean_paths):
        raise ValueError("Some leakage-free test paths from the audit no longer exist.")

    # Final leakage assertions on the written manifests.
    train_shas = {e["sha256"] for e in kept}
    train_hashes = np.array([e["dhash"] for e in kept if e["dhash"] is not None], dtype=np.uint64)
    for split_name, items in (("validation", validation), ("test_clean", test_clean)):
        for e in items:
            if e["sha256"] in train_shas or (
                    e["dhash"] is not None and (hamming_distances(train_hashes, e["dhash"]) <= near_distance).any()):
                raise AssertionError(f"{split_name} image {e['path']} has a copy in the train manifest")

    output_dir = Path(output_dir)
    write_manifest(output_dir / "train.csv", rows(kept))
    write_manifest(output_dir / "validation.csv", rows(validation))
    write_manifest(output_dir / "test_clean.csv", rows(test_clean))
    per_class = {name: {"train_original": 0, "train_manifest": 0, "validation": 0, "test_clean": 0} for name in class_names}
    for e in train:
        per_class[e["label"]]["train_original"] += 1
    for key, items in (("train_manifest", kept), ("validation", validation), ("test_clean", test_clean)):
        for e in items:
            per_class[e["label"]][key] += 1
    summary = {
        "class_names": class_names,
        "near_duplicate_max_distance": near_distance,
        "train_original": len(train),
        "train_manifest": len(kept),
        "validation": len(validation),
        "test_clean": len(test_clean),
        "excluded_from_train": {reason: len(paths) for reason, paths in excluded.items()},
        "excluded_paths": dict(excluded),
        "per_class": per_class,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    summary = build()
    print(json.dumps({k: v for k, v in summary.items() if k not in ("excluded_paths", "per_class", "class_names")}, indent=2))


if __name__ == "__main__":
    main()
