"""Dataset audit: processed splits and the real-world test set.

Read-only: images are never moved, renamed or deleted. Run from the
project root:

    .venv-gpu\\Scripts\\python.exe -m evaluation.dataset_audit

Writes reports/real_world/dataset_audit.json and dataset_audit.md.
"""

import argparse
import csv
import glob
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .hashing import (
    SUPPORTED_EXTENSIONS,
    build_reference_index,
    dhash,
    hamming_distances,
    sha256_file,
)
from .reports import audit_markdown, write_json, write_text

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REAL_WORLD_DIR = ROOT / "data" / "real_world_test"
DEFAULT_PROCESSED_DIR = ROOT / "data" / "processed"
DEFAULT_CHECKPOINT = ROOT / "models" / "efficientnet_b0" / "food_efficientnet_b0_best.pth"
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "real_world"
DEFAULT_MIN_PER_CLASS = 10
# Measured on this dataset by evaluation/calibrate_dhash.py (1,000 sampled
# originals, seed 0): distance <= 3 matched 99.9% of copies re-created the
# way prepare_dataset.py made them and 99.7% of re-saved copies (<=1280px,
# JPEG q80), while 0.38% of processed images had a different-label image
# that close. At distance 6 that rose to 2.99%.
DEFAULT_NEAR_DISTANCE = 3
PROVENANCE_FILE = "provenance.csv"
PROVENANCE_REQUIRED_COLUMNS = ("file", "source")
IGNORED_ROOT_FILES = {PROVENANCE_FILE, "README.md"}
SPLITS = ("train", "validation", "test")

DUPLICATE_DETECTION_LIMITS = (
    "Exact duplicates are found by SHA-256 of file bytes, so any re-encoding, "
    "resizing, cropping or metadata change hides a copy. Possible near-duplicates "
    "are found with a 64-bit difference hash (Hamming distance <= {d}); this can miss "
    "crops, rotations, mirrored or heavily edited copies and can occasionally match "
    "different but similar-looking photos, so matches need manual review."
)


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def rel(path):
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def find_source_archive():
    pattern = os.path.join(
        Path.home(), ".cache", "huggingface", "hub",
        "datasets--ewanlong--Food_Classification_Dataset", "snapshots", "*",
        "Food_Classification_Dataset.zip",
    )
    matches = sorted(glob.glob(pattern))
    return Path(matches[0]) if matches else None


def resolve_archive(option):
    if option in (None, "", "none"):
        return None
    if option == "auto":
        return find_source_archive()
    path = Path(option)
    if not path.is_file():
        raise FileNotFoundError(f"Source archive not found: {path}")
    return path


def load_class_names(checkpoint_path):
    import torch  # only needed here; keeps the rest importable without torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    return list(checkpoint["class_names"])


# ---------------------------------------------------------------------
# Real-world folder
# ---------------------------------------------------------------------

def scan_real_world(root, class_names):
    root = Path(root)
    result = {
        "path": rel(root),
        "exists": root.is_dir(),
        "images": [],
        "invalid_images": [],
        "unsupported_files": [],
        "unexpected_entries": [],
        "missing_class_folders": [],
    }
    if not root.is_dir():
        result["missing_class_folders"] = list(class_names)
        return result

    known = set(class_names)
    lowered = {name.lower(): name for name in class_names}
    for entry in sorted(root.iterdir()):
        if entry.is_dir() and entry.name not in known:
            hint = lowered.get(entry.name.lower())
            result["unexpected_entries"].append({
                "path": rel(entry), "type": "folder",
                "note": (f"Not a model class; ignored. Folder names are case-sensitive: rename it to '{hint}'."
                         if hint else "Not a model class; ignored."),
            })
        elif entry.is_file() and entry.name not in IGNORED_ROOT_FILES:
            result["unexpected_entries"].append({
                "path": rel(entry), "type": "file",
                "note": "Images must be inside a class folder; ignored.",
            })

    # Exact on-disk names only. On Windows `root / label` would also match a
    # folder whose name differs only in case, which is reported as ignored above.
    exact_folders = {entry.name for entry in root.iterdir() if entry.is_dir()}
    for label in class_names:
        folder = root / label
        if label not in exact_folders:
            result["missing_class_folders"].append(label)
            continue
        for path in sorted(folder.iterdir()):
            if path.is_dir():
                result["unexpected_entries"].append({
                    "path": rel(path), "type": "folder",
                    "note": "Nested folders are not scanned; ignored.",
                })
                continue
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                result["unsupported_files"].append({"path": rel(path), "label": label})
                continue
            try:
                with Image.open(path) as image:
                    image_format = image.format
                    oriented = ImageOps.exif_transpose(image)
                    oriented.load()  # full decode catches truncated files
                    record = {
                        "path": rel(path),
                        "relative_path": path.relative_to(root).as_posix(),
                        "label": label,
                        "sha256": sha256_file(path),
                        "dhash": dhash(oriented),
                        "width": oriented.width,
                        "height": oriented.height,
                        "format": image_format,
                    }
            except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as error:
                result["invalid_images"].append({
                    "path": rel(path), "label": label,
                    "error": f"{type(error).__name__}: {error}",
                })
                continue
            if image_format not in ("JPEG", "PNG", "WEBP"):
                record["format_warning"] = f"File content is {image_format}, not JPEG/PNG/WEBP."
            result["images"].append(record)
    return result


def read_provenance(root, images):
    path = Path(root) / PROVENANCE_FILE
    image_paths = {image["relative_path"] for image in images}
    info = {
        "file": rel(path),
        "present": path.is_file(),
        "required_columns": list(PROVENANCE_REQUIRED_COLUMNS),
        "missing_columns": [],
        "documented_images": 0,
        "undocumented_images": len(image_paths),
        "entries_without_image": [],
    }
    if not path.is_file():
        return info

    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = [c.strip().lower() for c in (reader.fieldnames or [])]
        info["missing_columns"] = [c for c in PROVENANCE_REQUIRED_COLUMNS if c not in columns]
        rows = [{(k or "").strip().lower(): (v or "").strip() for k, v in row.items()} for row in reader]

    documented = set()
    for row in rows:
        name = row.get("file", "").replace("\\", "/").lstrip("./")
        if not name:
            continue
        if name in image_paths and row.get("source"):
            documented.add(name)
        elif name not in image_paths:
            info["entries_without_image"].append(name)
    info["documented_images"] = len(documented)
    info["undocumented_images"] = len(image_paths - documented)
    return info


def duplicates_within(images, near_distance):
    groups = {}
    for image in images:
        groups.setdefault(image["sha256"], []).append(image)
    exact = [
        {
            "sha256": sha,
            "paths": [i["path"] for i in members],
            "labels": sorted({i["label"] for i in members}),
            "conflicting_labels": len({i["label"] for i in members}) > 1,
        }
        for sha, members in groups.items() if len(members) > 1
    ]

    near = []
    if len(images) > 1:
        hashes = np.array([i["dhash"] for i in images], dtype=np.uint64)
        for a in range(len(images) - 1):
            distances = hamming_distances(hashes[a + 1:], images[a]["dhash"])
            for offset in np.flatnonzero(distances <= near_distance):
                b = a + 1 + int(offset)
                if images[a]["sha256"] != images[b]["sha256"]:
                    near.append({
                        "paths": [images[a]["path"], images[b]["path"]],
                        "labels": [images[a]["label"], images[b]["label"]],
                        "distance": int(distances[offset]),
                    })
    return exact, near


def _summarize_match(match):
    return {key: match[key] for key in ("source", "path", "label") if key in match} | (
        {"distance": match["distance"]} if "distance" in match else {}
    )


def check_overlaps(images, reference, near_distance):
    exact, near = [], []
    for image in images:
        hits = reference.exact_matches(image["sha256"])
        if hits:
            exact.append({
                "path": image["path"], "label": image["label"],
                "match_count": len(hits),
                "matches": [_summarize_match(h) for h in hits[:5]],
            })
            continue
        close = reference.near_matches(image["dhash"], near_distance)
        if close:
            near.append({
                "path": image["path"], "label": image["label"],
                "matches": [_summarize_match(h) for h in close],
            })
    return exact, near


def independence_assessment(images, exact_overlaps, near_overlaps, reference, provenance):
    if not images:
        return "not_assessed", "There are no valid real-world images to assess."
    if reference is None:
        return "overlap_not_checked", "Overlap with the training data was not checked."
    if exact_overlaps or near_overlaps:
        return "overlap_detected", (
            "Some images are identical or possibly near-identical to images the model "
            "was built from. Those images are not independent."
        )
    if not provenance["present"] or provenance["undocumented_images"]:
        return "provenance_incomplete", (
            "No overlaps were detected, but the source of every image has not been "
            "documented in provenance.csv, so independence cannot be confirmed."
        )
    return "checks_passed", (
        "No exact or near-duplicate overlaps were detected and every image has a "
        "documented source. Independence still depends on that documentation being "
        "accurate and on the limits of duplicate detection."
    )


def audit_real_world(real_dir, class_names, reference, min_per_class=DEFAULT_MIN_PER_CLASS,
                     near_distance=DEFAULT_NEAR_DISTANCE):
    scan = scan_real_world(real_dir, class_names)
    images = scan["images"]

    per_class = {name: 0 for name in class_names}
    for image in images:
        per_class[image["label"]] += 1

    exact_within, near_within = duplicates_within(images, near_distance)
    if reference is not None and images:
        exact_overlaps, near_overlaps = check_overlaps(images, reference, near_distance)
    else:
        exact_overlaps, near_overlaps = [], []
    provenance = read_provenance(real_dir, images)
    status, status_text = independence_assessment(
        images, exact_overlaps, near_overlaps, reference, provenance,
    )

    without = [name for name in class_names if per_class[name] == 0]
    below = [name for name in class_names if 0 < per_class[name] < min_per_class]

    warnings = []
    if not scan["exists"]:
        warnings.append(f"The folder {scan['path']} does not exist.")
    if not images:
        warnings.append("No valid real-world images were found, so no real-world metrics can be calculated.")
    if scan["invalid_images"]:
        warnings.append(f"{len(scan['invalid_images'])} file(s) could not be read as images and were excluded.")
    if scan["unsupported_files"]:
        warnings.append(f"{len(scan['unsupported_files'])} file(s) have unsupported extensions (use JPG, JPEG, PNG or WEBP) and were ignored.")
    if scan["unexpected_entries"]:
        warnings.append(f"{len(scan['unexpected_entries'])} unexpected file(s) or folder(s) were ignored.")
    if scan["missing_class_folders"] and scan["exists"]:
        warnings.append(f"{len(scan['missing_class_folders'])} class folder(s) are missing.")
    if images and without:
        warnings.append(f"{len(without)} of {len(class_names)} classes have no images; their recall and F1 cannot be measured.")
    if below:
        warnings.append(f"{len(below)} class(es) have fewer than {min_per_class} images, so their per-class metrics are unreliable.")
    if exact_within:
        extra = sum(len(g['paths']) - 1 for g in exact_within)
        conflicts = sum(1 for g in exact_within if g["conflicting_labels"])
        warnings.append(f"{len(exact_within)} group(s) of exact duplicate images within the real-world set ({extra} extra cop{'y' if extra == 1 else 'ies'}; {conflicts} with conflicting labels).")
    if near_within:
        warnings.append(f"{len(near_within)} pair(s) of possible near-duplicate images within the real-world set; review manually.")
    if exact_overlaps:
        warnings.append(f"{len(exact_overlaps)} real-world image(s) are exact copies of images in the training data sources; they are not independent.")
    if near_overlaps:
        warnings.append(f"{len(near_overlaps)} real-world image(s) are possible near-duplicates of training data sources; review manually.")
    if images and reference is None:
        warnings.append("Overlap with the training data was not checked.")
    if images and not provenance["present"]:
        warnings.append(f"No {PROVENANCE_FILE} found; the source of each image is undocumented.")
    elif images and provenance["undocumented_images"]:
        warnings.append(f"{provenance['undocumented_images']} image(s) have no documented source in {PROVENANCE_FILE}.")
    if provenance["missing_columns"]:
        warnings.append(f"{PROVENANCE_FILE} is missing required column(s): {', '.join(provenance['missing_columns'])}.")

    return {
        "path": scan["path"],
        "exists": scan["exists"],
        "valid_images": len(images),
        "invalid_images": scan["invalid_images"],
        "unsupported_files": scan["unsupported_files"],
        "unexpected_entries": scan["unexpected_entries"],
        "missing_class_folders": scan["missing_class_folders"],
        "images_per_class": per_class,
        "classes_without_images": without,
        "classes_below_minimum": below,
        "min_images_per_class": min_per_class,
        "exact_duplicates_within": exact_within,
        "near_duplicates_within": near_within,
        "exact_overlaps": exact_overlaps,
        "near_duplicate_overlaps": near_overlaps,
        "overlap_reference": reference.sources() if reference is not None else None,
        "near_duplicate_max_distance": near_distance,
        "provenance": provenance,
        "independence_status": status,
        "independence_note": status_text,
        "duplicate_detection_limits": DUPLICATE_DETECTION_LIMITS.format(d=near_distance),
        "warnings": warnings,
        "images": [
            {**{k: v for k, v in image.items() if k != "dhash"}, "dhash": f"{image['dhash']:016x}"}
            for image in images
        ],
        "_records": images,  # in-memory only; removed before writing reports
    }


# ---------------------------------------------------------------------
# Processed train / validation / test splits
# ---------------------------------------------------------------------

def audit_processed(processed_dir, reference, class_names, near_distance=DEFAULT_NEAR_DISTANCE, examples=20):
    processed_dir = Path(processed_dir)
    entries = [e for e in reference.entries if e["source"].startswith("processed/")]

    counts, folder_issues = {}, {}
    for split in SPLITS:
        split_dir = processed_dir / split
        folders = sorted(p.name for p in split_dir.iterdir() if p.is_dir()) if split_dir.is_dir() else []
        folder_issues[split] = {
            "missing_class_folders": [c for c in class_names if c not in folders],
            "unexpected_folders": [f for f in folders if f not in class_names],
            "empty_class_folders": [f for f in folders if not any((split_dir / f).iterdir())],
        }
        counts[split] = {c: 0 for c in class_names}
    for entry in entries:
        split = entry["source"].split("/", 1)[1]
        counts[split][entry["label"]] = counts[split].get(entry["label"], 0) + 1

    groups = {}
    for entry in entries:
        groups.setdefault(entry["sha256"], []).append(entry)
    duplicates = [members for members in groups.values() if len(members) > 1]
    cross_split = [m for m in duplicates if len({e["source"] for e in m}) > 1]
    conflicting = [m for m in duplicates if len({e["label"] for e in m}) > 1]

    def exact_in_train(split):
        return sum(
            1 for e in entries
            if e["source"] == f"processed/{split}"
            and any(o["source"] == "processed/train" for o in groups[e["sha256"]])
        )

    train_hashes = np.array(
        [e["dhash"] for e in entries if e["source"] == "processed/train" and e["dhash"] is not None],
        dtype=np.uint64,
    )
    train_shas = {e["sha256"] for e in entries if e["source"] == "processed/train"}

    def near_in_train(split):
        count = 0
        for e in entries:
            if e["source"] != f"processed/{split}" or e["dhash"] is None or e["sha256"] in train_shas:
                continue
            if (hamming_distances(train_hashes, e["dhash"]) <= near_distance).any():
                count += 1
        return count

    split_totals = {split: sum(counts[split].values()) for split in SPLITS}

    def describe(members):
        return [{"path": e["path"], "label": e["label"]} for e in members]

    return {
        "path": rel(processed_dir),
        "images_per_split": split_totals,
        "images_per_class": counts,
        "folder_issues": folder_issues,
        "exact_duplicate_groups": len(duplicates),
        "exact_duplicate_extra_copies": sum(len(m) - 1 for m in duplicates),
        "cross_split_duplicate_groups": len(cross_split),
        "conflicting_label_groups": len(conflicting),
        "conflicting_label_examples": [describe(m) for m in conflicting[:examples]],
        "cross_split_examples": [describe(m) for m in cross_split[:examples]],
        "validation_images_with_exact_copy_in_train": exact_in_train("validation"),
        "test_images_with_exact_copy_in_train": exact_in_train("test"),
        "validation_images_with_possible_near_duplicate_in_train": near_in_train("validation"),
        "test_images_with_possible_near_duplicate_in_train": near_in_train("test"),
        "near_duplicate_max_distance": near_distance,
    }


def audit_archive_classes(reference, class_names):
    labels = {}
    for e in reference.entries:
        if e["source"] == "source-archive":
            labels[e["label"]] = labels.get(e["label"], 0) + 1
    if not labels:
        return None
    return {
        "images": sum(labels.values()),
        "class_folders": len(labels),
        "images_per_class": dict(sorted(labels.items())),
        "folders_not_in_model": sorted(set(labels) - set(class_names)),
        "model_classes_not_in_archive": sorted(set(class_names) - set(labels)),
    }


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def strip_private(audit):
    return {k: v for k, v in audit.items() if not k.startswith("_")}


def add_common_arguments(parser):
    parser.add_argument("--real-world-dir", type=Path, default=DEFAULT_REAL_WORLD_DIR,
                        help="Folder with one sub-folder per class (default: %(default)s).")
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                        help="Model checkpoint to evaluate (default: the production model, %(default)s).")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Report folder (default: %(default)s, the dashboard's production report; "
                             "other checkpoints need their own folder).")
    parser.add_argument("--source-archive", default="auto",
                        help="'auto' (Hugging Face cache), 'none', or a path to the source ZIP.")
    parser.add_argument("--min-per-class", type=int, default=DEFAULT_MIN_PER_CLASS)
    parser.add_argument("--near-distance", type=int, default=DEFAULT_NEAR_DISTANCE)
    parser.add_argument("--no-cache", action="store_true", help="Do not read or write the hash cache.")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Audit the processed splits and the real-world test set.")
    add_common_arguments(parser)
    args = parser.parse_args(argv)

    class_names = load_class_names(args.checkpoint)
    archive = resolve_archive(args.source_archive)
    cache_path = None if args.no_cache else args.output_dir / "cache" / "reference_index.json"
    reference = build_reference_index(args.processed_dir, archive, cache_path)

    print("Auditing processed splits ...")
    processed = audit_processed(args.processed_dir, reference, class_names, args.near_distance)
    print("Auditing real-world folder ...")
    real_world = audit_real_world(args.real_world_dir, class_names, reference,
                                  args.min_per_class, args.near_distance)

    audit = {
        "report_type": "dataset_audit",
        "schema_version": 1,
        "generated_at": utc_now(),
        "checkpoint": rel(args.checkpoint),
        "class_names": class_names,
        "source_archive": str(archive) if archive else None,
        "source_archive_classes": audit_archive_classes(reference, class_names),
        "processed": processed,
        "real_world": strip_private(real_world),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "dataset_audit.json", audit)
    write_text(args.output_dir / "dataset_audit.md", audit_markdown(audit))
    print(f"Wrote {rel(args.output_dir / 'dataset_audit.json')} and dataset_audit.md")
    for warning in real_world["warnings"]:
        print(f"WARNING (real-world): {warning}")
    return audit


if __name__ == "__main__":
    main()
