"""Intake of additional labelled food images for training (Phase 2).

Run from the project root:

    .venv-gpu\\Scripts\\python.exe -m evaluation.intake init <batch_id>
    .venv-gpu\\Scripts\\python.exe -m evaluation.intake sync <batch_id>
    .venv-gpu\\Scripts\\python.exe -m evaluation.intake validate <batch_id> [--model-hint]

- init      creates data/intake/<batch_id>/images/<class>/ for every model
            class and an empty intake.csv.
- sync      appends a row to intake.csv for every image file not listed yet
            (proposed_label = its folder). Existing rows are never changed.
- validate  read-only. Writes reports/intake/<batch_id>/validation.json and
            .md: decoding, resolution, quality, labels, provenance,
            presentation tags, review decisions, and exact / near
            duplicates against every processed split, the real-world test
            set, other intake batches and the batch itself.

Image files are never moved, renamed, edited or deleted. Labels come only
from the person filling in intake.csv: `proposed_label` from the folder the
image was put in, and `decision` from the reviewer. The optional model hint
is shown for spotting mistakes and never sets or changes a label.

Images listed here are candidates for TRAINING / VALIDATION only. Photos for
the independent real-world test set go to data/real_world_test/ instead
(see evaluation/README.md) and must never be added to an intake batch.
"""

import argparse
import csv
import json
import os
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .dataset_audit import (
    DEFAULT_CHECKPOINT,
    DEFAULT_NEAR_DISTANCE,
    DEFAULT_PROCESSED_DIR,
    DEFAULT_REAL_WORLD_DIR,
    ROOT,
    load_class_names,
    rel,
    utc_now,
)
from .hashing import (
    SUPPORTED_EXTENSIONS,
    ReferenceIndex,
    build_reference_index,
    dhash,
    dhash_from_bytes,
    hamming_distances,
    sha256_bytes,
    sha256_file,
)
from .reports import _bullets, _table, write_json, write_text

DEFAULT_INTAKE_DIR = ROOT / "data" / "intake"
DEFAULT_REPORT_DIR = ROOT / "reports" / "intake"
HASH_CACHE = ROOT / "reports" / "real_world" / "cache" / "reference_index.json"
DEFAULT_HINT_CHECKPOINT = ROOT / "models" / "efficientnet_b0_v2" / "food_efficientnet_b0_v2_best.pth"
INTAKE_FILE = "intake.csv"
# Optional per-batch settings, e.g. {"kind": "user_feedback", "min_short_side": 160}.
BATCH_SETTINGS_FILE = "batch.json"
BATCH_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")

# Short side of the EXIF-oriented original. Training images are stored at
# 224 px; 448 px leaves room for random-resized-crop augmentation.
MIN_SHORT_SIDE = 448

# Presentation tags (one value per axis, blank = untagged). They make
# coverage gaps visible; they are not used as labels.
AXES = {
    "serving": ("single", "several", "with_sides", "thali", "partly_eaten"),
    "variant": ("standard", "mini", "stuffed_or_rava", "cut_or_sliced", "other"),
    "garnish": ("none", "herbs", "chilli", "nuts", "saffron", "sauce", "mixed", "other"),
    "plate": ("white_plate", "steel_plate", "banana_leaf", "dark_plate", "bowl", "takeaway", "other"),
    "background": ("table", "cloth", "counter", "restaurant", "cluttered", "other"),
    "lighting": ("daylight", "warm_indoor", "dim", "flash", "harsh_shadow"),
    "angle": ("top_down", "angle_45", "side"),
    "distance": ("close_up", "whole_plate", "table_scene"),
}
REQUIRED_FIELDS = ("source", "license", "captured_by", "capture_date", "session_id")
COLUMNS = ["file", "proposed_label", *REQUIRED_FIELDS, *AXES, "reviewer", "decision", "notes"]

# Quality heuristics, computed on a grey copy scaled to 512 px on the long
# side. They are NOT calibrated on this project's data: flagged images are
# warnings for a person to look at, never rejected automatically.
QUALITY_SIDE = 512
QUALITY_THRESHOLDS = {
    "sharpness_below": 100.0,  # variance of the Laplacian; low = possibly blurred
    "brightness_below": 40.0,  # mean grey level 0-255
    "brightness_above": 215.0,
    "contrast_below": 20.0,  # grey-level standard deviation
}

# Sources whose images must never reach training, even as a possible
# near-duplicate: they are what the models are evaluated on.
HELD_OUT_SOURCES = ("processed/validation", "processed/test", "real_world_test")


# ---------------------------------------------------------------------
# intake.csv and the image folders
# ---------------------------------------------------------------------

def check_batch_id(batch_id):
    if not BATCH_ID_PATTERN.match(batch_id or ""):
        raise ValueError(
            f"Invalid batch id {batch_id!r}: use letters, digits, '_', '-' or '.', "
            "starting with a letter or digit (e.g. 2026-10-15_phone_idli)."
        )
    return batch_id


def batch_settings(batch_dir):
    path = Path(batch_dir) / BATCH_SETTINGS_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def normalize_file(value):
    value = (value or "").strip().replace("\\", "/")
    while value.startswith("./"):
        value = value[2:]
    return value


def read_intake(csv_path):
    """Returns (rows, fieldnames, errors). Each row also carries `_line`."""
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        return [], list(COLUMNS), [f"`{rel(csv_path)}` not found. Run `sync` first."]
    with open(csv_path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = [(name or "").strip() for name in reader.fieldnames or []]
        rows = []
        for line, raw in enumerate(reader, start=2):
            row = {(k or "").strip(): (v or "").strip() for k, v in raw.items() if isinstance(k, str)}
            row["_line"] = line
            rows.append(row)
    missing = [c for c in COLUMNS if c not in fields]
    errors = [f"`{INTAKE_FILE}` is missing columns: {', '.join(missing)}."] if missing else []
    return rows, fields, errors


def write_intake(csv_path, rows, fields):
    csv_path = Path(csv_path)
    temporary = csv_path.with_name(csv_path.name + ".tmp")
    # utf-8-sig so Excel opens names with non-ASCII characters correctly.
    with open(temporary, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    try:
        os.replace(temporary, csv_path)
    except PermissionError as error:
        temporary.unlink(missing_ok=True)
        raise PermissionError(f"Cannot write {csv_path}; close it in Excel and try again.") from error


def scan_images(images_dir, class_names):
    """Image files under images/<class>/. Returns ({file: folder}, errors, warnings)."""
    images_dir = Path(images_dir)
    files, errors, warnings = {}, [], []
    if not images_dir.is_dir():
        return files, [f"`{rel(images_dir)}` does not exist."], warnings
    known = set(class_names)
    lowered = {name.lower(): name for name in class_names}
    for entry in sorted(images_dir.iterdir()):
        if entry.is_file():
            errors.append(f"`{rel(entry)}` is not inside a class folder.")
            continue
        if entry.name not in known:
            hint = lowered.get(entry.name.lower())
            errors.append(f"Folder `{rel(entry)}` is not a model class"
                          + (f" (did you mean `{hint}`?)." if hint else "."))
            continue
        for path in sorted(entry.iterdir()):
            if path.is_dir():
                errors.append(f"Nested folder `{rel(path)}` is not scanned; move its images up one level.")
            elif path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                warnings.append(f"`{rel(path)}` ignored: only .jpg, .jpeg, .png and .webp are supported.")
            else:
                files[f"{entry.name}/{path.name}"] = entry.name
    return files, errors, warnings


def init_batch(batch_id, class_names, intake_dir=DEFAULT_INTAKE_DIR):
    batch_dir = Path(intake_dir) / check_batch_id(batch_id)
    if batch_dir.exists():
        raise FileExistsError(f"{batch_dir} already exists; nothing was changed.")
    for name in class_names:
        (batch_dir / "images" / name).mkdir(parents=True)
    write_intake(batch_dir / INTAKE_FILE, [], COLUMNS)
    return batch_dir


def sync_batch(batch_dir, class_names):
    """Appends rows for unlisted image files. Returns (added files, scan errors, scan warnings)."""
    batch_dir = Path(batch_dir)
    csv_path = batch_dir / INTAKE_FILE
    rows, fields, _ = read_intake(csv_path) if csv_path.is_file() else ([], [], [])
    missing_columns = [c for c in COLUMNS if c not in fields]
    files, errors, warnings = scan_images(batch_dir / "images", class_names)
    listed = {normalize_file(row.get("file")) for row in rows}
    added = [{"file": f, "proposed_label": label} for f, label in sorted(files.items()) if f not in listed]
    if added or missing_columns:
        write_intake(csv_path, rows + added, fields + missing_columns)
    return [row["file"] for row in added], errors, warnings


# ---------------------------------------------------------------------
# Per-image checks
# ---------------------------------------------------------------------

def quality_stats(image):
    gray = image.convert("L")
    scale = QUALITY_SIDE / max(gray.size)
    if scale < 1:
        gray = gray.resize((max(1, round(gray.width * scale)), max(1, round(gray.height * scale))),
                           Image.Resampling.BILINEAR)
    a = np.asarray(gray, dtype=np.float64)
    laplacian = a[1:-1, :-2] + a[1:-1, 2:] + a[:-2, 1:-1] + a[2:, 1:-1] - 4 * a[1:-1, 1:-1]
    return {
        "sharpness": round(float(laplacian.var()), 1) if laplacian.size else 0.0,
        "brightness": round(float(a.mean()), 1),
        "contrast": round(float(a.std()), 1),
    }


def inspect_image(path):
    with Image.open(path) as image:
        image_format = image.format
        oriented = ImageOps.exif_transpose(image)
        oriented.load()  # full decode catches truncated files
        info = {"format": image_format, "width": oriented.width, "height": oriented.height,
                "dhash": dhash(oriented), **quality_stats(oriented)}
    info["sha256"] = sha256_file(path)
    return info


def quality_warnings(info, thresholds=QUALITY_THRESHOLDS):
    out = []
    if info["sharpness"] < thresholds["sharpness_below"]:
        out.append(f"possibly blurred (sharpness {info['sharpness']} < {thresholds['sharpness_below']:g})")
    if info["brightness"] < thresholds["brightness_below"]:
        out.append(f"very dark (mean brightness {info['brightness']})")
    if info["brightness"] > thresholds["brightness_above"]:
        out.append(f"very bright or washed out (mean brightness {info['brightness']})")
    if info["contrast"] < thresholds["contrast_below"]:
        out.append(f"very low contrast ({info['contrast']})")
    return out


def parse_decision(value, class_names):
    """Returns (kind, target_label, error). kind is None when not yet reviewed."""
    value = (value or "").strip()
    if not value:
        return None, None, None
    lowered = value.lower()
    if lowered in ("accept", "reject"):
        return lowered, None, None
    if lowered.startswith("relabel:"):
        target = value.split(":", 1)[1].strip()
        if target in class_names:
            return "relabel", target, None
        hint = {n.lower(): n for n in class_names}.get(target.lower())
        return None, None, (f"decision relabels to unknown class `{target}`"
                            + (f" (did you mean `{hint}`?)" if hint else ""))
    return None, None, f"decision `{value}` must be `accept`, `reject` or `relabel:<class>`"


def check_fields(row, folder, class_names, today):
    errors, warnings = [], []
    label = row.get("proposed_label", "")
    if label not in class_names:
        hint = {n.lower(): n for n in class_names}.get(label.lower())
        errors.append(f"proposed_label `{label}` is not a model class" + (f" (did you mean `{hint}`?)" if hint else ""))
    elif folder is not None and folder != label:
        errors.append(f"proposed_label `{label}` does not match its folder `{folder}`")
    for field in REQUIRED_FIELDS:
        if not row.get(field):
            errors.append(f"`{field}` is empty")
    raw_date = row.get("capture_date", "")
    if raw_date:
        try:
            captured = date.fromisoformat(raw_date)
            if captured > today:
                errors.append(f"capture_date {raw_date} is in the future")
        except ValueError:
            errors.append(f"capture_date `{raw_date}` must be YYYY-MM-DD (format the column as text in Excel)")
    untagged = []
    for axis, allowed in AXES.items():
        value = row.get(axis, "")
        if not value:
            untagged.append(axis)
        elif value not in allowed:
            errors.append(f"{axis} `{value}` is not one of: {', '.join(allowed)}")
    if untagged:
        warnings.append(f"untagged: {', '.join(untagged)}")
    kind, target, decision_error = parse_decision(row.get("decision"), class_names)
    if decision_error:
        errors.append(decision_error)
    if row.get("decision") and not row.get("reviewer"):
        errors.append("a decision is filled in but `reviewer` is empty")
    if kind == "relabel" and target == label:
        warnings.append("relabel target equals proposed_label; use `accept`")
    return errors, warnings, kind, target


# ---------------------------------------------------------------------
# Reference images: everything an intake image must not duplicate
# ---------------------------------------------------------------------

def hash_folder(root, source, pattern):
    entries = []
    for path in sorted(Path(root).glob(pattern)):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            data = path.read_bytes()
            entries.append({"source": source, "path": rel(path), "label": path.parent.name,
                            "sha256": sha256_bytes(data), "dhash": dhash_from_bytes(data)})
    return entries


def build_references(batch_dir, processed_dir, real_world_dir, cache_path):
    entries = []
    if processed_dir and Path(processed_dir).is_dir():
        entries += build_reference_index(processed_dir, None, cache_path, progress=lambda *_: None).entries
    if real_world_dir and Path(real_world_dir).is_dir():
        entries += hash_folder(real_world_dir, "real_world_test", "*/*")
    batch_dir = Path(batch_dir).resolve()
    for other in sorted(batch_dir.parent.iterdir()):
        if other.is_dir() and other.resolve() != batch_dir and (other / "images").is_dir():
            entries += hash_folder(other / "images", f"intake/{other.name}", "*/*")
    return ReferenceIndex(entries)


def reference_findings(record, references, near_distance):
    errors, warnings = [], []
    exact = references.exact_matches(record["sha256"])
    exact_paths = {m["path"] for m in exact}
    for m in exact:
        errors.append(f"exact copy of {m['source']} `{m['path']}` ({m['label']})")
    for m in references.near_matches(record["dhash"], near_distance, limit=10):
        if m["path"] in exact_paths:
            continue
        message = (f"possible near-duplicate of {m['source']} `{m['path']}` ({m['label']}, "
                   f"dHash distance {m['distance']})")
        if m["source"] in HELD_OUT_SOURCES:
            errors.append(message + ": excluded so it cannot leak into an evaluation set")
        else:
            warnings.append(message + ": review; it adds little new information")
    return errors, warnings


def within_batch_findings(records, near_distance):
    """Adds exact / near duplicate findings between images of the same batch."""
    hashed = [r for r in records if r.get("sha256")]
    by_sha = {}
    for r in hashed:
        by_sha.setdefault(r["sha256"], []).append(r)
    for group in by_sha.values():
        for r in group:
            for other in group:
                if other is not r:
                    r["errors"].append(f"exact duplicate of `{other['file']}` in this batch")
    with_dhash = [r for r in hashed if r["dhash"] is not None]
    values = np.array([r["dhash"] for r in with_dhash], dtype=np.uint64)
    for i, r in enumerate(with_dhash):
        distances = hamming_distances(values, r["dhash"])
        for j in np.flatnonzero(distances <= near_distance):
            other = with_dhash[j]
            if j == i or other["sha256"] == r["sha256"]:
                continue
            if other["label"] != r["label"]:
                r["warnings"].append(f"possible near-duplicate of `{other['file']}` which is labelled "
                                     f"`{other['label']}` (distance {int(distances[j])})")
            elif other["session_id"] != r["session_id"]:
                r["warnings"].append(f"possible near-duplicate of `{other['file']}` from a different session_id; "
                                     "photos of the same plate must share one session_id so they stay in one split")


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def validate_batch(batch_dir, class_names, processed_dir=DEFAULT_PROCESSED_DIR, real_world_dir=DEFAULT_REAL_WORLD_DIR,
                   cache_path=HASH_CACHE, near_distance=DEFAULT_NEAR_DISTANCE, min_short_side=None,
                   hint=None, today=None):
    """Read-only. `hint` is an optional (description, callable(paths) -> [[(label, prob)] * 3]) pair.
    min_short_side: None = the batch's batch.json setting, else MIN_SHORT_SIDE."""
    batch_dir = Path(batch_dir)
    settings = batch_settings(batch_dir)
    if min_short_side is None:
        min_short_side = int(settings.get("min_short_side", MIN_SHORT_SIDE))
    today = today or date.today()
    rows, _, batch_errors = read_intake(batch_dir / INTAKE_FILE)
    files, scan_errors, batch_warnings = scan_images(batch_dir / "images", class_names)
    batch_errors += scan_errors

    seen = {}
    for row in rows:
        name = normalize_file(row.get("file"))
        if name in seen:
            batch_errors.append(f"`{name}` is listed twice (lines {seen[name]} and {row['_line']}).")
        seen.setdefault(name, row["_line"])
    unlisted = sorted(set(files) - set(seen))
    if unlisted:
        batch_errors.append(f"{len(unlisted)} image(s) are not listed in {INTAKE_FILE}; run `sync`: "
                            + ", ".join(f"`{f}`" for f in unlisted[:10]) + (" ..." if len(unlisted) > 10 else ""))

    references = build_references(batch_dir, processed_dir, real_world_dir, cache_path)
    records = []
    for row in rows:
        name = normalize_file(row.get("file"))
        folder = name.split("/", 1)[0] if "/" in name else None
        errors, warnings, kind, target = check_fields(row, folder, class_names, today)
        record = {
            "file": name, "line": row["_line"], "proposed_label": row.get("proposed_label", ""),
            "label": target or row.get("proposed_label", ""), "decision": kind,
            "session_id": row.get("session_id", ""), "tags": {axis: row.get(axis, "") for axis in AXES},
            "sha256": None, "dhash": None, "errors": errors, "warnings": warnings,
        }
        path = batch_dir / "images" / name
        if not name:
            errors.append("`file` is empty")
        elif not path.is_file():
            errors.append(f"file not found: `{rel(path)}`")
        elif path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            errors.append("unsupported file type; use .jpg, .jpeg, .png or .webp")
        else:
            try:
                info = inspect_image(path)
            except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as error:
                errors.append(f"cannot be decoded ({type(error).__name__}: {error})")
            else:
                record.update(info)
                if info["format"] not in ("JPEG", "PNG", "WEBP"):
                    errors.append(f"file content is {info['format']}, not JPEG, PNG or WEBP")
                if min(info["width"], info["height"]) < min_short_side:
                    errors.append(f"too small: {info['width']}x{info['height']}, short side must be at least "
                                  f"{min_short_side} px")
                warnings += quality_warnings(info)
                more_errors, more_warnings = reference_findings(record, references, near_distance)
                errors += more_errors
                warnings += more_warnings
        records.append(record)
    within_batch_findings(records, near_distance)

    hint_info = None
    if hint:
        description, predict_top3 = hint
        decodable = [r for r in records if r["sha256"]]
        for r, top3 in zip(decodable, predict_top3([batch_dir / "images" / r["file"] for r in decodable])):
            r["model_hint"] = [{"label": label, "probability": round(float(p), 4)} for label, p in top3]
            top_label, top_p = top3[0]
            if top_label != r["label"] and top_p >= 0.5:
                r["warnings"].append(f"model hint disagrees: {top_label} {top_p:.0%}. Check the photo; "
                                     "the model is often wrong, so never relabel on this alone")
        hint_info = {"model": description,
                     "note": "Shown only to help spot labelling mistakes; it never sets or changes a label."}

    for r in records:
        if r["errors"]:
            r["status"] = "error"
        elif r["decision"] is None:
            r["status"] = "pending_review"
        elif r["decision"] == "reject":
            r["status"] = "rejected"
        else:
            r["status"] = "ready"
        r["dhash"] = None if r["dhash"] is None else f"{r['dhash']:016x}"

    statuses = [r["status"] for r in records]
    summary = {
        "rows": len(records),
        "images_on_disk": len(files),
        "ready": statuses.count("ready"),
        "pending_review": statuses.count("pending_review"),
        "rejected": statuses.count("rejected"),
        "with_errors": statuses.count("error"),
        "with_warnings": sum(bool(r["warnings"]) for r in records),
        "batch_errors": len(batch_errors),
    }
    return {
        "report_type": "intake_validation",
        "generated_at": utc_now(),
        "batch": rel(batch_dir),
        "batch_settings": settings,
        "class_names": list(class_names),
        "near_duplicate_max_distance": near_distance,
        "min_short_side": min_short_side,
        "quality_thresholds": {**QUALITY_THRESHOLDS,
                               "note": "Heuristics, not calibrated on this project's data. Warnings only."},
        "reference_images": references.sources(),
        "model_hint": hint_info,
        "summary": summary,
        "batch_errors": batch_errors,
        "batch_warnings": batch_warnings,
        "coverage": coverage(records),
        "images": records,
    }


def coverage(records):
    """Presentation-tag counts per label, over images not rejected and without errors."""
    out = {}
    for r in records:
        if r["status"] in ("error", "rejected"):
            continue
        c = out.setdefault(r["label"], {"images": 0, "ready": 0, "sessions": set(),
                                        "tags": {axis: {} for axis in AXES}})
        c["images"] += 1
        c["ready"] += r["status"] == "ready"
        c["sessions"].add(r["session_id"])
        for axis, value in r["tags"].items():
            key = value or "untagged"
            c["tags"][axis][key] = c["tags"][axis].get(key, 0) + 1
    for c in out.values():
        c["sessions"] = len(c["sessions"])
    return dict(sorted(out.items()))


def validation_markdown(report):
    s = report["summary"]
    lines = [
        f"# Intake validation: `{report['batch']}`", "",
        f"Generated {report['generated_at']}. Read-only: no image or CSV was changed.", "",
        _table(["", "Images"], [
            ("Rows in intake.csv", s["rows"]), ("Image files on disk", s["images_on_disk"]),
            ("**Ready** (reviewed, no errors)", s["ready"]), ("Pending review (no decision yet)", s["pending_review"]),
            ("Rejected by reviewer", s["rejected"]), ("With errors", s["with_errors"]),
            ("With warnings", s["with_warnings"]),
        ]), "",
        "Only **ready** images may be used to build a training manifest. Errors must be fixed (or the image "
        "rejected); warnings need a person to look at the image.", "",
        "Duplicates were checked against: "
        + ", ".join(f"{source} ({n})" for source, n in sorted(report["reference_images"].items()))
        + f". Near-duplicate = dHash distance <= {report['near_duplicate_max_distance']}; it can miss crops "
          "and rotations and can match different but similar photos.", "",
        f"Quality warnings use uncalibrated heuristics: {', '.join(f'{k} {v:g}' for k, v in report['quality_thresholds'].items() if k != 'note')}.",
    ]
    if report["model_hint"]:
        lines += ["", f"Model hint: `{report['model_hint']['model']}`. {report['model_hint']['note']}"]
    lines += ["", "## Batch errors", _bullets(report["batch_errors"]), "", "## Batch warnings", _bullets(report["batch_warnings"])]
    errored = [r for r in report["images"] if r["errors"]]
    lines += ["", f"## Images with errors ({len(errored)})"]
    lines += [f"- `{r['file']}` (line {r['line']}): " + "; ".join(r["errors"])
              + (f". Warnings: {'; '.join(r['warnings'])}" if r["warnings"] else "") for r in errored] or ["None."]
    warned = [r for r in report["images"] if r["warnings"] and not r["errors"]]
    lines += ["", f"## Images with warnings only ({len(warned)})"]
    lines += [f"- `{r['file']}` ({r['status']}): " + "; ".join(r["warnings"]) for r in warned] or ["None."]
    lines += ["", "## Coverage by class (images not rejected and without errors)"]
    if not report["coverage"]:
        lines.append("None.")
    for label, c in report["coverage"].items():
        lines += ["", f"### {label}: {c['images']} images ({c['ready']} ready), {c['sessions']} session(s)",
                  _table(["Tag", "Values"], [
                      (axis, ", ".join(f"{v} {n}" for v, n in sorted(values.items(), key=lambda kv: -kv[1])))
                      for axis, values in c["tags"].items()])]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------

def model_hint(checkpoint, device_name):
    import torch

    from backend.app.models.classifier import load_classifier

    from .diagnose_reliability import predict

    auto = "cuda" if torch.cuda.is_available() else "cpu"
    classifier = load_classifier(Path(checkpoint), torch.device(auto if device_name == "auto" else device_name))

    def predict_top3(paths):
        probs = predict(classifier, paths) if paths else np.zeros((0, len(classifier.class_names)))
        return [[(classifier.class_names[i], p[i]) for i in np.argsort(-p)[:3]] for p in probs]

    return rel(checkpoint), predict_top3


def main(argv=None):
    parser = argparse.ArgumentParser(description="Intake of additional labelled food images.")
    parser.add_argument("command", choices=["init", "sync", "validate"])
    parser.add_argument("batch_id")
    parser.add_argument("--intake-dir", type=Path, default=DEFAULT_INTAKE_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="Source of the class list.")
    parser.add_argument("--near-distance", type=int, default=DEFAULT_NEAR_DISTANCE)
    parser.add_argument("--min-short-side", type=int, default=None,
                        help=f"Default: the batch's {BATCH_SETTINGS_FILE} setting, else {MIN_SHORT_SIDE}.")
    parser.add_argument("--model-hint", action="store_true", help="Add the v2 model's top-3 as a review aid.")
    parser.add_argument("--hint-checkpoint", type=Path, default=DEFAULT_HINT_CHECKPOINT)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args(argv)

    class_names = load_class_names(args.checkpoint)
    batch_dir = args.intake_dir / check_batch_id(args.batch_id)
    if args.command == "init":
        init_batch(args.batch_id, class_names, args.intake_dir)
        print(f"Created {rel(batch_dir)}. Put images in images/<class>/, then run `sync`.")
        return 0
    if not batch_dir.is_dir():
        print(f"{rel(batch_dir)} does not exist; run `init` first.", file=sys.stderr)
        return 2
    if args.command == "sync":
        added, errors, warnings = sync_batch(batch_dir, class_names)
        print(f"Added {len(added)} row(s) to {rel(batch_dir / INTAKE_FILE)}.")
        for message in errors + warnings:
            print(f"  - {message}")
        return 1 if errors else 0

    hint = model_hint(args.hint_checkpoint, args.device) if args.model_hint else None
    report = validate_batch(batch_dir, class_names, near_distance=args.near_distance,
                            min_short_side=args.min_short_side, hint=hint)
    out = args.report_dir / args.batch_id
    write_json(out / "validation.json", report)
    write_text(out / "validation.md", validation_markdown(report))
    s = report["summary"]
    print(f"{s['rows']} rows: {s['ready']} ready, {s['pending_review']} pending review, {s['rejected']} rejected, "
          f"{s['with_errors']} with errors, {s['batch_errors']} batch error(s). Wrote {rel(out / 'validation.md')}")
    return 1 if s["with_errors"] or s["batch_errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
