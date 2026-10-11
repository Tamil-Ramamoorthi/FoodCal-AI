"""Exports user feedback from the prediction history into an intake batch
for review (human-in-the-loop, step 1 of 3).

    .venv-gpu\\Scripts\\python.exe -m evaluation.feedback_export [--batch-id ID] [--incorrect-only]

Then review and validate it like any other intake batch:

    #   fill in reviewer + decision in data/intake/<batch_id>/intake.csv
    .venv-gpu\\Scripts\\python.exe -m evaluation.intake validate <batch_id>

and, once approved rows exist, build a candidate training manifest with
training/build_feedback_manifest.py.

Safety rules:
- The history database is opened READ-ONLY (SQLite `mode=ro`); feedback is
  never changed, deleted or marked by this script.
- Only predictions a user verified (status `correct` or `incorrect`) are
  considered, and every exported row starts with an EMPTY review decision:
  user feedback is a proposed label, not verified ground truth.
- The stored image is the backend's thumbnail (at most 256 px, re-encoded
  from the upload; the original upload is never stored). It is copied byte
  for byte into the batch; the thumbnail itself is not touched.
- One row per distinct image (SHA-256 of the uploaded bytes). Repeated
  uploads of the same image with the same label are merged (all prediction
  ids are kept); repeated uploads with DIFFERENT labels are not exported
  and are listed as conflicts for a person to resolve in the app.
- Images exported in an earlier batch are skipped, so re-running the export
  never duplicates feedback.
"""

import argparse
import csv
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

from .dataset_audit import DEFAULT_CHECKPOINT, ROOT, load_class_names, rel, utc_now
from .intake import (
    BATCH_SETTINGS_FILE,
    COLUMNS,
    DEFAULT_INTAKE_DIR,
    INTAKE_FILE,
    init_batch,
    read_intake,
    write_intake,
)
from .reports import _bullets, _table, write_json, write_text

DEFAULT_DATA_DIR = ROOT / "storage"
DEFAULT_REPORT_DIR = ROOT / "reports" / "feedback"
# Thumbnails are at most 256 px on the long side, so their short side is
# below the 448 px intake minimum for new photos. The existing training
# images are 224 x 224, which is also the model's input size. 160 px is a
# chosen floor (not a measured one) that admits 4:3 and 3:2 thumbnails.
FEEDBACK_MIN_SHORT_SIDE = 160
FEEDBACK_COLUMNS = ["prediction_id", "all_prediction_ids", "image_sha256", "predicted_food", "confidence",
                    "user_verdict", "verified_at", "model_name"]
SOURCE = "FoodCal app upload (backend thumbnail, max 256 px)"


def read_feedback(db_path):
    """User-verified predictions, oldest first. The database is opened read-only."""
    statuses = ("correct", "incorrect")
    uri = f"{Path(db_path).resolve().as_uri()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT id, created_at, image_sha256, predicted_food, confidence, verification_status, "
            "actual_food, verified_at, model_name FROM predictions "
            f"WHERE verification_status IN ({','.join('?' * len(statuses))}) "
            "ORDER BY verified_at, id",
            statuses,
        ).fetchall()
    finally:
        conn.close()
    return [dict(row) for row in rows]


def already_exported(intake_dir):
    """image_sha256 -> batch for every image exported by an earlier feedback export."""
    found = {}
    for csv_path in sorted(Path(intake_dir).glob(f"*/{INTAKE_FILE}")):
        rows, _, _ = read_intake(csv_path)
        for row in rows:
            if row.get("image_sha256"):
                found.setdefault(row["image_sha256"], csv_path.parent.name)
    return found


def plan_export(feedback, class_names, thumbnail_dir, exported, incorrect_only=False):
    """Groups feedback by image and decides what to export. Pure; touches nothing.

    Conflicts are found across ALL verified feedback for an image, so a
    'correct' verdict and an 'incorrect' one with another label still count
    as a conflict when only incorrect predictions are exported."""
    groups = {}
    for record in feedback:
        groups.setdefault(record["image_sha256"], []).append(record)
    plan = {"export": [], "conflicts": [], "already_exported": [], "invalid_label": [], "missing_thumbnail": []}
    for sha, records in groups.items():
        ids = [r["id"] for r in records]
        labels = sorted({r["actual_food"] for r in records})
        if incorrect_only and all(r["verification_status"] != "incorrect" for r in records):
            continue
        if sha in exported:
            plan["already_exported"].append({"image_sha256": sha, "prediction_ids": ids, "batch": exported[sha]})
        elif len(labels) > 1:
            plan["conflicts"].append({"image_sha256": sha, "prediction_ids": ids, "labels": labels})
        elif labels[0] not in class_names:
            plan["invalid_label"].append({"image_sha256": sha, "prediction_ids": ids, "label": labels[0]})
        elif not (Path(thumbnail_dir) / f"{sha}.jpg").is_file():
            plan["missing_thumbnail"].append({"image_sha256": sha, "prediction_ids": ids})
        else:
            latest = max(records, key=lambda r: (r["verified_at"], r["id"]))
            plan["export"].append({**latest, "all_prediction_ids": ids})
    return plan


def intake_row(record):
    verdict = record["verification_status"]
    note = (f"User reported the prediction {record['predicted_food']} ({record['confidence']:.1f}%) as incorrect."
            if verdict == "incorrect" else f"User confirmed the prediction {record['predicted_food']}.")
    return {
        "file": f"{record['actual_food']}/{record['id']}.jpg",
        "proposed_label": record["actual_food"],
        "source": SOURCE,
        "license": "uploaded by the app user (local, not redistributable)",
        "captured_by": "FoodCal app user",
        "capture_date": record["created_at"][:10],  # upload date; the capture date is unknown
        # Same image = same session, so repeated uploads stay in one split.
        "session_id": f"img-{record['image_sha256'][:16]}",
        "reviewer": "",
        "decision": "",
        "notes": note,
        "prediction_id": record["id"],
        "all_prediction_ids": " ".join(record["all_prediction_ids"]),
        "image_sha256": record["image_sha256"],
        "predicted_food": record["predicted_food"],
        "confidence": record["confidence"],
        "user_verdict": verdict,
        "verified_at": record["verified_at"],
        "model_name": record["model_name"],
    }


def export_feedback(class_names, data_dir=DEFAULT_DATA_DIR, intake_dir=DEFAULT_INTAKE_DIR, batch_id=None,
                    incorrect_only=False):
    data_dir, intake_dir = Path(data_dir), Path(intake_dir)
    db_path = data_dir / "foodcal.db"
    if not db_path.is_file():
        raise FileNotFoundError(f"History database not found: {db_path}")
    feedback = read_feedback(db_path)
    plan = plan_export(feedback, class_names, data_dir / "thumbnails", already_exported(intake_dir), incorrect_only)
    batch_id = batch_id or "feedback_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    batch_dir = None
    if plan["export"]:
        batch_dir = init_batch(batch_id, class_names, intake_dir)
        write_json(batch_dir / BATCH_SETTINGS_FILE, {
            "kind": "user_feedback", "min_short_side": FEEDBACK_MIN_SHORT_SIDE,
            "exported_at": utc_now(), "database": rel(db_path), "incorrect_only": incorrect_only,
        })
        rows = []
        for record in plan["export"]:
            row = intake_row(record)
            shutil.copyfile(data_dir / "thumbnails" / f"{record['image_sha256']}.jpg",
                            batch_dir / "images" / row["file"])
            rows.append(row)
        write_intake(batch_dir / INTAKE_FILE, rows, COLUMNS + FEEDBACK_COLUMNS)
    return {
        "report_type": "feedback_export",
        "generated_at": utc_now(),
        "database": rel(db_path),
        "database_opened_read_only": True,
        "incorrect_only": incorrect_only,
        "verified_predictions_read": len(feedback),
        "batch": rel(batch_dir) if batch_dir else None,
        "exported": len(plan["export"]),
        "exported_rows": [{k: intake_row(r)[k] for k in ("file", "prediction_id", "all_prediction_ids",
                                                         "predicted_food", "user_verdict")} for r in plan["export"]],
        **{key: plan[key] for key in ("conflicts", "already_exported", "invalid_label", "missing_thumbnail")},
    }


def export_markdown(report):
    lines = [
        "# Feedback export", "",
        f"Generated {report['generated_at']} from `{report['database']}` (opened read-only; nothing in it was changed).", "",
        _table(["", "Images"], [
            ("User-verified predictions read", report["verified_predictions_read"]),
            ("Exported for review", report["exported"]),
            ("Skipped: conflicting labels for the same image", len(report["conflicts"])),
            ("Skipped: exported in an earlier batch", len(report["already_exported"])),
            ("Skipped: label not one of the model's classes", len(report["invalid_label"])),
            ("Skipped: no stored thumbnail", len(report["missing_thumbnail"])),
        ]), "",
    ]
    if report["batch"]:
        lines += [f"Batch: `{report['batch']}`. **Every row has an empty `decision`**: a reviewer must look at "
                  "each image and write `accept`, `reject` or `relabel:<class>` before it can be used.", "",
                  "Next: fill in `reviewer` and `decision`, then run "
                  f"`python -m evaluation.intake validate {Path(report['batch']).name}`.", "",
                  _table(["File", "User verdict", "Model predicted", "Predictions"],
                         [(f"`{r['file']}`", r["user_verdict"], r["predicted_food"], r["all_prediction_ids"])
                          for r in report["exported_rows"]])]
    else:
        lines.append("Nothing new to export, so no batch was created.")
    if report["conflicts"]:
        lines += ["", "## Conflicting feedback (resolve in the app's history page)", _bullets([
            f"image `{c['image_sha256'][:12]}`: labels {', '.join(c['labels'])}; predictions {', '.join(c['prediction_ids'])}"
            for c in report["conflicts"]])]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Export user feedback into an intake batch for review.")
    parser.add_argument("--batch-id")
    parser.add_argument("--incorrect-only", action="store_true", help="Only predictions users marked incorrect.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--intake-dir", type=Path, default=DEFAULT_INTAKE_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="Source of the class list.")
    args = parser.parse_args(argv)
    report = export_feedback(load_class_names(args.checkpoint), args.data_dir, args.intake_dir, args.batch_id,
                             args.incorrect_only)
    name = Path(report["batch"]).name if report["batch"] else "no_batch_" + report["generated_at"][:19].replace(":", "")
    out = args.report_dir / name
    write_json(out / "export.json", report)
    write_text(out / "export.md", export_markdown(report))
    print(f"Read {report['verified_predictions_read']} verified predictions; exported {report['exported']} image(s); "
          f"{len(report['conflicts'])} conflict(s). Wrote {rel(out / 'export.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
