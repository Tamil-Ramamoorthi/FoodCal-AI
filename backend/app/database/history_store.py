"""SQLite persistence for predictions and user-verified feedback.

Uploaded images are never stored in SQLite or written to disk as
uploaded. Each distinct image (by SHA-256 of the uploaded bytes) gets at
most one small JPEG thumbnail, re-encoded by the backend from decoded
pixels, at <data_dir>/thumbnails/<sha256>.jpg. File names come only from
the hash, never from user input.
"""

import json
import os
import re
import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


SCHEMA_VERSION = 1

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
# Names written by save_thumbnail; cleanup never touches any other file.
_THUMBNAIL_NAME = re.compile(r"^[0-9a-f]{64}\.jpg$")
_TEMPORARY_NAME = re.compile(r"^[0-9a-f]{64}\.[0-9a-f]{32}\.tmp$")
# A temporary file this old is left over from an interrupted write, not
# one still being written.
STALE_TEMPORARY_SECONDS = 3600

_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id                  TEXT PRIMARY KEY CHECK (length(id) = 32),
    created_at          TEXT NOT NULL,
    original_filename   TEXT NOT NULL,
    image_sha256        TEXT NOT NULL CHECK (length(image_sha256) = 64),
    predicted_food      TEXT NOT NULL,
    confidence          REAL NOT NULL CHECK (confidence BETWEEN 0 AND 100),
    top_predictions     TEXT NOT NULL CHECK (json_valid(top_predictions)),
    device              TEXT NOT NULL,
    model_name          TEXT NOT NULL,
    verification_status TEXT NOT NULL DEFAULT 'unverified'
        CHECK (verification_status IN ('unverified', 'correct', 'incorrect')),
    actual_food         TEXT,
    verified_at         TEXT,
    CHECK (
        (verification_status = 'unverified'
            AND actual_food IS NULL AND verified_at IS NULL)
        OR (verification_status = 'correct'
            AND actual_food = predicted_food AND verified_at IS NOT NULL)
        OR (verification_status = 'incorrect'
            AND actual_food IS NOT NULL AND actual_food <> predicted_food
            AND verified_at IS NOT NULL)
    )
);
CREATE INDEX IF NOT EXISTS idx_predictions_created
    ON predictions (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_image
    ON predictions (image_sha256);
CREATE INDEX IF NOT EXISTS idx_predictions_status
    ON predictions (verification_status);
"""

_COLUMNS = (
    "id, created_at, original_filename, image_sha256, predicted_food, "
    "confidence, top_predictions, device, model_name, "
    "verification_status, actual_food, verified_at"
)


def _utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _escape_like(text):
    return (
        text.replace("\\", "\\\\")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )


class HistoryStore:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.db_path = self.data_dir / "foodcal.db"
        self.thumbnail_dir = self.data_dir / "thumbnails"

    # ---------------------------------------------------------------
    # Setup
    # ---------------------------------------------------------------

    def initialize(self):
        self.thumbnail_dir.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version > SCHEMA_VERSION:
                raise RuntimeError(
                    f"Database schema version {version} is newer than "
                    f"this application supports ({SCHEMA_VERSION})."
                )
            conn.executescript(_SCHEMA)
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=5)
        conn.row_factory = sqlite3.Row
        try:
            with conn:  # commits on success, rolls back on error
                yield conn
        finally:
            conn.close()

    # ---------------------------------------------------------------
    # Thumbnails
    # ---------------------------------------------------------------

    def thumbnail_path(self, image_sha256):
        if not _SHA256.match(image_sha256):
            raise ValueError("Invalid image hash.")
        return self.thumbnail_dir / f"{image_sha256}.jpg"

    def save_thumbnail(self, image_sha256, jpeg_bytes):
        path = self.thumbnail_path(image_sha256)
        if path.exists():
            return
        temporary = path.with_name(f"{path.stem}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_bytes(jpeg_bytes)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def remove_thumbnail_if_unused(self, image_sha256):
        with self._connect() as conn:
            in_use = conn.execute(
                "SELECT 1 FROM predictions WHERE image_sha256 = ? LIMIT 1",
                (image_sha256,),
            ).fetchone()
        if not in_use:
            self.thumbnail_path(image_sha256).unlink(missing_ok=True)

    def remove_unused_thumbnails(self):
        """Removes thumbnails no record refers to and stale temporary files.

        Returns the number of files removed. A file that cannot be removed
        is skipped; the next cleanup tries it again.
        """
        with self._connect() as conn:
            in_use = {
                row[0]
                for row in conn.execute(
                    "SELECT DISTINCT image_sha256 FROM predictions"
                )
            }
        stale_before = time.time() - STALE_TEMPORARY_SECONDS
        removed = 0
        for path in self.thumbnail_dir.iterdir():
            try:
                if _THUMBNAIL_NAME.match(path.name):
                    unused = path.name[:64] not in in_use
                elif _TEMPORARY_NAME.match(path.name):
                    unused = path.stat().st_mtime < stale_before
                else:
                    continue
                if unused and path.is_file():
                    path.unlink()
                    removed += 1
            except OSError as error:
                print(f"WARNING: could not remove {path.name}: {error}")
        return removed

    # ---------------------------------------------------------------
    # Records
    # ---------------------------------------------------------------

    def _to_record(self, row):
        record = dict(row)
        record["top_predictions"] = json.loads(record["top_predictions"])
        record["thumbnail_available"] = self.thumbnail_path(
            record["image_sha256"]
        ).is_file()
        return record

    def add_prediction(
        self,
        *,
        filename,
        image_sha256,
        predicted_food,
        confidence,
        top_predictions,
        device,
        model_name,
    ):
        record_id = uuid.uuid4().hex
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO predictions (id, created_at, original_filename, "
                "image_sha256, predicted_food, confidence, top_predictions, "
                "device, model_name) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record_id,
                    _utc_now(),
                    (filename or "unnamed")[:255],
                    image_sha256,
                    predicted_food,
                    confidence,
                    json.dumps(top_predictions),
                    device,
                    model_name,
                ),
            )
        return self.get_prediction(record_id)

    def get_prediction(self, record_id):
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT {_COLUMNS} FROM predictions WHERE id = ?",
                (record_id,),
            ).fetchone()
        return self._to_record(row) if row else None

    def list_predictions(self, *, limit, offset, query=""):
        where = ""
        params = []
        query = query.strip().lower().replace("_", " ")
        if query:
            pattern = f"%{_escape_like(query)}%"
            where = (
                "WHERE replace(lower(predicted_food), '_', ' ') "
                "LIKE ? ESCAPE '\\' "
                "OR replace(lower(coalesce(actual_food, '')), '_', ' ') "
                "LIKE ? ESCAPE '\\' "
                "OR lower(original_filename) LIKE ? ESCAPE '\\'"
            )
            params = [pattern, pattern, pattern]

        with self._connect() as conn:
            total = conn.execute(
                f"SELECT COUNT(*) FROM predictions {where}", params
            ).fetchone()[0]
            rows = conn.execute(
                f"SELECT {_COLUMNS} FROM predictions {where} "
                "ORDER BY created_at DESC, rowid DESC LIMIT ? OFFSET ?",
                [*params, limit, offset],
            ).fetchall()
        return [self._to_record(row) for row in rows], total

    def set_feedback(self, record_id, actual_food):
        # Correctness is derived from the stored prediction, never from
        # anything the client claims.
        with self._connect() as conn:
            updated = conn.execute(
                "UPDATE predictions SET actual_food = ?, "
                "verification_status = CASE WHEN predicted_food = ? "
                "THEN 'correct' ELSE 'incorrect' END, verified_at = ? "
                "WHERE id = ?",
                (actual_food, actual_food, _utc_now(), record_id),
            ).rowcount
        return self.get_prediction(record_id) if updated else None

    def clear_feedback(self, record_id):
        with self._connect() as conn:
            updated = conn.execute(
                "UPDATE predictions SET actual_food = NULL, "
                "verification_status = 'unverified', verified_at = NULL "
                "WHERE id = ?",
                (record_id,),
            ).rowcount
        return self.get_prediction(record_id) if updated else None

    def delete_prediction(self, record_id):
        with self._connect() as conn:
            row = conn.execute(
                "SELECT image_sha256 FROM predictions WHERE id = ?",
                (record_id,),
            ).fetchone()
            if row is None:
                return False
            conn.execute("DELETE FROM predictions WHERE id = ?", (record_id,))
        # Only after the delete has committed. The record is gone either
        # way; a thumbnail left behind is removed by the next cleanup.
        try:
            self.remove_thumbnail_if_unused(row["image_sha256"])
        except (OSError, sqlite3.Error) as error:
            print(f"WARNING: thumbnail not removed: {error}")
        return True

    def clear_predictions(self):
        """Deletes every prediction record, then the thumbnails they used.

        Returns (records_deleted, thumbnails_removed). The table, schema and
        database file are kept. A prediction saved while this runs keeps its
        record; at worst its thumbnail is then reported as unavailable.
        """
        with self._connect() as conn:
            deleted = conn.execute("DELETE FROM predictions").rowcount
        # Only after the delete has committed, which must still be reported
        # as done if the cleanup fails.
        try:
            removed = self.remove_unused_thumbnails()
        except (OSError, sqlite3.Error) as error:
            print(f"WARNING: thumbnails not cleaned up: {error}")
            removed = 0
        return deleted, removed

    # ---------------------------------------------------------------
    # Live evaluation (user-verified labels only)
    # ---------------------------------------------------------------

    def live_metrics(self):
        with self._connect() as conn:
            totals = conn.execute(
                "SELECT COUNT(*) AS total, "
                "COALESCE(SUM(verification_status <> 'unverified'), 0) AS verified, "
                "COALESCE(SUM(verification_status = 'correct'), 0) AS correct "
                "FROM predictions"
            ).fetchone()
            actual_rows = conn.execute(
                "SELECT actual_food AS food, COUNT(*) AS actual, "
                "SUM(verification_status = 'correct') AS correct "
                "FROM predictions WHERE verification_status <> 'unverified' "
                "GROUP BY actual_food"
            ).fetchall()
            predicted_rows = conn.execute(
                "SELECT predicted_food AS food, COUNT(*) AS predicted "
                "FROM predictions WHERE verification_status <> 'unverified' "
                "GROUP BY predicted_food"
            ).fetchall()
            most_predicted = conn.execute(
                "SELECT predicted_food AS food, COUNT(*) AS count "
                "FROM predictions GROUP BY predicted_food "
                "ORDER BY count DESC, predicted_food LIMIT 1"
            ).fetchone()

        per_class = {}
        for row in actual_rows:
            per_class[row["food"]] = {
                "food": row["food"],
                "actual": row["actual"],
                "predicted": 0,
                "correct": row["correct"],
            }
        for row in predicted_rows:
            entry = per_class.setdefault(
                row["food"],
                {"food": row["food"], "actual": 0, "predicted": 0, "correct": 0},
            )
            entry["predicted"] = row["predicted"]

        verified = totals["verified"]
        correct = totals["correct"]
        return {
            "total_predictions": totals["total"],
            "verified": verified,
            "correct": correct,
            "incorrect": verified - correct,
            "accuracy": correct / verified if verified else None,
            "per_class": sorted(
                per_class.values(),
                key=lambda item: (-item["actual"], -item["predicted"], item["food"]),
            ),
            "most_predicted": dict(most_predicted) if most_predicted else None,
        }
