"""Unit tests for the SQLite history store. Each test uses its own
temporary directory, so the real storage/ data is never touched.

Run from the project root:
    .venv-gpu\\Scripts\\python.exe -B -m unittest discover -s tests -v
"""
import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.database.history_store import (  # noqa: E402
    SCHEMA_VERSION,
    HistoryStore,
)

HASH_A = "a" * 64
HASH_B = "b" * 64
TOP3 = [
    {"food": "pizza", "confidence": 90.0},
    {"food": "samosa", "confidence": 6.0},
    {"food": "burger", "confidence": 1.0},
]


class HistoryStoreTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self._tmp.name) / "data"
        self.store = HistoryStore(self.data_dir)
        self.store.initialize()

    def tearDown(self):
        self._tmp.cleanup()

    def raw(self):
        # sqlite3's own context manager commits but does not close.
        return closing(sqlite3.connect(self.store.db_path, isolation_level=None))

    def add(self, predicted="pizza", image_hash=HASH_A, filename="meal.jpg"):
        return self.store.add_prediction(
            filename=filename,
            image_sha256=image_hash,
            predicted_food=predicted,
            confidence=90.0,
            top_predictions=[{"food": predicted, "confidence": 90.0}, *TOP3[1:]],
            device="cpu",
            model_name="EfficientNet-B0",
        )

    # ---- initialization ---------------------------------------------
    def test_initialize_creates_database_schema_and_folders(self):
        self.assertTrue(self.store.db_path.is_file())
        self.assertTrue(self.store.thumbnail_dir.is_dir())
        with self.raw() as conn:
            self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], SCHEMA_VERSION)
            columns = {row[1] for row in conn.execute("PRAGMA table_info(predictions)")}
        self.assertTrue({
            "id", "created_at", "original_filename", "image_sha256", "predicted_food",
            "confidence", "top_predictions", "device", "verification_status",
            "actual_food", "verified_at",
        } <= columns)

    def test_initialize_is_idempotent_and_keeps_data(self):
        record = self.add()
        HistoryStore(self.data_dir).initialize()
        self.assertIsNotNone(HistoryStore(self.data_dir).get_prediction(record["id"]))

    def test_rejects_newer_schema_version(self):
        with self.raw() as conn:
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
        with self.assertRaises(RuntimeError):
            self.store.initialize()

    # ---- saving and reading ---------------------------------------------
    def test_empty_history(self):
        self.assertEqual(self.store.list_predictions(limit=20, offset=0), ([], 0))
        metrics = self.store.live_metrics()
        self.assertEqual(metrics["total_predictions"], 0)
        self.assertEqual(metrics["verified"], 0)
        self.assertIsNone(metrics["accuracy"])
        self.assertEqual(metrics["per_class"], [])
        self.assertIsNone(metrics["most_predicted"])

    def test_add_prediction_round_trip(self):
        record = self.add()
        self.assertRegex(record["id"], r"^[0-9a-f]{32}$")
        self.assertEqual(record["original_filename"], "meal.jpg")
        self.assertEqual(record["image_sha256"], HASH_A)
        self.assertEqual(record["predicted_food"], "pizza")
        self.assertEqual(record["confidence"], 90.0)
        self.assertEqual(record["top_predictions"][0], {"food": "pizza", "confidence": 90.0})
        self.assertEqual(record["verification_status"], "unverified")
        self.assertIsNone(record["actual_food"])
        self.assertFalse(record["thumbnail_available"])

    def test_list_is_newest_first_and_paginated(self):
        ids = [self.add(filename=f"{i}.jpg")["id"] for i in range(5)]
        page1, total = self.store.list_predictions(limit=2, offset=0)
        page3, _ = self.store.list_predictions(limit=2, offset=4)
        self.assertEqual(total, 5)
        self.assertEqual([r["id"] for r in page1], [ids[4], ids[3]])
        self.assertEqual([r["id"] for r in page3], [ids[0]])
        self.assertEqual(self.store.list_predictions(limit=2, offset=10), ([], 5))

    def test_search_matches_food_names_with_spaces_and_filenames(self):
        self.add(predicted="kadai_paneer", filename="dinner.jpg")
        self.add(predicted="pizza", filename="lunch.png")
        self.assertEqual(self.store.list_predictions(limit=20, offset=0, query="kadai paneer")[1], 1)
        self.assertEqual(self.store.list_predictions(limit=20, offset=0, query="LUNCH")[1], 1)
        self.assertEqual(self.store.list_predictions(limit=20, offset=0, query="100%")[1], 0)

    # ---- feedback ---------------------------------------------------------
    def test_feedback_correct_incorrect_update_and_clear(self):
        record = self.add(predicted="pizza")
        correct = self.store.set_feedback(record["id"], "pizza")
        self.assertEqual((correct["verification_status"], correct["actual_food"]), ("correct", "pizza"))
        self.assertIsNotNone(correct["verified_at"])

        updated = self.store.set_feedback(record["id"], "samosa")
        self.assertEqual((updated["verification_status"], updated["actual_food"]), ("incorrect", "samosa"))

        cleared = self.store.clear_feedback(record["id"])
        self.assertEqual(
            (cleared["verification_status"], cleared["actual_food"], cleared["verified_at"]),
            ("unverified", None, None),
        )

    def test_feedback_on_missing_record_returns_none(self):
        self.assertIsNone(self.store.set_feedback("0" * 32, "pizza"))
        self.assertIsNone(self.store.clear_feedback("0" * 32))

    # ---- constraints ----------------------------------------------------------
    def test_database_constraints_reject_inconsistent_rows(self):
        record = self.add(predicted="pizza")
        bad_updates = [
            "UPDATE predictions SET confidence = 101 WHERE id = ?",
            "UPDATE predictions SET verification_status = 'maybe' WHERE id = ?",
            # 'correct' must match the prediction and carry a timestamp.
            "UPDATE predictions SET verification_status = 'correct', actual_food = 'samosa', verified_at = 'x' WHERE id = ?",
            "UPDATE predictions SET verification_status = 'incorrect', actual_food = 'pizza', verified_at = 'x' WHERE id = ?",
            "UPDATE predictions SET actual_food = 'samosa' WHERE id = ?",
            "UPDATE predictions SET top_predictions = 'not json' WHERE id = ?",
        ]
        for sql in bad_updates:
            with self.subTest(sql=sql), self.raw() as conn:
                with self.assertRaises(sqlite3.IntegrityError):
                    conn.execute(sql, (record["id"],))

    # ---- thumbnails and deletion ------------------------------------------
    def test_thumbnail_paths_come_only_from_valid_hashes(self):
        for bad in ["../../etc/passwd", "A" * 64, "a" * 63, "a" * 63 + "/"]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.store.thumbnail_path(bad)

    def test_delete_removes_record_and_unused_thumbnail_only(self):
        self.store.save_thumbnail(HASH_A, b"jpeg-bytes")
        first = self.add(image_hash=HASH_A)
        second = self.add(image_hash=HASH_A)
        self.assertTrue(first["thumbnail_available"])

        self.assertTrue(self.store.delete_prediction(first["id"]))
        self.assertIsNone(self.store.get_prediction(first["id"]))
        self.assertTrue(self.store.thumbnail_path(HASH_A).exists(), "still used by second record")

        self.assertTrue(self.store.delete_prediction(second["id"]))
        self.assertFalse(self.store.thumbnail_path(HASH_A).exists())
        self.assertFalse(self.store.delete_prediction(second["id"]))

    # ---- live metrics -------------------------------------------------------
    def test_live_metrics_known_case(self):
        # 2x pizza correct, pizza predicted but actually samosa,
        # samosa correct, one unverified burger.
        for _ in range(2):
            self.store.set_feedback(self.add(predicted="pizza")["id"], "pizza")
        self.store.set_feedback(self.add(predicted="pizza")["id"], "samosa")
        self.store.set_feedback(self.add(predicted="samosa", image_hash=HASH_B)["id"], "samosa")
        self.add(predicted="burger")

        metrics = self.store.live_metrics()
        self.assertEqual(metrics["total_predictions"], 5)
        self.assertEqual((metrics["verified"], metrics["correct"], metrics["incorrect"]), (4, 3, 1))
        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertEqual(metrics["most_predicted"], {"food": "pizza", "count": 3})
        per_class = {row["food"]: row for row in metrics["per_class"]}
        self.assertEqual(per_class["pizza"], {"food": "pizza", "actual": 2, "predicted": 3, "correct": 2})
        self.assertEqual(per_class["samosa"], {"food": "samosa", "actual": 2, "predicted": 1, "correct": 1})
        self.assertNotIn("burger", per_class, "unverified predictions are excluded")

    def test_data_survives_a_new_store_instance(self):
        record = self.add()
        self.store.set_feedback(record["id"], "samosa")
        reopened = HistoryStore(self.data_dir)
        reopened.initialize()
        again = reopened.get_prediction(record["id"])
        self.assertEqual((again["verification_status"], again["actual_food"]), ("incorrect", "samosa"))


if __name__ == "__main__":
    unittest.main()
