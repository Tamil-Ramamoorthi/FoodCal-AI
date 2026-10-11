"""Tests for the human-in-the-loop feedback pipeline:
feedback export -> review in an intake batch -> candidate manifest ->
candidate evaluation and promotion gate.

Every database, batch and manifest is built in a temporary directory; the
real storage/, data/ and models/ folders are only read.

Run from the project root:
    .venv-gpu\\Scripts\\python.exe -B -m unittest tests.test_feedback_pipeline -v
"""
import argparse
import hashlib
import io
import json
import shutil
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.database.history_store import HistoryStore  # noqa: E402
from evaluation import compare_models  # noqa: E402
from evaluation.feedback_export import FEEDBACK_MIN_SHORT_SIDE, export_feedback  # noqa: E402
from evaluation.intake import INTAKE_FILE, read_intake, validate_batch, write_intake  # noqa: E402
from tests.test_intake import food_like, save  # noqa: E402
from training import build_feedback_manifest  # noqa: E402
from training.build_manifests import read_manifest, write_manifest  # noqa: E402

CLASSES = ["chapati", "idli", "kulfi", "momos"]


def thumbnail_bytes(image):
    image = image.copy()
    image.thumbnail((256, 256), Image.Resampling.LANCZOS)  # as backend make_thumbnail
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=85)
    return buffer.getvalue()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FeedbackPipelineTests(unittest.TestCase):
    """One scenario, run in order: export, re-export, review, build."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = t = Path(tempfile.mkdtemp())
        # Existing dataset: processed splits + v2-style base manifests.
        for split, seed, label in (("train", 101, "idli"), ("validation", 102, "kulfi"), ("test", 103, "momos")):
            save(food_like(seed), t / "processed" / split / label / "000000.jpg", size=(224, 224))
        base = t / "manifests" / "v2"
        for name, split, label in (("train.csv", "train", "idli"), ("validation.csv", "validation", "kulfi"),
                                   ("test_clean.csv", "test", "momos")):
            path = t / "processed" / split / label / "000000.jpg"
            write_manifest(base / name, [{"path": f"data/processed/{split}/{label}/000000.jpg", "label": label,
                                          "class_index": CLASSES.index(label), "sha256": file_sha(path)}])
        (t / "real_world").mkdir()
        cls.base = base

        # Prediction history with feedback, written through the app's own store.
        cls.store = store = HistoryStore(t / "storage")
        store.initialize()
        cls.ids = {}

        def predict(key, image, predicted, actual=None, keep_thumbnail=True):
            data = thumbnail_bytes(image)
            sha = hashlib.sha256(data).hexdigest()  # same image -> same hash, like repeated uploads
            store.save_thumbnail(sha, data)
            record = store.add_prediction(filename=f"{key}.jpg", image_sha256=sha, predicted_food=predicted,
                                          confidence=31.5, top_predictions=[{"food": predicted, "confidence": 31.5}],
                                          device="cpu", model_name="EfficientNet-B0")
            if actual:
                store.set_feedback(record["id"], actual)
            if not keep_thumbnail:
                store.thumbnail_path(sha).unlink()
            cls.ids.setdefault(key, []).append(record["id"])
            return sha

        cls.sha_a = predict("A", food_like(1), "chapati", "momos")       # incorrect
        predict("A", food_like(1), "chapati", "momos")                   # same image, same label -> merged
        predict("A", food_like(1), "chapati")                            # same image, unverified
        cls.sha_b = predict("B", food_like(2), "kulfi", "kulfi")         # confirmed correct
        predict("C", food_like(3), "idli", "idli")                       # conflicting labels
        predict("C", food_like(3), "idli", "kulfi")
        predict("D", food_like(4), "chapati")                            # unverified only
        predict("E", food_like(5), "idli", "kulfi", keep_thumbnail=False)  # thumbnail gone
        predict("F", food_like(6), "idli", "samosa")                     # not one of CLASSES
        cls.sha_g = predict("G", food_like(103), "chapati", "momos")     # near copy of a TEST image

        cls.db = t / "storage" / "foodcal.db"
        cls.db_sha = file_sha(cls.db)
        cls.thumbs = {p.name: file_sha(p) for p in (t / "storage" / "thumbnails").iterdir()}
        cls.intake = t / "intake"
        cls.report = export_feedback(CLASSES, t / "storage", cls.intake, batch_id="fb1")
        cls.batch = cls.intake / "fb1"

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def rows(self):
        return {r["prediction_id"]: r for r in read_intake(self.batch / INTAKE_FILE)[0]}

    def validate(self):
        return validate_batch(self.batch, CLASSES, processed_dir=self.tmp / "processed",
                              real_world_dir=self.tmp / "real_world", cache_path=None)

    def build(self, name):
        return build_feedback_manifest.build(["fb1"], name, CLASSES, base_dir=self.base,
                                             manifests_root=self.tmp / "manifests", intake_dir=self.intake,
                                             processed_dir=self.tmp / "processed",
                                             real_world_dir=self.tmp / "real_world", cache_path=None)

    # --- export -------------------------------------------------------

    def test_01_export_never_changes_the_database_or_thumbnails(self):
        self.assertEqual(file_sha(self.db), self.db_sha)
        self.assertEqual({p.name: file_sha(p) for p in (self.tmp / "storage" / "thumbnails").iterdir()}, self.thumbs)
        with self.assertRaises(sqlite3.OperationalError):  # the connection really is read-only
            conn = sqlite3.connect(f"{self.db.resolve().as_uri()}?mode=ro", uri=True)
            try:
                conn.execute("DELETE FROM predictions")
            finally:
                conn.close()

    def test_02_export_contents_dedup_conflicts_and_skips(self):
        r = self.report
        self.assertEqual(r["verified_predictions_read"], 8)  # A x2, B, C x2, E, F, G; unverified ones are not read
        self.assertEqual(r["exported"], 3)
        rows = self.rows()
        a = rows[self.ids["A"][1]]  # the most recently verified prediction of image A
        self.assertEqual((a["file"], a["proposed_label"], a["predicted_food"], a["user_verdict"]),
                         (f"momos/{self.ids['A'][1]}.jpg", "momos", "chapati", "incorrect"))
        self.assertEqual(a["all_prediction_ids"].split(), self.ids["A"][:2])
        self.assertEqual(rows[self.ids["B"][0]]["user_verdict"], "correct")
        self.assertEqual(len(r["conflicts"]), 1)
        self.assertEqual(r["conflicts"][0]["labels"], ["idli", "kulfi"])
        self.assertEqual([x["label"] for x in r["invalid_label"]], ["samosa"])
        self.assertEqual(len(r["missing_thumbnail"]), 1)

    def test_03_exported_rows_are_unreviewed_with_provenance(self):
        for row in self.rows().values():
            self.assertEqual((row["decision"], row["reviewer"]), ("", ""))
            self.assertTrue(row["source"] and row["license"] and row["session_id"].startswith("img-"))
        image = self.batch / "images" / f"momos/{self.ids['A'][1]}.jpg"
        self.assertEqual(file_sha(image), self.sha_a, "byte copy of the stored thumbnail")
        settings = json.loads((self.batch / "batch.json").read_text(encoding="utf-8"))
        self.assertEqual((settings["kind"], settings["min_short_side"]), ("user_feedback", FEEDBACK_MIN_SHORT_SIDE))

    def test_04_re_export_creates_no_duplicates(self):
        again = export_feedback(CLASSES, self.tmp / "storage", self.intake, batch_id="fb2")
        self.assertEqual((again["exported"], again["batch"], len(again["already_exported"])), (0, None, 3))
        self.assertFalse((self.intake / "fb2").exists())

    def test_05_incorrect_only(self):
        other = self.tmp / "intake_incorrect_only"
        report = export_feedback(CLASSES, self.tmp / "storage", other, batch_id="x", incorrect_only=True)
        verdicts = {r["user_verdict"] for r in report["exported_rows"]}
        self.assertEqual((report["exported"], verdicts), (2, {"incorrect"}))  # A and G; B was confirmed correct
        # C has a 'correct: idli' and an 'incorrect: kulfi' verdict; that is still a conflict.
        self.assertEqual([c["labels"] for c in report["conflicts"]], [["idli", "kulfi"]])

    # --- review gate --------------------------------------------------

    def test_06_unreviewed_feedback_is_never_ready_or_used(self):
        report = self.validate()
        self.assertEqual(report["summary"]["ready"], 0)
        g = next(i for i in report["images"] if i["file"].endswith(f"{self.ids['G'][0]}.jpg"))
        self.assertIn("near-duplicate of processed/test", " ".join(g["errors"]))
        self.assertEqual(self.build("unreviewed")["written"], False)
        self.assertFalse((self.tmp / "manifests" / "unreviewed").exists())

    def test_07_only_approved_rows_reach_train_and_held_out_sets_are_unchanged(self):
        rows, fields, _ = read_intake(self.batch / INTAKE_FILE)
        decisions = {self.ids["A"][1]: "accept", self.ids["B"][0]: "reject", self.ids["G"][0]: "accept"}
        for row in rows:
            row["reviewer"], row["decision"] = "tester", decisions[row["prediction_id"]]
        write_intake(self.batch / INTAKE_FILE, rows, fields)

        result = self.build("cand1")
        self.assertTrue(result["written"])
        out = self.tmp / "manifests" / "cand1"
        train = read_manifest(out / "train.csv")
        self.assertEqual(len(train), 2)  # 1 base + only image A
        self.assertEqual(train[-1]["label"], "momos")
        self.assertEqual(train[-1]["class_index"], CLASSES.index("momos"))
        self.assertTrue(train[-1]["path"].endswith(f"momos/{self.ids['A'][1]}.jpg"))
        for split in ("validation.csv", "test_clean.csv"):
            self.assertEqual(file_sha(out / split), file_sha(self.base / split), f"{split} copied unchanged")
        left = {(x["file"].split("/")[1][:32], x["status"]) for x in result["left_out"]}
        self.assertEqual(left, {(self.ids["B"][0], "rejected"), (self.ids["G"][0], "error")})
        self.assertEqual((result["affected_classes"], result["correction_pairs"]),
                         (["chapati", "momos"], [["momos", "chapati"]]))
        with self.assertRaises(FileExistsError):
            self.build("cand1")

    def test_08_leakage_assertion_is_independent_of_the_validator(self):
        # Even if validation were wrong and called a held-out copy "ready",
        # the builder refuses to write the manifest.
        test_image = self.tmp / "processed" / "test" / "momos" / "000000.jpg"
        fake = {"summary": {}, "images": [{
            "file": "momos/leak.jpg", "status": "ready", "label": "momos", "proposed_label": "momos",
            "decision": "accept", "sha256": file_sha(test_image), "dhash": None}]}
        with mock.patch.object(build_feedback_manifest, "validate_batch", return_value=fake):
            with self.assertRaisesRegex(AssertionError, "exact copy of a validation / test"):
                self.build("leak")
        self.assertFalse((self.tmp / "manifests" / "leak").exists())


class FeedbackStorageTests(unittest.TestCase):
    def test_repeated_feedback_updates_one_record_instead_of_duplicating(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = HistoryStore(tmp)
            store.initialize()
            record = store.add_prediction(filename="x.jpg", image_sha256="a" * 64, predicted_food="chapati",
                                          confidence=40.0, top_predictions=[], device="cpu", model_name="m")
            store.set_feedback(record["id"], "momos")
            updated = store.set_feedback(record["id"], "idli")
            items, total = store.list_predictions(limit=10, offset=0)
            self.assertEqual((total, updated["actual_food"], updated["verification_status"]), (1, "idli", "incorrect"))


def fake_report(old_f1, new_f1, ci, fixed, broken, p, per_class):
    return {"overall": {"old": {"macro_f1": old_f1}, "new": {"macro_f1": new_f1}},
            "paired_test": {"old_right_new_wrong": broken, "old_wrong_new_right": fixed, "p_value": p},
            "bootstrap": {"macro_f1_difference_95ci": ci},
            "per_class": [{"food": f, "support": s, "old": {"f1": o}, "new": {"f1": n}} for f, s, o, n in per_class]}


class PromotionGateTests(unittest.TestCase):
    def test_clear_improvement_passes(self):
        gate = compare_models.promotion_gate(fake_report(0.80, 0.85, [0.02, 0.07], 50, 10, 1e-6,
                                                         [("a", 100, 0.8, 0.85), ("b", 10, 0.9, 0.5)]))
        self.assertEqual((gate["passed"], gate["failures"]), (True, []), "class b has < 20 images, ignored")

    def test_uncertain_or_harmful_changes_fail(self):
        cases = {
            "includes 0": fake_report(0.80, 0.81, [-0.01, 0.03], 50, 10, 1e-6, []),
            "McNemar": fake_report(0.80, 0.85, [0.02, 0.07], 12, 10, 0.8, []),
            "did not improve": fake_report(0.85, 0.85, [0.01, 0.02], 50, 10, 1e-6, []),
            "kulfi F1 dropped": fake_report(0.80, 0.85, [0.02, 0.07], 50, 10, 1e-6, [("kulfi", 28, 0.9, 0.8)]),
        }
        for expected, report in cases.items():
            with self.subTest(expected=expected):
                gate = compare_models.promotion_gate(report)
                self.assertFalse(gate["passed"])
                self.assertIn(expected, " ".join(gate["failures"]))


class CandidateEvaluationTests(unittest.TestCase):
    """Runs the real comparison (production vs v2 checkpoint) on a small
    manifest of real clean-test images, on CPU, with a feedback summary."""

    def test_comparison_report_has_metrics_gate_and_feedback_focus(self):
        clean = read_manifest(ROOT / "data" / "manifests" / "v2" / "test_clean.csv")
        first = {}
        for row in clean:
            first.setdefault(row["label"], row)  # one image of each of the 34 classes
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_manifest(tmp / "small.csv", list(first.values()))
            (tmp / "summary.json").write_text(json.dumps({
                "train_added": 1, "affected_classes": ["chapati", "momos"],
                "correction_pairs": [["momos", "chapati"]]}), encoding="utf-8")
            args = argparse.Namespace(old=compare_models.DEFAULT_OLD, new=compare_models.DEFAULT_NEW,
                                      manifest=tmp / "small.csv", output_dir=tmp / "out", device="cpu",
                                      feedback_summary=tmp / "summary.json")
            production_sha = file_sha(compare_models.DEFAULT_OLD)
            report = compare_models.run(args)
            self.assertEqual(file_sha(compare_models.DEFAULT_OLD), production_sha, "production checkpoint unchanged")

        self.assertEqual(report["test_images"], 34)
        for tag in ("old", "new"):
            o = report["overall"][tag]
            self.assertEqual(o["correct"], round(o["accuracy"] * 34))
            for key in ("macro_f1", "weighted_f1"):
                self.assertTrue(0 <= o[key] <= 1)
        gate = report["promotion_gate"]
        self.assertEqual(gate["passed"], not gate["failures"])
        focus = report["feedback_focus"]
        names = report["confusion_matrices"]["labels"]
        i, j = names.index("momos"), names.index("chapati")
        self.assertEqual(focus["correction_pairs"], [{
            "actual": "momos", "predicted": "chapati", "actual_images": 1,
            "old": report["confusion_matrices"]["old"][i][j], "new": report["confusion_matrices"]["new"][i][j]}])
        self.assertEqual([c["food"] for c in focus["classes"]], ["chapati", "momos"])
        self.assertIn("## Promotion gate", compare_models.markdown(report))


if __name__ == "__main__":
    unittest.main()
