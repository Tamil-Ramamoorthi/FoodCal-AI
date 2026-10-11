"""Tests for the image intake workflow (evaluation/intake.py). Every dataset
is built in a temporary directory; the real data/ folders are not touched.

Run from the project root:
    .venv-gpu\\Scripts\\python.exe -B -m unittest tests.test_intake -v
"""
import csv
import hashlib
import shutil
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.intake import (  # noqa: E402
    AXES,
    COLUMNS,
    INTAKE_FILE,
    init_batch,
    read_intake,
    sync_batch,
    validate_batch,
    validation_markdown,
    write_intake,
)

CLASSES = ["idli", "kulfi"]
TODAY = date(2026, 10, 11)


def food_like(seed, size=(640, 512)):
    """Smooth random colour layout plus fine texture: stable dHash under resizing, not blurred."""
    rng = np.random.default_rng(seed)
    layout = Image.fromarray(rng.integers(0, 256, (6, 6, 3), dtype=np.uint8)).resize(size, Image.Resampling.BICUBIC)
    texture = rng.normal(0, 12, (size[1], size[0], 3))
    return Image.fromarray(np.clip(np.asarray(layout, dtype=np.float64) + texture, 0, 255).astype(np.uint8))


def save(image, path, quality=92, size=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    (image.resize(size, Image.Resampling.LANCZOS) if size else image).save(path, "JPEG", quality=quality)
    return path


def row(file, label, **overrides):
    values = {"file": file, "proposed_label": label, "source": "own phone camera", "license": "own",
              "captured_by": "tester", "capture_date": "2026-10-01", "session_id": "s1",
              **{axis: allowed[0] for axis, allowed in AXES.items()},
              "reviewer": "tester", "decision": "accept", "notes": ""}
    values.update(overrides)
    return values


def tree_digest(root):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(root).rglob("*")) if p.is_file()}


class InitAndSyncTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_init_creates_class_folders_and_empty_csv_and_refuses_to_overwrite(self):
        batch = init_batch("2026-10-15_phone", CLASSES, self.tmp)
        self.assertEqual(sorted(p.name for p in (batch / "images").iterdir()), CLASSES)
        rows, fields, errors = read_intake(batch / INTAKE_FILE)
        self.assertEqual((rows, fields, errors), ([], COLUMNS, []))
        with self.assertRaises(FileExistsError):
            init_batch("2026-10-15_phone", CLASSES, self.tmp)
        with self.assertRaises(ValueError):
            init_batch("../escape", CLASSES, self.tmp)

    def test_sync_appends_new_files_and_keeps_existing_edits_and_columns(self):
        batch = init_batch("b", CLASSES, self.tmp)
        save(food_like(1), batch / "images" / "idli" / "one.jpg")
        (batch / "images" / "idli" / "notes.txt").write_text("x")
        added, errors, warnings = sync_batch(batch, CLASSES)
        self.assertEqual((added, errors), (["idli/one.jpg"], []))
        self.assertEqual(len(warnings), 1, "unsupported file reported, not listed")

        rows, fields, _ = read_intake(batch / INTAKE_FILE)
        rows[0]["source"] = "own phone camera"
        rows[0]["my_column"] = "kept"
        write_intake(batch / INTAKE_FILE, rows, fields + ["my_column"])

        save(food_like(2), batch / "images" / "kulfi" / "two.jpg")
        added, _, _ = sync_batch(batch, CLASSES)
        self.assertEqual(added, ["kulfi/two.jpg"])
        rows, fields, _ = read_intake(batch / INTAKE_FILE)
        self.assertIn("my_column", fields)
        self.assertEqual([(r["file"], r["proposed_label"], r["source"], r["my_column"]) for r in rows],
                         [("idli/one.jpg", "idli", "own phone camera", "kept"), ("kulfi/two.jpg", "kulfi", "", "")])
        self.assertEqual(rows[1]["decision"], "", "sync never fills in a review decision")
        self.assertEqual(sync_batch(batch, CLASSES)[0], [], "a second sync adds nothing")


class ValidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        t = cls.tmp
        # Existing data. Processed images are 224 px re-encodes, like data/processed.
        save(food_like(101), t / "processed" / "train" / "idli" / "000000.jpg", size=(224, 224))
        save(food_like(102), t / "processed" / "validation" / "idli" / "000000.jpg", size=(224, 224))
        test_copy = save(food_like(103), t / "processed" / "test" / "kulfi" / "000000.jpg", size=(224, 224))
        save(food_like(104), t / "real_world" / "idli" / "mine.jpg", size=(800, 600))
        save(food_like(105), t / "intake" / "older" / "images" / "kulfi" / "old.jpg")

        cls.batch = batch = t / "intake" / "new"
        img = batch / "images"
        rows = []

        def add(name, label, image=None, **overrides):
            if image is not None:
                save(image, img / name)
            rows.append(row(name, label, **overrides))

        add("idli/good.jpg", "idli", food_like(1))
        add("idli/pending.jpg", "idli", food_like(2), decision="", reviewer="")
        add("kulfi/rejected.jpg", "kulfi", food_like(3), decision="reject")
        add("idli/relabelled.jpg", "idli", food_like(4), decision="relabel:kulfi")
        add("idli/near_validation.jpg", "idli", food_like(102))
        add("idli/near_train.jpg", "idli", food_like(101))
        add("idli/near_real_world.jpg", "idli", food_like(104))
        shutil.copy(test_copy, img / "kulfi" / "exact_test.jpg")
        add("kulfi/exact_test.jpg", "kulfi")
        add("kulfi/near_older_batch.jpg", "kulfi", food_like(105).resize((600, 480)))
        add("idli/small.jpg", "idli", food_like(5, size=(300, 300)))
        add("kulfi/wrong_folder.jpg", "idli", food_like(6))
        add("idli/bad_fields.jpg", "idli", food_like(7), garnish="parsley", capture_date="12-10-2026",
            decision="relabel:Idli", license="")
        add("idli/no_reviewer.jpg", "idli", food_like(8), reviewer="")
        add("idli/future.jpg", "idli", food_like(9), capture_date="2027-01-01")
        (img / "idli" / "corrupt.jpg").write_bytes(b"not an image")
        add("idli/corrupt.jpg", "idli")
        add("idli/missing.jpg", "idli")
        add("kulfi/dup_a.jpg", "kulfi", food_like(10))
        shutil.copy(img / "kulfi" / "dup_a.jpg", img / "kulfi" / "dup_b.jpg")
        add("kulfi/dup_b.jpg", "kulfi")
        add("idli/session_a.jpg", "idli", food_like(11), session_id="A")
        save(food_like(11).resize((600, 480)), img / "idli" / "session_b.jpg", quality=75)
        add("idli/session_b.jpg", "idli", session_id="B")
        add("idli/flat.jpg", "idli", Image.new("RGB", (640, 512), (128, 128, 128)), serving="")
        save(food_like(12), img / "idli" / "unlisted.jpg")
        write_intake(batch / INTAKE_FILE, rows, COLUMNS)

        cls.before = tree_digest(t)

        def fake_hint(paths):
            return [[("kulfi", 0.9), ("idli", 0.05), ("chai", 0.01)] for _ in paths]

        cls.report = validate_batch(batch, CLASSES, processed_dir=t / "processed", real_world_dir=t / "real_world",
                                    cache_path=None, today=TODAY, hint=("fake", fake_hint))
        cls.by_file = {r["file"]: r for r in cls.report["images"]}

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def errors(self, name):
        return " | ".join(self.by_file[name]["errors"])

    def warnings(self, name):
        return " | ".join(self.by_file[name]["warnings"])

    def test_nothing_on_disk_is_changed(self):
        self.assertEqual(tree_digest(self.tmp), self.before)

    def test_review_statuses(self):
        self.assertEqual(self.by_file["idli/good.jpg"]["status"], "ready")
        self.assertEqual(self.by_file["idli/pending.jpg"]["status"], "pending_review")
        self.assertEqual(self.by_file["kulfi/rejected.jpg"]["status"], "rejected")
        relabelled = self.by_file["idli/relabelled.jpg"]
        self.assertEqual((relabelled["status"], relabelled["label"], relabelled["proposed_label"]),
                         ("ready", "kulfi", "idli"))

    def test_held_out_images_are_blocked_even_as_near_duplicates(self):
        self.assertIn("near-duplicate of processed/validation", self.errors("idli/near_validation.jpg"))
        self.assertIn("near-duplicate of real_world_test", self.errors("idli/near_real_world.jpg"))
        self.assertIn("exact copy of processed/test", self.errors("kulfi/exact_test.jpg"))

    def test_near_copies_of_training_data_and_older_batches_are_warnings(self):
        for name, source in (("idli/near_train.jpg", "processed/train"),
                             ("kulfi/near_older_batch.jpg", "intake/older")):
            with self.subTest(name=name):
                self.assertEqual(self.by_file[name]["errors"], [])
                self.assertIn(f"near-duplicate of {source}", self.warnings(name))

    def test_image_problems(self):
        self.assertIn("too small: 300x300", self.errors("idli/small.jpg"))
        self.assertIn("cannot be decoded", self.errors("idli/corrupt.jpg"))
        self.assertIn("file not found", self.errors("idli/missing.jpg"))
        flat = self.warnings("idli/flat.jpg")
        self.assertIn("possibly blurred", flat)
        self.assertIn("very low contrast", flat)
        self.assertIn("untagged: serving", flat)
        self.assertEqual(self.by_file["idli/flat.jpg"]["status"], "ready", "quality issues are warnings only")

    def test_label_provenance_and_decision_problems(self):
        self.assertIn("does not match its folder `kulfi`", self.errors("kulfi/wrong_folder.jpg"))
        bad = self.errors("idli/bad_fields.jpg")
        for expected in ("garnish `parsley`", "must be YYYY-MM-DD", "did you mean `idli`", "`license` is empty"):
            self.assertIn(expected, bad)
        self.assertIn("`reviewer` is empty", self.errors("idli/no_reviewer.jpg"))
        self.assertIn("in the future", self.errors("idli/future.jpg"))

    def test_duplicates_within_the_batch(self):
        self.assertIn("exact duplicate of `kulfi/dup_b.jpg`", self.errors("kulfi/dup_a.jpg"))
        self.assertIn("exact duplicate of `kulfi/dup_a.jpg`", self.errors("kulfi/dup_b.jpg"))
        self.assertIn("different session_id", self.warnings("idli/session_b.jpg"))

    def test_unlisted_files_are_a_batch_error(self):
        self.assertTrue(any("unlisted.jpg" in e for e in self.report["batch_errors"]))

    def test_model_hint_warns_but_never_changes_a_label(self):
        good = self.by_file["idli/good.jpg"]
        self.assertEqual((good["label"], good["status"]), ("idli", "ready"))
        self.assertIn("model hint disagrees: kulfi", self.warnings("idli/good.jpg"))
        self.assertNotIn("model hint disagrees", self.warnings("idli/relabelled.jpg"), "agrees with kulfi")

    def test_summary_coverage_and_markdown(self):
        s = self.report["summary"]
        self.assertEqual(s["rows"], len(self.by_file))
        self.assertEqual(s["ready"] + s["pending_review"] + s["rejected"] + s["with_errors"], s["rows"])
        self.assertEqual(self.report["coverage"]["kulfi"]["ready"], 2)  # relabelled + near_older_batch
        self.assertNotIn("rejected", str(self.report["coverage"]))
        self.assertIn("# Intake validation", validation_markdown(self.report))


if __name__ == "__main__":
    unittest.main()
