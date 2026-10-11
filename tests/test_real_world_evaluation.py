"""Tests for the real-world dataset audit and evaluation. All datasets are
built in temporary directories; the real data/ folders are only read.

Run from the project root:
    .venv-gpu\\Scripts\\python.exe -B -m unittest discover -s tests -v
"""
import csv
import inspect
import io
import json
import shutil
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.models.classifier import (  # noqa: E402
    build_eval_transform,
    load_classifier,
    preprocess_image,
)
from evaluation.dataset_audit import DEFAULT_NEAR_DISTANCE, audit_real_world, scan_real_world  # noqa: E402
from evaluation.evaluate_real_world import run_evaluation  # noqa: E402
from evaluation.hashing import build_reference_index, dhash  # noqa: E402
from evaluation.metrics import classification_metrics, confidence_summary  # noqa: E402

CHECKPOINT = ROOT / "models" / "efficientnet_b0" / "food_efficientnet_b0_best.pth"
PROCESSED = ROOT / "data" / "processed"
CPU = torch.device("cpu")


def test_image(food, name="000000.jpg"):
    return PROCESSED / "test" / food / name


def synthetic_image(seed, size=(320, 240)):
    rng = np.random.RandomState(seed)
    base = rng.randint(0, 255, (6, 8, 3)).astype(np.uint8)
    return Image.fromarray(base).resize(size, Image.Resampling.BICUBIC)


class ClassifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.classifier = load_classifier(CHECKPOINT, CPU)

    def test_checkpoint_loads_with_34_classes(self):
        self.assertEqual(len(self.classifier.class_names), 34)
        self.assertEqual(self.classifier.image_size, 224)
        self.assertEqual(self.classifier.model.classifier[1].out_features, 34)
        self.assertFalse(self.classifier.model.training)

    def test_class_mapping_matches_dataset_folders(self):
        for split in ("train", "validation", "test"):
            with self.subTest(split=split):
                self.assertEqual(sorted(p.name for p in (PROCESSED / split).iterdir()), self.classifier.class_names)
        real_world = sorted(p.name for p in (ROOT / "data" / "real_world_test").iterdir() if p.is_dir())
        self.assertEqual(real_world, self.classifier.class_names)

    def test_shared_preprocessing_matches_training_pipeline(self):
        with Image.open(test_image("pizza")) as image:
            shared = preprocess_image(image, 224, build_eval_transform(224))
            reference = ImageOps.exif_transpose(image).convert("RGB").resize((224, 224), Image.Resampling.LANCZOS)
        expected = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])(reference)
        self.assertTrue(torch.equal(shared, expected))


class MetricsTests(unittest.TestCase):
    CLASSES = ["a", "b", "c", "d"]

    def test_known_example(self):
        # a: 3 images (2 right, 1 -> b); b: 2 images (1 right, 1 -> a); c: 1 image -> c; d: none.
        y_true = [0, 0, 0, 1, 1, 2]
        y_pred = [0, 0, 1, 1, 0, 2]
        m = classification_metrics(y_true, y_pred, self.CLASSES)
        self.assertEqual((m["total"], m["correct"], m["incorrect"]), (6, 4, 2))
        self.assertAlmostEqual(m["accuracy"], 4 / 6)
        self.assertEqual(m["confusion_matrix"]["matrix"], [[2, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]])
        a, b, c, d = m["per_class"]
        self.assertAlmostEqual(a["precision"], 2 / 3)
        self.assertAlmostEqual(a["recall"], 2 / 3)
        self.assertAlmostEqual(a["f1"], 2 / 3)
        self.assertAlmostEqual(b["precision"], 1 / 2)
        self.assertAlmostEqual(b["recall"], 1 / 2)
        self.assertAlmostEqual(c["f1"], 1.0)
        self.assertEqual(m["macro_f1_classes"], 3)
        self.assertAlmostEqual(m["macro_f1"], (2 / 3 + 1 / 2 + 1) / 3)
        self.assertAlmostEqual(m["weighted_f1"], (2 / 3 * 3 + 1 / 2 * 2 + 1 * 1) / 6)

    def test_missing_classes_are_not_counted_as_success(self):
        m = classification_metrics([0, 0], [0, 0], self.CLASSES)
        d = m["per_class"][3]
        self.assertEqual(d["support"], 0)
        self.assertIsNone(d["recall"])
        self.assertIsNone(d["f1"])
        self.assertIsNone(d["precision"])
        self.assertEqual(m["classes_without_samples"], ["b", "c", "d"])
        self.assertEqual(m["macro_f1_classes"], 1)

    def test_absent_class_predicted_and_present_class_never_predicted(self):
        m = classification_metrics([0, 0], [3, 3], self.CLASSES)
        a, d = m["per_class"][0], m["per_class"][3]
        self.assertIsNone(a["precision"], "never predicted: precision undefined")
        self.assertEqual((a["recall"], a["f1"]), (0.0, 0.0))
        self.assertEqual(d["precision"], 0.0)
        self.assertEqual(m["predicted_but_absent"], ["d"])
        self.assertEqual(m["accuracy"], 0.0)
        self.assertEqual(m["macro_f1"], 0.0)

    def test_empty_input(self):
        m = classification_metrics([], [], self.CLASSES)
        self.assertEqual(m["total"], 0)
        self.assertIsNone(m["accuracy"])
        self.assertIsNone(m["macro_f1"])
        self.assertIsNone(m["weighted_f1"])
        self.assertIsNone(confidence_summary([], []))

    def test_confidence_summary_is_separate_from_accuracy(self):
        s = confidence_summary([0.95, 0.9, 0.35], [False, True, True])
        self.assertAlmostEqual(s["mean_confidence"], (0.95 + 0.9 + 0.35) / 3)
        self.assertAlmostEqual(s["mean_confidence_correct"], (0.9 + 0.35) / 2)
        self.assertAlmostEqual(s["mean_confidence_incorrect"], 0.95)
        top = s["bins"][9]
        self.assertEqual((top["count"], top["accuracy"]), (2, 0.5))


class DatasetFixture(unittest.TestCase):
    """Builds a temporary real-world folder, a tiny 'processed' reference
    set and a tiny source archive."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        self.class_names = load_classifier.__globals__["torch"].load(
            CHECKPOINT, map_location="cpu", weights_only=True)["class_names"]
        self.real = tmp / "real_world_test"
        for name in self.class_names:
            (self.real / name).mkdir(parents=True)
        self.processed = tmp / "processed"
        for split, food in (("train", "pizza"), ("test", "samosa")):
            (self.processed / split / food).mkdir(parents=True)
            shutil.copy(test_image(food), self.processed / split / food / "000000.jpg")
        self.archive = tmp / "source.zip"
        with zipfile.ZipFile(self.archive, "w") as archive:
            archive.write(test_image("burger"), "Food Classification Dataset/burger/original.jpg")
            archive.writestr("__MACOSX/._ignored.jpg", b"junk")
        self.output = tmp / "reports"

    def tearDown(self):
        self._tmp.cleanup()

    def put(self, label, name, source=None, image=None, fmt=None, data=None):
        path = self.real / label / name
        if source is not None:
            shutil.copy(source, path)
        elif image is not None:
            image.save(path, fmt)
        else:
            path.write_bytes(data)
        return path

    def reference(self):
        return build_reference_index(self.processed, self.archive, cache_path=None, progress=lambda *_: None)


class AuditTests(DatasetFixture):
    def test_scan_formats_invalid_unsupported_and_unexpected(self):
        self.put("pizza", "a.jpg", image=synthetic_image(1), fmt="JPEG")
        self.put("pizza", "b.jpeg", image=synthetic_image(2), fmt="JPEG")
        self.put("sushi", "c.png", image=synthetic_image(3), fmt="PNG")
        self.put("momos", "d.webp", image=synthetic_image(4), fmt="WEBP")
        self.put("idli", "broken.jpg", data=b"not an image")
        good = io.BytesIO()
        synthetic_image(5).save(good, "JPEG")
        self.put("idli", "truncated.jpg", data=good.getvalue()[: len(good.getvalue()) // 2])
        self.put("idli", "notes.txt", data=b"text")
        (self.real / "pasta").mkdir()
        (self.real / "loose.jpg").write_bytes(b"x")
        (self.real / "chai" / "nested").mkdir()
        shutil.rmtree(self.real / "kulfi")

        scan = scan_real_world(self.real, self.class_names)
        self.assertEqual(sorted(i["format"] for i in scan["images"]), ["JPEG", "JPEG", "PNG", "WEBP"])
        self.assertEqual(sorted(Path(i["path"]).name for i in scan["invalid_images"]), ["broken.jpg", "truncated.jpg"])
        self.assertEqual([Path(f["path"]).name for f in scan["unsupported_files"]], ["notes.txt"])
        unexpected = {Path(e["path"]).name for e in scan["unexpected_entries"]}
        self.assertEqual(unexpected, {"pasta", "loose.jpg", "nested"})
        self.assertEqual(scan["missing_class_folders"], ["kulfi"])

    def test_duplicates_overlaps_and_limits(self):
        self.put("pizza", "train_copy.jpg", source=PROCESSED / "test" / "pizza" / "000000.jpg")  # == processed/train copy
        self.put("burger", "archive_copy.jpg", source=test_image("burger"))  # == archive original
        with Image.open(test_image("samosa")) as image:  # re-encoded, resized copy of a processed image
            self.put("samosa", "resaved.png", image=image.resize((640, 480), Image.Resampling.LANCZOS), fmt="PNG")
        unique = self.put("sushi", "own_photo.jpg", image=synthetic_image(7), fmt="JPEG")
        shutil.copy(unique, self.real / "momos" / "same_photo_other_label.jpg")

        audit = audit_real_world(self.real, self.class_names, self.reference(), min_per_class=10, near_distance=6)
        exact = {Path(o["path"]).name: o for o in audit["exact_overlaps"]}
        self.assertEqual(set(exact), {"train_copy.jpg", "archive_copy.jpg"})
        self.assertEqual(exact["train_copy.jpg"]["matches"][0]["source"], "processed/train")
        self.assertEqual(exact["archive_copy.jpg"]["matches"][0]["source"], "source-archive")
        near = {Path(o["path"]).name: o for o in audit["near_duplicate_overlaps"]}
        self.assertEqual(set(near), {"resaved.png"}, "re-encoded copy found only by dHash")
        self.assertLessEqual(near["resaved.png"]["matches"][0]["distance"], 6)
        self.assertEqual(len(audit["exact_duplicates_within"]), 1)
        self.assertTrue(audit["exact_duplicates_within"][0]["conflicting_labels"])
        self.assertEqual(audit["independence_status"], "overlap_detected")
        self.assertIn("SHA-256", audit["duplicate_detection_limits"])
        self.assertEqual(audit["classes_below_minimum"], sorted(["burger", "momos", "pizza", "samosa", "sushi"], key=self.class_names.index))
        self.assertEqual(len(audit["classes_without_images"]), 29)

    def test_default_distance_is_shared_and_finds_resaved_copy(self):
        self.assertEqual(DEFAULT_NEAR_DISTANCE, 3)
        self.assertEqual(inspect.signature(run_evaluation).parameters["near_distance"].default, DEFAULT_NEAR_DISTANCE)
        self.assertEqual(inspect.signature(audit_real_world).parameters["near_distance"].default, DEFAULT_NEAR_DISTANCE)
        with Image.open(test_image("samosa")) as image:
            self.put("samosa", "resaved.jpg", image=image.resize((640, 480), Image.Resampling.LANCZOS), fmt="JPEG")
        audit = audit_real_world(self.real, self.class_names, self.reference())
        self.assertEqual(audit["near_duplicate_max_distance"], 3)
        self.assertEqual([Path(o["path"]).name for o in audit["near_duplicate_overlaps"]], ["resaved.jpg"])

    def test_dhash_survives_resize_and_reencode(self):
        with Image.open(test_image("pizza")) as image:
            original = dhash(image)
            buffer = io.BytesIO()
            image.resize((500, 380)).save(buffer, "JPEG", quality=70)
        with Image.open(buffer) as copy:
            self.assertLessEqual(bin(original ^ dhash(copy)).count("1"), 6)
        self.assertGreater(bin(dhash(synthetic_image(1)) ^ dhash(synthetic_image(2))).count("1"), 6)

    def test_provenance_and_independence(self):
        self.put("sushi", "mine1.jpg", image=synthetic_image(11), fmt="JPEG")
        self.put("pizza", "mine2.jpg", image=synthetic_image(12), fmt="JPEG")
        audit = audit_real_world(self.real, self.class_names, self.reference())
        self.assertEqual(audit["independence_status"], "provenance_incomplete")
        self.assertFalse(audit["provenance"]["present"])

        with open(self.real / "provenance.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["file", "source", "captured_by", "date", "license", "notes"])
            writer.writerow(["sushi/mine1.jpg", "own phone camera", "me", "2026-10-10", "own", ""])
            writer.writerow(["pizza/missing.jpg", "own phone camera", "me", "", "", ""])
        audit = audit_real_world(self.real, self.class_names, self.reference())
        self.assertEqual((audit["provenance"]["documented_images"], audit["provenance"]["undocumented_images"]), (1, 1))
        self.assertEqual(audit["provenance"]["entries_without_image"], ["pizza/missing.jpg"])
        self.assertEqual(audit["independence_status"], "provenance_incomplete")

        with open(self.real / "provenance.csv", "a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(["pizza/mine2.jpg", "own phone camera", "me", "", "", ""])
        audit = audit_real_world(self.real, self.class_names, self.reference())
        self.assertEqual(audit["independence_status"], "checks_passed")


class EvaluationReportTests(DatasetFixture):
    def run_eval(self):
        return run_evaluation(
            real_world_dir=self.real, checkpoint_path=CHECKPOINT, output_dir=self.output,
            processed_dir=self.processed, archive_path=self.archive, device=CPU,
            command="test", progress=lambda *_: None,
        )

    def test_empty_dataset_produces_honest_empty_report(self):
        report = self.run_eval()
        self.assertEqual(report["status"], "no_valid_images")
        self.assertIsNone(report["metrics"])
        self.assertIsNone(report["confidence"])
        self.assertEqual(report["dataset"]["valid_images"], 0)
        self.assertEqual(len(report["dataset"]["classes_without_images"]), 34)
        saved = json.loads((self.output / "real_world_evaluation.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["status"], "no_valid_images")
        markdown = (self.output / "real_world_evaluation.md").read_text(encoding="utf-8")
        self.assertIn("no metrics were calculated", markdown)

    def test_end_to_end_json_and_markdown(self):
        # The fixture's processed/train/pizza/000000.jpg and processed/test/samosa/000000.jpg
        # are copies of these, so both are exact overlaps; 000001.jpg is not in the fixture.
        self.put("pizza", "dataset_pizza.jpg", source=test_image("pizza"))
        self.put("pizza", "other_pizza.jpg", source=test_image("pizza", "000001.jpg"))
        self.put("samosa", "samosa.jpg", source=test_image("samosa"))
        for seed, label in ((21, "sushi"), (22, "idli")):
            self.put(label, f"synthetic_{seed}.png", image=synthetic_image(seed), fmt="PNG")
        self.put("idli", "broken.webp", data=b"broken")

        report = self.run_eval()
        saved = json.loads((self.output / "real_world_evaluation.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["status"], "ok")
        for key in ("generated_at", "checkpoint", "dataset", "class_mapping", "metrics", "quality", "predictions"):
            self.assertIn(key, saved)
        self.assertEqual(saved["checkpoint"]["path"], "models/efficientnet_b0/food_efficientnet_b0_best.pth")
        self.assertEqual(len(saved["checkpoint"]["sha256"]), 64)
        self.assertEqual([c["name"] for c in saved["class_mapping"]], self.class_names)
        self.assertEqual(saved["dataset"]["valid_images"], 5)
        self.assertEqual([Path(i["path"]).name for i in saved["dataset"]["invalid_images"]], ["broken.webp"])

        m = saved["metrics"]
        predictions = saved["predictions"]
        right = sum(p["correct"] for p in predictions)
        self.assertEqual((m["total"], m["correct"]), (5, right))
        self.assertAlmostEqual(m["accuracy"], right / 5, msg="accuracy from labels, not confidence")
        self.assertEqual(sum(map(sum, m["confusion_matrix"]["matrix"])), 5)
        self.assertEqual(len(m["confusion_matrix"]["matrix"]), 34)
        self.assertEqual(len(m["classes_without_samples"]), 30)  # pizza, samosa, sushi, idli have images

        pizza = next(p for p in predictions if p["path"].endswith("dataset_pizza.jpg"))
        self.assertEqual(pizza["predicted"], "pizza")
        self.assertAlmostEqual(pizza["confidence"], 0.8913, places=3, msg="matches the API result for this image")

        overlap_paths = {Path(o["path"]).name for o in saved["quality"]["exact_overlaps"]}
        self.assertEqual(overlap_paths, {"dataset_pizza.jpg", "samosa.jpg"})
        self.assertEqual(saved["overlapping_images_excluded"], 2)
        self.assertEqual(saved["metrics_excluding_overlaps"]["total"], 3)
        self.assertEqual(saved["quality"]["independence_status"], "overlap_detected")
        self.assertTrue(any("exact copies" in w for w in saved["quality"]["warnings"]))

        markdown = (self.output / "real_world_evaluation.md").read_text(encoding="utf-8")
        for heading in ("## Overall metrics", "## Per-class metrics", "## Confusion matrix",
                        "## Model confidence (not accuracy)", "### Warnings", "## Reproduce"):
            self.assertIn(heading, markdown)
        self.assertEqual(report["status"], "ok")


if __name__ == "__main__":
    unittest.main()
