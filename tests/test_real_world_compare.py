"""Tests for evaluating several checkpoints on real-world photos and comparing
them (evaluation/evaluate_real_world.py, evaluation/compare_real_world.py).

All datasets and reports are written to temporary directories. The test
images are copies of processed test images, used only to exercise the code;
they are NOT real-world photos. Checkpoints are only read.

Run from the project root:
    .venv-gpu\\Scripts\\python.exe -B -m unittest tests.test_real_world_compare -v
"""
import contextlib
import hashlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import compare_real_world, evaluate_real_world  # noqa: E402
from evaluation.dataset_audit import scan_real_world  # noqa: E402
from evaluation.evaluate_real_world import check_checkpoint, main, run_evaluation  # noqa: E402

PRODUCTION = ROOT / "models" / "efficientnet_b0" / "food_efficientnet_b0_best.pth"
V2 = ROOT / "models" / "efficientnet_b0_v2" / "food_efficientnet_b0_v2_best.pth"
DASHBOARD_REPORT = ROOT / "reports" / "real_world" / "real_world_evaluation.json"
CPU = torch.device("cpu")
FOODS = ("pizza", "idli", "samosa", "burger", "kulfi")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest() if Path(path).is_file() else None


def evaluate(real_dir, checkpoint, out):
    return run_evaluation(real_world_dir=real_dir, checkpoint_path=checkpoint, output_dir=out, device=CPU,
                          progress=lambda *_: None)


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.reference = torch.load(PRODUCTION, map_location="cpu", weights_only=True)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def save(self, name, **changes):
        path = self.tmp / name
        torch.save({**{k: self.reference[k] for k in ("class_names", "image_size")},
                    "model_state_dict": {}, **changes}, path)
        return path

    def test_production_and_v2_are_compatible(self):
        check_checkpoint(PRODUCTION)
        check_checkpoint(V2)

    def test_invalid_checkpoints_are_rejected(self):
        corrupt = self.tmp / "corrupt.pth"
        corrupt.write_bytes(b"not a checkpoint")
        reordered = list(reversed(self.reference["class_names"]))
        cases = {
            "not a readable": corrupt,
            "class list": self.save("reordered.pth", class_names=reordered),
            "class list ": self.save("33.pth", class_names=self.reference["class_names"][:33]),
            "px input": self.save("size.pth", image_size=256),
        }
        for message, path in cases.items():
            with self.subTest(path=path.name):
                with self.assertRaisesRegex(ValueError, message.strip()):
                    check_checkpoint(path)
        with self.assertRaises(FileNotFoundError):
            check_checkpoint(self.tmp / "missing.pth")
        no_names = self.tmp / "no_names.pth"
        torch.save({"model_state_dict": {}, "image_size": 224}, no_names)
        with self.assertRaisesRegex(ValueError, "missing class_names"):
            check_checkpoint(no_names)

    def test_cli_rejects_invalid_checkpoint_and_shared_dashboard_output(self):
        before = sha(DASHBOARD_REPORT)
        for argv in (["--checkpoint", str(self.tmp / "missing.pth"), "--output-dir", str(self.tmp / "o")],
                     ["--checkpoint", str(V2)]):  # v2 into the dashboard's default folder
            with self.subTest(argv=argv), mock.patch("sys.stderr"):
                with self.assertRaises(SystemExit):
                    main(argv + ["--real-world-dir", str(self.tmp), "--source-archive", "none", "--no-cache"])
        self.assertEqual(sha(DASHBOARD_REPORT), before, "the dashboard's report was not touched")


class EvaluateAndCompareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = t = Path(tempfile.mkdtemp())
        cls.real = t / "real_world"
        for food in FOODS:
            folder = cls.real / food
            folder.mkdir(parents=True)
            for name in ("000000.jpg", "000001.jpg"):
                shutil.copy(ROOT / "data" / "processed" / "test" / food / name, folder / name)
        (cls.real / "pizza_slices").mkdir()  # not a class folder: must be ignored
        shutil.copy(ROOT / "data" / "processed" / "test" / "pizza" / "000002.jpg", cls.real / "pizza_slices" / "x.jpg")
        cls.checkpoint_shas = (sha(PRODUCTION), sha(V2))
        cls.old = evaluate(cls.real, PRODUCTION, t / "out" / "production")
        cls.new = evaluate(cls.real, V2, t / "out" / "v2")
        cls.comparison = compare_real_world.compare(t / "out" / "production" / "real_world_evaluation.json",
                                                    t / "out" / "v2" / "real_world_evaluation.json")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_checkpoint_selection_and_separate_outputs(self):
        self.assertEqual(self.old["checkpoint"]["path"], "models/efficientnet_b0/food_efficientnet_b0_best.pth")
        self.assertEqual(self.new["checkpoint"]["path"], "models/efficientnet_b0_v2/food_efficientnet_b0_v2_best.pth")
        self.assertEqual((self.old["checkpoint"]["sha256"], self.new["checkpoint"]["sha256"]), self.checkpoint_shas)
        for tag in ("production", "v2"):
            for ext in ("json", "md"):
                self.assertTrue((self.tmp / "out" / tag / f"real_world_evaluation.{ext}").is_file())
        self.assertEqual((sha(PRODUCTION), sha(V2)), self.checkpoint_shas, "checkpoints unchanged")

    def test_labels_come_only_from_exact_class_folders(self):
        self.assertEqual(self.old["dataset"]["valid_images"], 2 * len(FOODS))
        self.assertTrue(all(Path(p["path"]).parent.name == p["actual"] for p in self.old["predictions"]))
        self.assertTrue(any("pizza_slices" in e["path"] for e in self.old["quality"]["unexpected_entries"]))

    def test_label_that_does_not_match_its_folder_is_rejected(self):
        real_audit = evaluate_real_world.audit_real_world

        def mislabelled(*args, **kwargs):
            audit = real_audit(*args, **kwargs)
            audit["_records"][0] = {**audit["_records"][0], "label": "idli"}  # a pizza/ path labelled idli
            return audit

        with mock.patch.object(evaluate_real_world, "audit_real_world", mislabelled):
            with self.assertRaisesRegex(ValueError, "does not match the class folder"):
                evaluate(self.real, PRODUCTION, self.tmp / "out" / "mislabelled")

    def test_reports_include_required_metrics(self):
        for report in (self.old, self.new):
            m = report["metrics"]
            for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "confusion_matrix"):
                self.assertIsNotNone(m[key])
            present = [r for r in m["per_class"] if r["support"]]
            self.assertEqual(len(present), len(FOODS))
            self.assertAlmostEqual(m["macro_recall"], sum(r["recall"] for r in present) / len(present))
            self.assertEqual(sum(map(sum, m["confusion_matrix"]["matrix"])), 2 * len(FOODS))
        md = (self.tmp / "out" / "v2" / "real_world_evaluation.md").read_text(encoding="utf-8")
        for text in ("Macro precision", "Macro recall", "## Incorrect predictions", "Image manifest"):
            self.assertIn(text, md)

    def test_both_models_saw_the_identical_image_manifest(self):
        self.assertEqual(self.old["image_manifest"], self.new["image_manifest"])
        c = self.comparison
        self.assertEqual((c["status"], c["image_manifest"], c["checks"]["same_images_labels_and_file_hashes"]),
                         ("ok", self.old["image_manifest"], True))

    def test_comparison_numbers_match_the_individual_reports(self):
        c = self.comparison
        for tag, report in (("old", self.old), ("new", self.new)):
            for key in ("accuracy", "macro_f1", "weighted_f1"):
                self.assertAlmostEqual(c["overall"][tag][key], report["metrics"][key])
            self.assertEqual(c["confusion_matrices"][tag], report["metrics"]["confusion_matrix"]["matrix"])
        changed = [
            o["path"] for o, n in zip(sorted(self.old["predictions"], key=lambda p: p["path"]),
                                      sorted(self.new["predictions"], key=lambda p: p["path"]))
            if o["predicted"] != n["predicted"]]
        self.assertEqual([x["path"] for x in c["changed_predictions"]], changed)
        pt = c["paired_test"]
        self.assertEqual(pt["old_wrong_new_right"] - pt["old_right_new_wrong"],
                         c["overall"]["new"]["correct"] - c["overall"]["old"]["correct"])
        self.assertIn("## Limitations of this sample", compare_real_world.markdown(c))

    def test_incompatible_reports_stay_pending_with_a_reason(self):
        base = json.loads((self.tmp / "out" / "v2" / "real_world_evaluation.json").read_text(encoding="utf-8"))
        old_path = self.tmp / "out" / "production" / "real_world_evaluation.json"
        variants = {
            "fewer images": {**base, "predictions": base["predictions"][1:]},
            "changed file": {**base, "predictions": [{**base["predictions"][0], "sha256": "0" * 64},
                                                     *base["predictions"][1:]]},
            "class mapping": {**base, "class_mapping": list(reversed(base["class_mapping"]))},
        }
        for name, report in variants.items():
            with self.subTest(name=name):
                path = self.tmp / f"{name}.json"
                path.write_text(json.dumps(report), encoding="utf-8")
                result = compare_real_world.compare(old_path, path)
                self.assertEqual(result["status"], "pending")
                self.assertIn("incompatible", result["reasons"][0])
                self.assertNotIn("overall", result)


class EmptyRealWorldTests(unittest.TestCase):
    def test_empty_folder_gives_no_metrics_and_a_pending_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            real = tmp / "real_world"
            for food in FOODS:
                (real / food).mkdir(parents=True)
            old = evaluate(real, PRODUCTION, tmp / "production")
            new = evaluate(real, V2, tmp / "v2")
            for report in (old, new):
                self.assertEqual((report["status"], report["metrics"], report["image_manifest"]["images"]),
                                 ("no_valid_images", None, 0))
            c = compare_real_world.compare(tmp / "production" / "real_world_evaluation.json",
                                           tmp / "v2" / "real_world_evaluation.json")
            self.assertEqual(c["status"], "pending")
            self.assertEqual(len(c["reasons"]), 2)
            self.assertNotIn("overall", c)
            self.assertIn("Status: pending", compare_real_world.markdown(c))
            missing = compare_real_world.compare(tmp / "production" / "real_world_evaluation.json", tmp / "nope.json")
            self.assertEqual(missing["status"], "pending")


class CliAndFolderTests(unittest.TestCase):
    def test_help_documents_checkpoint_output_dir_and_examples(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.assertRaises(SystemExit) as exit_:
            main(["--help"])
        self.assertEqual(exit_.exception.code, 0)
        text = " ".join(out.getvalue().split())  # argparse wraps help lines
        for expected in ("--checkpoint", "--output-dir", "production model",
                         "--checkpoint models/efficientnet_b0_v2/food_efficientnet_b0_v2_best.pth "
                         "--output-dir reports/real_world_v2"):
            self.assertIn(expected, text)

    def test_positional_checkpoint_is_rejected_by_the_parser(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as exit_:
            main([str(V2), "--output-dir", "reports/real_world_v2"])
        self.assertEqual(exit_.exception.code, 2)
        self.assertIn("unrecognized arguments", err.getvalue())

    def test_class_folders_must_match_exactly_including_case(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "Pizza").mkdir()  # wrong case; on Windows `tmp / "pizza"` also finds it
            shutil.copy(ROOT / "data" / "processed" / "test" / "pizza" / "000000.jpg", tmp / "Pizza" / "a.jpg")
            scan = scan_real_world(tmp, ["pizza", "idli"])
            self.assertEqual(scan["images"], [], "images in a wrongly named folder are not evaluated")
            self.assertEqual(scan["missing_class_folders"], ["pizza", "idli"])
            self.assertIn("rename it to 'pizza'", scan["unexpected_entries"][0]["note"])

    def test_empty_dataset_cli_explains_how_to_add_photos(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for food in FOODS:
                (tmp / "real" / food).mkdir(parents=True)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                main(["--real-world-dir", str(tmp / "real"), "--output-dir", str(tmp / "report"),
                      "--source-archive", "none", "--no-cache", "--device", "cpu"])
            text = out.getvalue()
            for expected in ("No valid real-world images: no metrics were calculated.", "Found 0 valid images",
                             ".jpg, .jpeg, .png, .webp", "case-sensitive", "Baked Potato", "provenance.csv",
                             "never copies of training"):
                self.assertIn(expected, text)
            report = json.loads((tmp / "report" / "real_world_evaluation.json").read_text(encoding="utf-8"))
            self.assertEqual((report["status"], report["metrics"]), ("no_valid_images", None))
            md = (tmp / "report" / "real_world_evaluation.md").read_text(encoding="utf-8")
            self.assertIn("## How to add real-world photos", md)
            self.assertNotIn("Accuracy", md)


if __name__ == "__main__":
    unittest.main()
