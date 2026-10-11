"""End-to-end tests for GET /api/evaluation/real-world and a regression
check of the offline GET /api/evaluation report. A real uvicorn server is
started with FOODCAL_DATA_DIR and FOODCAL_REAL_WORLD_REPORT pointing at a
temporary directory, so real storage/ and reports/ are never touched.

Run from the project root:
    .venv-gpu\\Scripts\\python.exe -B -m unittest discover -s tests -v
"""
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluation.evaluate_real_world import run_evaluation  # noqa: E402
from test_history_api import Server  # noqa: E402

CHECKPOINT = ROOT / "models" / "efficientnet_b0" / "food_efficientnet_b0_best.pth"
TEST_IMAGES = ROOT / "data" / "processed" / "test"
OFFLINE_REPORT = ROOT / "models" / "efficientnet_b0" / "evaluation_report.json"


class RealWorldApiTests(unittest.TestCase):
    """Ordered: the route re-reads the report file on every request, so one
    server serves each report state in turn."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls._tmp.name)
        cls.reports = tmp / "reports"
        cls.report_path = cls.reports / "real_world_evaluation.json"
        cls.real = tmp / "real_world_test"
        cls.class_names = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)["class_names"]
        for name in cls.class_names:
            (cls.real / name).mkdir(parents=True)

        previous = os.environ.get("FOODCAL_REAL_WORLD_REPORT")
        os.environ["FOODCAL_REAL_WORLD_REPORT"] = str(cls.report_path)
        try:
            cls.server = Server(tmp / "storage")
            cls.server.start()
        finally:
            if previous is None:
                os.environ.pop("FOODCAL_REAL_WORLD_REPORT")
            else:
                os.environ["FOODCAL_REAL_WORLD_REPORT"] = previous

    @classmethod
    def tearDownClass(cls):
        cls.server.stop()
        cls._tmp.cleanup()

    def evaluate(self):
        return run_evaluation(
            real_world_dir=self.real, checkpoint_path=CHECKPOINT, output_dir=self.reports,
            device=torch.device("cpu"), command="test", progress=lambda *_: None,
        )

    def test_01_missing_report_is_404(self):
        status, body = self.server.json("GET", "/api/evaluation/real-world")
        self.assertEqual(status, 404)
        self.assertIn("evaluate_real_world", body["detail"])

    def test_02_empty_report_is_served_without_metrics(self):
        self.evaluate()
        status, body = self.server.json("GET", "/api/evaluation/real-world")
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "no_valid_images")
        self.assertIsNone(body["metrics"])
        self.assertEqual(body["dataset"]["valid_images"], 0)
        self.assertTrue(body["quality"]["warnings"])

    def test_03_report_is_served_unchanged_without_predictions(self):
        for food in ("pizza", "samosa", "sushi"):
            shutil.copy(TEST_IMAGES / food / "000001.jpg", self.real / food / "photo.jpg")
        (self.real / "idli" / "broken.jpg").write_bytes(b"not an image")
        self.evaluate()
        saved = json.loads(self.report_path.read_text(encoding="utf-8"))

        status, body = self.server.json("GET", "/api/evaluation/real-world")
        self.assertEqual(status, 200)
        self.assertNotIn("predictions", body)
        self.assertEqual(body["metrics"], saved["metrics"])
        self.assertEqual(body["quality"], saved["quality"])
        self.assertEqual(body["dataset"]["valid_images"], 3)
        self.assertEqual(len(body["dataset"]["invalid_images"]), 1)
        self.assertEqual(body["metrics"]["total"], 3)
        self.assertEqual(len(body["metrics"]["classes_without_samples"]), 31)
        self.assertTrue(body["checkpoint_matches_deployed"])
        self.assertTrue(body["classes_match_model"])
        self.assertEqual(body["report_file"], "real_world_evaluation.json")

    def test_04_checkpoint_mismatch_is_reported(self):
        report = json.loads(self.report_path.read_text(encoding="utf-8"))
        report["checkpoint"]["sha256"] = "0" * 64
        self.report_path.write_text(json.dumps(report), encoding="utf-8")
        status, body = self.server.json("GET", "/api/evaluation/real-world")
        self.assertEqual(status, 200)
        self.assertFalse(body["checkpoint_matches_deployed"])

    def test_05_unreadable_or_foreign_reports_are_500(self):
        good = json.loads(self.report_path.read_text(encoding="utf-8"))
        for content in ("{not json", json.dumps({**good, "report_type": "dataset_audit"}),
                        json.dumps({**good, "schema_version": 2}), json.dumps([1, 2])):
            with self.subTest(content=content[:30]):
                self.report_path.write_text(content, encoding="utf-8")
                status, body = self.server.json("GET", "/api/evaluation/real-world")
                self.assertEqual(status, 500)
                self.assertIn("could not be read", body["detail"])

    def test_06_offline_report_unchanged(self):
        recorded = json.loads(OFFLINE_REPORT.read_text(encoding="utf-8-sig"))
        status, body = self.server.json("GET", "/api/evaluation")
        self.assertEqual(status, 200)
        for key in ("accuracy", "macro_f1", "weighted_f1", "test_images", "num_classes"):
            self.assertEqual(body[key], recorded[key])
        self.assertEqual(len(body["per_class"]), 34)
        self.assertTrue(body["classes_match_model"])


if __name__ == "__main__":
    unittest.main()
