"""End-to-end tests for GET /api/nutrition and GET /api/nutrition/{food_class},
plus a backward-compatibility check of POST /api/predict. A real uvicorn
server is started with FOODCAL_DATA_DIR pointing at a temporary directory,
so the real storage/ history is never touched.

Run from the project root:
    .venv-gpu\\Scripts\\python.exe -B -m unittest tests.test_nutrition_api -v
"""
import json
import os
import sys
import tempfile
import unittest
import urllib.parse
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.app.services.nutrition import REFERENCE_PATH  # noqa: E402
from test_history_api import Server  # noqa: E402

REFERENCE = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
ENTRIES = {f["food_class"]: f for f in REFERENCE["foods"]}
# The /api/predict response keys before the nutrition feature; must not change.
PREDICT_KEYS = {
    "filename", "predicted_food", "confidence", "top_predictions", "device", "note",
    "low_confidence", "low_confidence_threshold", "prediction_id", "saved", "storage_error",
}


def kcal(value):
    """Whole kcal, rounded half up like the API (Python's round() rounds half to even)."""
    return int(Decimal(str(value)).quantize(Decimal(1), rounding=ROUND_HALF_UP))


def nutrition_path(food, grams):
    return f"/api/nutrition/{urllib.parse.quote(food)}?grams={urllib.parse.quote(str(grams))}"


class NutritionApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.server = Server(Path(cls._tmp.name) / "storage")
        cls.server.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.stop()
        cls._tmp.cleanup()

    def test_coverage_lists_every_model_class(self):
        status, classes = self.server.json("GET", "/api/classes")
        self.assertEqual(status, 200)
        status, body = self.server.json("GET", "/api/nutrition")
        self.assertEqual(status, 200)
        self.assertEqual(body["total_classes"], 34)
        self.assertEqual([f["food_class"] for f in body["foods"]], classes["classes"])
        self.assertEqual(body["available_classes"], sum(f["available"] for f in body["foods"]))
        self.assertEqual(body["unavailable_classes"], sum(not f["available"] for f in body["foods"]))
        self.assertEqual(body["direct_matches"] + body["approximate_matches"], body["available_classes"])
        licences = " ".join(source["license"] for source in body["sources"])
        self.assertIn("CC0", licences)
        self.assertEqual({source["key"] for source in body["sources"]}, {"FNDDS", "SR", "BRANDED"})
        self.assertEqual(body["research"]["file"], "docs/research/FoodCal_AI_Nutrition_Database.json")
        self.assertEqual((body["direct_matches"], body["approximate_matches"], body["unavailable_classes"]), (21, 8, 5))
        self.assertEqual(body["default_serving_g"], 100)
        pakode = next(f for f in body["foods"] if f["food_class"] == "pakode")
        self.assertTrue(pakode["flagged_for_review"])

    def test_direct_match_values_and_source(self):
        status, body = self.server.json("GET", nutrition_path("burger", 200))
        self.assertEqual(status, 200)
        entry = ENTRIES["burger"]
        self.assertEqual(body["match_type"], entry["match_type"])
        self.assertEqual(body["serving_size_g"], 200)
        self.assertEqual(body["source"]["record_id"], entry["reference"]["record_id"])
        self.assertEqual(body["source"]["fdc_id"], int(entry["reference"]["record_id"]))
        self.assertEqual(body["source"]["food_description"], entry["reference"]["food_description"])
        per_100g = entry["nutrition_per_100g"]
        self.assertEqual(body["estimated_nutrition"]["calories_kcal"], kcal(per_100g["calories_kcal"] * 2))
        self.assertAlmostEqual(body["estimated_nutrition"]["protein_g"], per_100g["protein_g"] * 2, delta=0.05)
        self.assertIn("not measured from your photo", body["notice"])

    def test_class_names_with_spaces_work(self):
        status, body = self.server.json("GET", nutrition_path("Baked Potato", 150))
        self.assertEqual(status, 200)
        self.assertEqual(body["food_class"], "Baked Potato")

    def test_requested_classes_succeed_with_reference_values(self):
        for food in ("chapati", "samosa", "idli", "masala_dosa", "butter_naan", "chicken_curry"):
            with self.subTest(food=food):
                status, body = self.server.json("GET", nutrition_path(food, 100))
                self.assertEqual(status, 200)
                self.assertTrue(body["available"])
                per_100g = ENTRIES[food]["nutrition_per_100g"]
                self.assertEqual(body["nutrition_per_100g"]["calories_kcal"], kcal(per_100g["calories_kcal"]))
                self.assertEqual(body["estimated_nutrition"], body["nutrition_per_100g"])
                self.assertEqual(body["source"]["record_id"], ENTRIES[food]["reference"]["record_id"])
        # A valid class without reliable data is a successful response, not a 404.
        status, body = self.server.json("GET", nutrition_path("paani_puri", 100))
        self.assertEqual((status, body["available"]), (200, False))

    def test_150_gram_serving_scales_the_100_gram_values(self):
        for food in ("chapati", "samosa"):
            per_100g = ENTRIES[food]["nutrition_per_100g"]
            status, body = self.server.json("GET", nutrition_path(food, 150))
            self.assertEqual(status, 200)
            self.assertEqual(body["serving_size_g"], 150)
            self.assertEqual(body["estimated_nutrition"]["calories_kcal"], kcal(Decimal(str(per_100g["calories_kcal"])) * Decimal("1.5")))
            self.assertAlmostEqual(body["estimated_nutrition"]["fat_g"], per_100g["fat_g"] * 1.5, delta=0.051)

    def test_route_not_found_differs_from_unknown_food(self):
        # The frontend relies on this to tell an outdated backend from a bad class.
        status, body = self.server.json("GET", "/api/nutritionz/chapati?grams=100")
        self.assertEqual((status, body["detail"]), (404, "Not Found"))
        status, body = self.server.json("GET", nutrition_path("pasta", 100))
        self.assertEqual(status, 404)
        self.assertNotEqual(body["detail"], "Not Found")
        status, body = self.server.json("GET", "/")
        self.assertEqual(body["nutrition_reference"], "ok")

    def test_response_includes_preparation_quality_and_source_metadata(self):
        status, body = self.server.json("GET", nutrition_path("kadai_paneer", 100))
        self.assertEqual(status, 200)
        entry = ENTRIES["kadai_paneer"]
        for key in ("food_name", "preparation_method", "data_quality", "limitations", "notes",
                    "verification_status", "flagged_for_review"):
            self.assertEqual(body[key], entry[key], key)
        self.assertEqual(body["source"]["mirror_url"], entry["reference"]["mirror_url"])
        self.assertEqual(body["source"]["carbohydrate_definition"], entry["reference"]["carbohydrate_definition"])

    def test_approximate_and_unavailable_classes(self):
        status, body = self.server.json("GET", nutrition_path("butter_naan", 100))
        self.assertEqual((status, body["match_type"], body["data_quality"]), (200, "APPROXIMATE_MATCH", "MEDIUM"))
        self.assertTrue(body["notice"].startswith("Approximate"))
        for food in ("chole_bhature", "kulfi"):
            status, body = self.server.json("GET", nutrition_path(food, 100))
            self.assertEqual((status, body["available"], body["estimated_nutrition"]), (200, False, None))
        status, body = self.server.json("GET", nutrition_path("jalebi", 100))
        self.assertEqual(status, 200)
        self.assertFalse(body["available"])
        self.assertIsNone(body["nutrition_per_100g"])
        self.assertIsNone(body["estimated_nutrition"])

    def test_invalid_serving_sizes_return_422(self):
        for grams in ("0", "-5", "2000.5", "nan", "inf", "-inf", "abc", ""):
            with self.subTest(grams=grams):
                status, _ = self.server.json("GET", nutrition_path("pizza", grams))
                self.assertEqual(status, 422)
        status, _ = self.server.json("GET", "/api/nutrition/pizza")
        self.assertEqual(status, 422)
        status, _ = self.server.json("GET", nutrition_path("pizza", "2000"))
        self.assertEqual(status, 200)

    def test_unknown_food_class_returns_404(self):
        for food in ("pasta", "Pizza", "apple pie"):
            with self.subTest(food=food):
                status, body = self.server.json("GET", nutrition_path(food, 100))
                self.assertEqual(status, 404)
                self.assertIn("not one of the 34", body["detail"])

    def test_predict_response_is_unchanged(self):
        status, body, _ = self.server.predict("pizza")
        self.assertEqual(status, 200)
        self.assertEqual(set(body), PREDICT_KEYS)

    def test_cors_allows_the_frontend_to_read_nutrition(self):
        status, headers, _ = self.server.request(
            "GET", nutrition_path("idli", 100), headers={"Origin": "http://localhost:5173"})
        self.assertEqual(status, 200)
        self.assertEqual(headers.get("access-control-allow-origin"), "http://localhost:5173")


class MissingReferenceTests(unittest.TestCase):
    def test_missing_reference_disables_only_nutrition(self):
        with tempfile.TemporaryDirectory() as tmp:
            previous = os.environ.get("FOODCAL_NUTRITION_REFERENCE")
            os.environ["FOODCAL_NUTRITION_REFERENCE"] = str(Path(tmp) / "missing.json")
            try:
                server = Server(Path(tmp) / "storage")
                server.start()
            finally:
                if previous is None:
                    os.environ.pop("FOODCAL_NUTRITION_REFERENCE")
                else:
                    os.environ["FOODCAL_NUTRITION_REFERENCE"] = previous
            try:
                status, body = server.json("GET", "/")
                self.assertEqual((status, body["nutrition_reference"]), (200, "unavailable"))
                for path in ("/api/nutrition", nutrition_path("pizza", 100)):
                    status, body = server.json("GET", path)
                    self.assertEqual(status, 503)
                    self.assertIn("Nutrition data is unavailable", body["detail"])
                status, body, _ = server.predict("pizza")
                self.assertEqual(status, 200)
                self.assertEqual(set(body), PREDICT_KEYS)
            finally:
                server.stop()


if __name__ == "__main__":
    unittest.main()
