"""Tests for the nutrition reference data and calculations
(backend/app/services/nutrition.py and nutrition_reference.json).

No server and no model inference. The checkpoint is only read for its
class list. The stored decisions and text are checked against the research
database in docs/research/, and when the source downloads are present in
data/external/, every stored value is also checked against them.

Run from the project root:
    .venv-gpu\\Scripts\\python.exe -B -m unittest tests.test_nutrition -v
"""
import copy
import json
import math
import shutil
import sys
import tempfile
import unittest
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.services import build_nutrition_reference as build  # noqa: E402
from backend.app.services.nutrition import (  # noqa: E402
    NUTRIENT_KEYS,
    REFERENCE_PATH,
    NutritionDataError,
    NutritionReference,
    parse_grams,
    scale,
    validate_reference,
)

CHECKPOINT = ROOT / "models" / "efficientnet_b0" / "food_efficientnet_b0_best.pth"
CLASS_NAMES = list(torch.load(CHECKPOINT, map_location="cpu", weights_only=True)["class_names"])
DATA = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
RESEARCH = {f["food_class"]: f for f in json.loads((ROOT / build.RESEARCH["file"]).read_text(encoding="utf-8"))["foods"]}
SOURCES_AVAILABLE = all((build.EXTERNAL / meta["file"]).is_file() for meta in build.SOURCES.values())


def expected_serving(value, grams, places):
    exact = Decimal(str(value)) * Decimal(str(grams)) / Decimal(100)
    return float(exact.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))


class ReferenceDataTests(unittest.TestCase):
    def setUp(self):
        self.reference = NutritionReference(DATA, CLASS_NAMES)

    def test_every_model_class_has_exactly_one_entry(self):
        names = [f["food_class"] for f in DATA["foods"]]
        self.assertEqual(len(CLASS_NAMES), 34)
        self.assertEqual(sorted(names), sorted(CLASS_NAMES))
        self.assertEqual(len(names), len(set(names)))

    def test_coverage_counts_come_from_the_data(self):
        coverage = self.reference.coverage()
        by_type = {t: sum(f["match_type"] == t for f in DATA["foods"])
                   for t in ("DIRECT_MATCH", "APPROXIMATE_MATCH", "UNAVAILABLE")}
        self.assertEqual(coverage["total_classes"], 34)
        self.assertEqual(coverage["direct_matches"], by_type["DIRECT_MATCH"])
        self.assertEqual(coverage["approximate_matches"], by_type["APPROXIMATE_MATCH"])
        self.assertEqual(coverage["unavailable_classes"], by_type["UNAVAILABLE"])
        self.assertEqual(coverage["available_classes"] + coverage["unavailable_classes"], 34)
        self.assertEqual([f["food_class"] for f in coverage["foods"]], CLASS_NAMES)

    def test_reviewed_decisions_for_dishes_that_must_not_be_treated_as_identical(self):
        def classes(match):
            return {f["food_class"] for f in DATA["foods"] if f["match_type"] == match}

        # Plain naan for butter naan, a generic wrap for a kathi roll, paneer makhani for kadai
        # paneer or the bhaji without the pav are never direct matches.
        self.assertEqual(classes("APPROXIMATE_MATCH"), {"butter_naan", "chai", "dal_makhani", "kaathi_rolls",
                                                        "kadai_paneer", "masala_dosa", "momos", "pav_bhaji"})
        # Chickpea curry alone, puri alone or ordinary ice cream must not stand in for these.
        self.assertEqual(classes("UNAVAILABLE"), {"chole_bhature", "dhokla", "jalebi", "kulfi", "paani_puri"})
        self.assertEqual(len(classes("DIRECT_MATCH")), 21)
        qualities = [f["data_quality"] for f in DATA["foods"]]
        self.assertEqual({q: qualities.count(q) for q in set(qualities)}, {"HIGH": 11, "MEDIUM": 14, "LOW": 4, "NONE": 5})

    def test_decisions_and_text_are_the_research_database_verbatim(self):
        for food in DATA["foods"]:
            research = RESEARCH[food["food_class"]]
            with self.subTest(food=food["food_class"]):
                for key in ("food_name", "preparation_method", "match_type", "data_quality",
                            "limitations", "notes", "verification_status"):
                    self.assertEqual(food[key], research[key], key)
                ref, source = food["reference"], research["source"]
                if ref is None:
                    self.assertIsNone(source["record_id"])
                    self.assertTrue(all(v is None for v in research["nutrition_per_100g"].values()))
                    continue
                self.assertEqual("FDC ID " + ref["record_id"], source["record_id"])
                self.assertEqual(build.FDC_URL.format(id=ref["record_id"]), source["url"])
                self.assertEqual(ref["research_description"], research["reference_description"])
                self.assertEqual(ref["mirror_url"], source["mirror_url"])
                self.assertEqual(ref["carbohydrate_definition"], source["carbohydrate_definition"])

    def test_research_values_are_the_source_values_rounded(self):
        # Stored values come from the source files; the research copies (read via a mirror)
        # must agree with them to the precision the research gives.
        for food in DATA["foods"]:
            if food["nutrition_per_100g"] is None:
                continue
            for key in NUTRIENT_KEYS:
                stored, research = food["nutrition_per_100g"][key], RESEARCH[food["food_class"]]["nutrition_per_100g"][key]
                with self.subTest(food=food["food_class"], key=key):
                    self.assertTrue(build.rounds_to(stored, research), f"{research} vs {stored}")

    def test_rounding_check_accepts_only_the_research_precision(self):
        self.assertTrue(build.rounds_to(21.45, 21.4))
        self.assertTrue(build.rounds_to(10.85, 10.8))
        self.assertTrue(build.rounds_to(6.67, 6.7))
        self.assertTrue(build.rounds_to(12.03, 12))
        self.assertTrue(build.rounds_to(140.0, 140))
        self.assertFalse(build.rounds_to(12.6, 12))
        self.assertFalse(build.rounds_to(251.0, 307))
        self.assertFalse(build.rounds_to(4.93, 4.8))

    def test_only_pakode_is_flagged_for_review(self):
        flagged = [f["food_class"] for f in DATA["foods"] if f["flagged_for_review"]]
        self.assertEqual(flagged, ["pakode"])
        self.assertTrue(self.reference.estimate("pakode", 100)["limitations"].startswith("FLAGGED FOR REVIEW"))

    def test_every_record_names_its_source_and_licence(self):
        for food in DATA["foods"]:
            ref = food["reference"]
            if ref is None:
                continue
            source = DATA["sources"][ref["source_key"]]
            with self.subTest(food=food["food_class"]):
                self.assertTrue(ref["record_id"])
                self.assertTrue(ref["food_description"])
                self.assertTrue(source["license"])
                self.assertEqual(source["sha256"], build.SOURCES[ref["source_key"]]["sha256"])
        self.assertIn("Open Government Licence v3.0", DATA["sources"]["COFID"]["license"])
        self.assertIn("CC0", DATA["sources"]["BRANDED"]["license"])

    def test_energy_is_consistent_with_macronutrients(self):
        # 4/4/9 kcal per gram; USDA uses food-specific factors, so allow 10%.
        for food in DATA["foods"]:
            values = food["nutrition_per_100g"]
            if values:
                atwater = 4 * values["protein_g"] + 4 * values["carbohydrates_g"] + 9 * values["fat_g"]
                self.assertAlmostEqual(atwater / values["calories_kcal"], 1, delta=0.10, msg=food["food_class"])

    @unittest.skipUnless(SOURCES_AVAILABLE, "source downloads not present in data/external/")
    def test_reference_file_matches_a_fresh_build_from_the_source_files(self):
        self.assertEqual(build.build(), DATA)


class ResponseTests(unittest.TestCase):
    def setUp(self):
        self.reference = NutritionReference(DATA, CLASS_NAMES)

    def test_direct_match_returns_usda_source(self):
        result = self.reference.estimate("idli", 100)
        self.assertTrue(result["available"])
        self.assertEqual(result["match_type"], "DIRECT_MATCH")
        self.assertEqual(result["source"]["dataset"], "USDA FoodData Central")
        self.assertEqual(result["source"]["food_description"], "Idli")
        self.assertEqual(result["source"]["record_id"], "2708346")
        self.assertEqual(result["source"]["fdc_id"], 2708346)
        self.assertIn("2708346", result["source"]["url"])
        self.assertIn("by difference", result["source"]["carbohydrate_definition"])
        self.assertIn("not measured from your photo", result["notice"])
        self.assertNotIn("Approximate", result["notice"])

    def test_response_carries_the_research_metadata(self):
        result = self.reference.estimate("samosa", 100)
        self.assertEqual(result["food_name"], "Samosa")
        self.assertEqual(result["preparation_method"], "Filled pastry, fried")
        self.assertEqual(result["data_quality"], "MEDIUM")
        self.assertFalse(result["flagged_for_review"])
        self.assertIn("does not state the filling", result["limitations"])
        self.assertEqual(result["verification_status"], "SOURCE_RECORD_READ_VIA_FDC_MIRROR_NOT_INSPECTED_ON_USDA_SITE")
        self.assertEqual(result["source"]["mirror_url"], "https://getfoodfacts.com/food/samosa-2708730")
        self.assertEqual(result["source"]["research_description"], "Samosa")

    def test_branded_record_is_marked_low_quality_with_label_carbohydrate(self):
        result = self.reference.estimate("kadai_paneer", 100)
        self.assertEqual((result["match_type"], result["data_quality"]), ("APPROXIMATE_MATCH", "LOW"))
        self.assertIn("Branded", result["source"]["dataset_detail"])
        self.assertEqual(result["source"]["food_description"], "PANEER MAKHANI")
        self.assertIn("label", result["source"]["carbohydrate_definition"])
        self.assertIn("paneer MAKHANI", result["limitations"])

    def test_approximate_match_can_be_medium_quality(self):
        result = self.reference.estimate("butter_naan", 100)
        self.assertEqual(result["match_type"], "APPROXIMATE_MATCH")
        self.assertEqual(result["data_quality"], "MEDIUM")
        self.assertTrue(result["notice"].startswith("Approximate"))
        self.assertIn("PLAIN naan", result["limitations"])

    def test_unavailable_class_has_no_values_at_all(self):
        result = self.reference.estimate("dhokla", 150)
        self.assertFalse(result["available"])
        self.assertEqual(result["match_type"], "UNAVAILABLE")
        self.assertIsNone(result["nutrition_per_100g"])
        self.assertIsNone(result["estimated_nutrition"])
        self.assertIsNone(result["source"])
        self.assertTrue(result["notes"])
        self.assertTrue(result["limitations"])
        self.assertEqual(result["verification_status"], "REQUIRES_RESEARCH")
        # Kulfi is not given ice cream's values.
        self.assertIsNone(self.reference.estimate("kulfi", 100)["estimated_nutrition"])

    def test_unknown_class_raises_key_error(self):
        with self.assertRaises(KeyError):
            self.reference.estimate("pasta", 100)


class CalculationTests(unittest.TestCase):
    def setUp(self):
        self.reference = NutritionReference(DATA, CLASS_NAMES)
        self.entry = next(f for f in DATA["foods"] if f["food_class"] == "burger")["nutrition_per_100g"]

    def test_serving_is_per_100g_times_grams_over_100(self):
        for grams in (1, 37.5, 100, 200, 333.3, 2000):
            result = self.reference.estimate("burger", grams)["estimated_nutrition"]
            for key in NUTRIENT_KEYS:
                places = 0 if key == "calories_kcal" else 1
                with self.subTest(grams=grams, key=key):
                    self.assertEqual(result[key], expected_serving(self.entry[key], grams, places))

    def test_calories_are_whole_numbers_and_grams_one_decimal(self):
        result = scale({"calories_kcal": 288.5, "protein_g": 17.25, "carbohydrates_g": 0.05, "fat_g": 1.0}, 100)
        self.assertEqual(result, {"calories_kcal": 289, "protein_g": 17.3, "carbohydrates_g": 0.1, "fat_g": 1.0})
        self.assertIsInstance(result["calories_kcal"], int)

    def test_values_scale_with_serving_size(self):
        small = self.reference.estimate("pizza", 100)["estimated_nutrition"]
        large = self.reference.estimate("pizza", 300)["estimated_nutrition"]
        for key in NUTRIENT_KEYS:
            self.assertAlmostEqual(large[key], 3 * small[key], delta=1.5 if key == "calories_kcal" else 0.15)

    def test_100_grams_equals_the_per_100g_values(self):
        for food in ("samosa", "chapati", "idli", "masala_dosa", "butter_naan", "chicken_curry"):
            with self.subTest(food=food):
                result = self.reference.estimate(food, 100)
                self.assertEqual(result["estimated_nutrition"], result["nutrition_per_100g"])
        self.assertEqual(self.reference.estimate("samosa")["serving_size_g"], 100)

    def test_150_grams_is_one_and_a_half_times_the_reference(self):
        for food in ("samosa", "chapati", "idli", "masala_dosa"):
            entry = next(f for f in DATA["foods"] if f["food_class"] == food)["nutrition_per_100g"]
            result = self.reference.estimate(food, 150)["estimated_nutrition"]
            for key in NUTRIENT_KEYS:
                with self.subTest(food=food, key=key):
                    self.assertEqual(result[key], expected_serving(entry[key], 150, 0 if key == "calories_kcal" else 1))

    def test_invalid_serving_sizes_are_rejected(self):
        for value in (0, -1, -0.0, 2000.01, 1e9, math.nan, math.inf, -math.inf, "abc", "", None, True, [100]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_grams(value)
        self.assertEqual(parse_grams("150"), 150.0)
        self.assertEqual(parse_grams(2000), 2000.0)
        self.assertEqual(parse_grams(0.5), 0.5)


class ValidationTests(unittest.TestCase):
    def broken(self, change):
        data = copy.deepcopy(DATA)
        change(data)
        return data

    def entry(self, data, name):
        return next(f for f in data["foods"] if f["food_class"] == name)

    def test_bad_reference_data_is_rejected(self):
        def set_value(name, key, value):
            return lambda d: self.entry(d, name)["nutrition_per_100g"].__setitem__(key, value)

        cases = {
            "negative": set_value("burger", "fat_g", -1.0),
            "nan": set_value("burger", "protein_g", math.nan),
            "infinite": set_value("burger", "calories_kcal", math.inf),
            "text": set_value("burger", "calories_kcal", "250"),
            "boolean": set_value("burger", "calories_kcal", True),
            "missing class": lambda d: d["foods"].pop(),
            "duplicate": lambda d: d["foods"].append(copy.deepcopy(d["foods"][0])),
            "unknown class": lambda d: self.entry(d, "pizza").__setitem__("food_class", "pasta"),
            "missing nutrient": lambda d: self.entry(d, "pizza")["nutrition_per_100g"].pop("fat_g"),
            "zero values for unavailable": lambda d: self.entry(d, "dhokla").__setitem__(
                "nutrition_per_100g", dict.fromkeys(NUTRIENT_KEYS, 0)),
            "approximate without notes": lambda d: self.entry(d, "butter_naan").__setitem__("notes", ""),
            "approximate without limitations": lambda d: self.entry(d, "butter_naan").__setitem__("limitations", ""),
            "missing food name": lambda d: self.entry(d, "idli").pop("food_name"),
            "missing preparation": lambda d: self.entry(d, "idli").__setitem__("preparation_method", None),
            "missing verification status": lambda d: self.entry(d, "dhokla").pop("verification_status"),
            "review flag not boolean": lambda d: self.entry(d, "pakode").__setitem__("flagged_for_review", "yes"),
            "missing carbohydrate definition": lambda d: self.entry(d, "idli")["reference"].pop("carbohydrate_definition"),
            "insecure mirror url": lambda d: self.entry(d, "idli")["reference"].__setitem__("mirror_url", "http://x"),
            "bad match type": lambda d: self.entry(d, "idli").__setitem__("match_type", "CLOSE_ENOUGH"),
            "approximate rated high": lambda d: self.entry(d, "butter_naan").__setitem__("data_quality", "HIGH"),
            "unavailable rated low": lambda d: self.entry(d, "dhokla").__setitem__("data_quality", "LOW"),
            "missing record id": lambda d: self.entry(d, "idli")["reference"].pop("record_id"),
            "non-numeric fdc id": lambda d: self.entry(d, "idli")["reference"].__setitem__("record_id", "abc"),
            "unknown source": lambda d: self.entry(d, "idli")["reference"].__setitem__("source_key", "WEB"),
            "unavailable without reason": lambda d: self.entry(d, "dhokla").__setitem__("notes", ""),
            "source without licence": lambda d: d["sources"]["COFID"].__setitem__("license", ""),
            "schema": lambda d: d.__setitem__("schema_version", 2),
        }
        for label, change in cases.items():
            with self.subTest(case=label), self.assertRaises(NutritionDataError):
                validate_reference(self.broken(change), CLASS_NAMES)

    def test_missing_or_corrupt_file_raises_a_clear_error(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            with self.assertRaisesRegex(NutritionDataError, "could not be read"):
                NutritionReference.load(CLASS_NAMES, tmp / "missing.json")
            (tmp / "corrupt.json").write_text("{not json", encoding="utf-8")
            with self.assertRaisesRegex(NutritionDataError, "could not be read"):
                NutritionReference.load(CLASS_NAMES, tmp / "corrupt.json")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
