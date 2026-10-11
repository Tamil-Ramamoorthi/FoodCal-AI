"""Nutrition reference lookup for the model's food classes.

Values come from nutrition_reference.json, built by
build_nutrition_reference.py from the research decisions in docs/research/
and the USDA FoodData Central files they name (every value checked against
those files). They are typical values for a reference food, scaled to a serving
weight the user enters. They are never a measurement of the food in an
uploaded photo, and portion size is never inferred from an image.

Kept separate from the prediction history database; nothing here writes
to disk or calls an external service.
"""

import json
import math
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

REFERENCE_PATH = Path(__file__).with_name("nutrition_reference.json")
SCHEMA_VERSION = 3

NUTRIENT_KEYS = ("calories_kcal", "protein_g", "carbohydrates_g", "fat_g")
# Decimal places shown: whole kilocalories, grams to one decimal.
DECIMALS = {"calories_kcal": 0, "protein_g": 1, "carbohydrates_g": 1, "fat_g": 1}
MATCH_TYPES = ("DIRECT_MATCH", "APPROXIMATE_MATCH", "UNAVAILABLE")
# An approximate record is never HIGH quality; it is MEDIUM when the related
# food is close (e.g. plain naan for butter naan) and LOW when it is a
# component, a different dish or a single branded product.
QUALITY = {
    "DIRECT_MATCH": ("HIGH", "MEDIUM", "LOW"),
    "APPROXIMATE_MATCH": ("MEDIUM", "LOW"),
    "UNAVAILABLE": ("NONE",),
}
# Text every entry must carry, whether or not values are available.
ENTRY_TEXT = ("food_name", "preparation_method", "limitations", "notes", "verification_status")
FDC_SOURCES = ("FNDDS", "SR", "BRANDED")
MIN_GRAMS, MAX_GRAMS = 0, 2000  # exclusive minimum, inclusive maximum
DEFAULT_GRAMS = 100

NOTICE = (
    "Typical reference values for this food from {source}, scaled to the serving weight you entered. They are "
    "not measured from your photo; actual nutrition varies with the recipe, ingredients and portion."
)
APPROXIMATE_NOTICE = (
    "Approximate: no reference for this exact dish was found, so a related food or a component of the dish is "
    "used (see limitations). "
)
UNAVAILABLE_NOTICE = (
    "No sufficiently reliable reference values were found for this food, so no nutrition estimate is shown."
)


class NutritionDataError(ValueError):
    """The reference file is missing, unreadable or inconsistent."""


def _finite_non_negative(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0)


def validate_reference(data, class_names):
    """Raises NutritionDataError unless `data` has exactly one valid entry per class."""
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
        raise NutritionDataError("Unsupported nutrition reference format.")
    sources = data.get("sources")
    if not isinstance(sources, dict) or not sources:
        raise NutritionDataError("The nutrition reference lists no sources.")
    for key, source in sources.items():
        if not all(isinstance(source.get(f), str) and source[f] for f in
                   ("name", "dataset", "license", "record_label", "carbohydrate_definition")):
            raise NutritionDataError(f"Source {key} is missing its name, licence or definitions.")
    foods = data.get("foods")
    if not isinstance(foods, list):
        raise NutritionDataError("The nutrition reference has no food list.")
    seen = set()
    for entry in foods:
        name = entry.get("food_class") if isinstance(entry, dict) else None
        if name not in class_names:
            raise NutritionDataError(f"Unknown food class in nutrition reference: {name!r}.")
        if name in seen:
            raise NutritionDataError(f"{name}: listed more than once.")
        seen.add(name)
        match = entry.get("match_type")
        if match not in MATCH_TYPES:
            raise NutritionDataError(f"{name}: invalid match type {match!r}.")
        if entry.get("data_quality") not in QUALITY[match]:
            raise NutritionDataError(f"{name}: data quality {entry.get('data_quality')!r} does not fit {match}.")
        for field in ENTRY_TEXT:
            if not (isinstance(entry.get(field), str) and entry[field]):
                raise NutritionDataError(f"{name}: needs {field} explaining the reference or why data is unavailable.")
        if not isinstance(entry.get("flagged_for_review"), bool):
            raise NutritionDataError(f"{name}: flagged_for_review must be true or false.")
        values, reference = entry.get("nutrition_per_100g"), entry.get("reference")
        if match == "UNAVAILABLE":
            if values is not None or reference is not None:
                raise NutritionDataError(f"{name}: an unavailable food must not have values or a reference.")
            continue
        if not isinstance(values, dict) or set(values) != set(NUTRIENT_KEYS):
            raise NutritionDataError(f"{name}: needs exactly {', '.join(NUTRIENT_KEYS)} per 100 g.")
        for key in NUTRIENT_KEYS:
            if not _finite_non_negative(values[key]):
                raise NutritionDataError(f"{name}: {key} must be a finite, non-negative number.")
        if not (isinstance(reference, dict) and reference.get("source_key") in sources
                and all(isinstance(reference.get(f), str) and reference[f] for f in
                        ("record_id", "food_description", "research_description", "carbohydrate_definition"))):
            raise NutritionDataError(f"{name}: needs a source reference with a record id, descriptions and "
                                     "carbohydrate definition.")
        mirror = reference.get("mirror_url")
        if mirror is not None and not (isinstance(mirror, str) and mirror.startswith("https://")):
            raise NutritionDataError(f"{name}: mirror_url must be an https URL or null.")
        if reference["source_key"] in FDC_SOURCES and not reference["record_id"].isdigit():
            raise NutritionDataError(f"{name}: a FoodData Central record id must be numeric.")
    missing = [n for n in class_names if n not in seen]
    if missing:
        raise NutritionDataError(f"No nutrition entry for: {', '.join(missing)}.")


def parse_grams(value):
    """Serving weight as a float, or ValueError. Must be > 0 and <= 2000 g."""
    if isinstance(value, bool):
        raise ValueError("Serving size must be a number of grams.")
    try:
        grams = float(value)
    except (TypeError, ValueError):
        raise ValueError("Serving size must be a number of grams.") from None
    if not math.isfinite(grams) or not MIN_GRAMS < grams <= MAX_GRAMS:
        raise ValueError(f"Serving size must be more than {MIN_GRAMS} g and at most {MAX_GRAMS} g.")
    return grams


def round_value(key, value):
    """Rounds half up to the nutrient's display precision."""
    places = Decimal(1).scaleb(-DECIMALS[key])
    rounded = Decimal(str(value)).quantize(places, rounding=ROUND_HALF_UP)
    return int(rounded) if DECIMALS[key] == 0 else float(rounded)


def scale(per_100g, grams):
    """nutrient_per_100g * grams / 100 for each nutrient, computed exactly and then rounded."""
    factor = Decimal(str(grams)) / Decimal(100)
    return {key: round_value(key, Decimal(str(per_100g[key])) * factor) for key in NUTRIENT_KEYS}


class NutritionReference:
    def __init__(self, data, class_names):
        validate_reference(data, class_names)
        self.sources = data["sources"]
        self.retrieved = data.get("retrieved")
        self.research = data.get("research")
        self.class_names = list(class_names)
        self.by_class = {entry["food_class"]: entry for entry in data["foods"]}

    @classmethod
    def load(cls, class_names, path=REFERENCE_PATH):
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise NutritionDataError(f"The nutrition reference could not be read: {error}") from error
        return cls(data, class_names)

    def _source(self, entry):
        ref = entry["reference"]
        if ref is None:
            return None
        source = self.sources[ref["source_key"]]
        record_id = ref["record_id"]
        return {
            "dataset": source["name"],
            "dataset_detail": source["dataset"],
            "release": source.get("release"),
            "record_label": source["record_label"],
            "record_id": record_id,
            "fdc_id": int(record_id) if ref["source_key"] in FDC_SOURCES else None,
            "food_description": ref["food_description"],
            "research_description": ref["research_description"],
            "url": source.get("record_url", "").replace("{id}", record_id) or None,
            "mirror_url": ref.get("mirror_url"),
            "license": source["license"],
            "carbohydrate_definition": ref["carbohydrate_definition"],
        }

    def coverage(self):
        foods = []
        counts = dict.fromkeys(MATCH_TYPES, 0)
        for name in self.class_names:
            entry = self.by_class[name]
            counts[entry["match_type"]] += 1
            source = self._source(entry)
            foods.append({
                "food_class": name,
                "food_name": entry["food_name"],
                "preparation_method": entry["preparation_method"],
                "available": entry["match_type"] != "UNAVAILABLE",
                "match_type": entry["match_type"],
                "data_quality": entry["data_quality"],
                "flagged_for_review": entry["flagged_for_review"],
                "limitations": entry["limitations"],
                "source_name": source["dataset"] if source else None,
                "record_label": source["record_label"] if source else None,
                "record_id": source["record_id"] if source else None,
                "fdc_id": source["fdc_id"] if source else None,
                "food_description": source["food_description"] if source else None,
                "notes": entry["notes"],
            })
        used = {entry["reference"]["source_key"] for entry in self.by_class.values() if entry["reference"]}
        return {
            "total_classes": len(foods),
            "available_classes": counts["DIRECT_MATCH"] + counts["APPROXIMATE_MATCH"],
            "unavailable_classes": counts["UNAVAILABLE"],
            "direct_matches": counts["DIRECT_MATCH"],
            "approximate_matches": counts["APPROXIMATE_MATCH"],
            "default_serving_g": DEFAULT_GRAMS,
            "retrieved": self.retrieved,
            "research": self.research,
            "sources": [
                {"key": key, **{f: self.sources[key][f] for f in ("name", "dataset", "license", "carbohydrate_definition")}}
                for key in self.sources if key in used
            ],
            "foods": foods,
        }

    def estimate(self, food_class, grams=DEFAULT_GRAMS):
        """Nutrition for one class and serving. KeyError for an unknown class,
        ValueError for an invalid serving size."""
        entry = self.by_class[food_class]
        grams = parse_grams(grams)
        source = self._source(entry)
        result = {
            "food_class": food_class,
            "food_name": entry["food_name"],
            "preparation_method": entry["preparation_method"],
            "available": entry["match_type"] != "UNAVAILABLE",
            "match_type": entry["match_type"],
            "data_quality": entry["data_quality"],
            "flagged_for_review": entry["flagged_for_review"],
            "serving_size_g": grams,
            "nutrition_per_100g": None,
            "estimated_nutrition": None,
            "source": source,
            "limitations": entry["limitations"],
            "notes": entry["notes"],
            "verification_status": entry["verification_status"],
            "value_type": "reference_estimate",
        }
        if not result["available"]:
            result["value_type"] = "unavailable"
            result["notice"] = UNAVAILABLE_NOTICE
            return result
        per_100g = entry["nutrition_per_100g"]
        result["nutrition_per_100g"] = {key: round_value(key, per_100g[key]) for key in NUTRIENT_KEYS}
        result["estimated_nutrition"] = scale(per_100g, grams)
        result["notice"] = ((APPROXIMATE_NOTICE if entry["match_type"] == "APPROXIMATE_MATCH" else "")
                            + NOTICE.format(source=source["dataset"]))
        return result
