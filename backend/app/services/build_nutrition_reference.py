"""Builds backend/app/services/nutrition_reference.json from downloaded
food-composition files. Run from the project root:

    .venv-gpu\\Scripts\\python.exe -m backend.app.services.build_nutrition_reference

The reviewed decisions come from the research database in docs/research/
(FoodCal_AI_Nutrition_Database.json, with its report and workbook): which
source record represents each model class, whether that is a DIRECT_MATCH
or an APPROXIMATE_MATCH or why the class is UNAVAILABLE, the data quality,
and the food name, preparation method, limitations and notes.

Every number in the output is copied from the source files listed in
SOURCES; none is typed in here. The research values were read from a public
FoodData Central mirror, so they are used only as a check: the build stops
unless each one equals the source file's value rounded to the research
value's precision. See docs/NUTRITION_DATA.md for the reasoning.

Sources (all openly licensed, no API key):
- USDA FoodData Central FNDDS and SR Legacy (public domain, CC0 1.0)
- USDA FoodData Central Branded Foods (public domain, CC0 1.0). The full
  download is 449 MB, so only the selected records are kept, copied
  verbatim into data/external/fdc/branded_selected_2026-04-30.csv by
      ... build_nutrition_reference --extract-branded <path to the full ZIP>
- UK CoFID 2021, McCance and Widdowson's Composition of Foods Integrated
  Dataset (Open Government Licence v3.0)

The class list is read (read-only) from the production checkpoint, and the
build stops unless the research lists every class exactly once.
"""

import argparse
import csv
import hashlib
import io
import json
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXTERNAL = ROOT / "data" / "external"
FDC_DIR = EXTERNAL / "fdc"
COFID_DIR = EXTERNAL / "cofid"
CHECKPOINT = ROOT / "models" / "efficientnet_b0" / "food_efficientnet_b0_best.pth"
OUTPUT = Path(__file__).with_name("nutrition_reference.json")
RETRIEVED = "2026-10-11"

FDC_LICENSE = "Public domain (CC0 1.0 Universal); USDA asks that FoodData Central be cited as the source."
FDC_CITATION = "U.S. Department of Agriculture, Agricultural Research Service. FoodData Central. https://fdc.nal.usda.gov/"
FDC_URL = "https://fdc.nal.usda.gov/food-details/{id}/nutrients"
FDC_CARBS = "Carbohydrate, by difference (includes fibre)"

SOURCES = {
    "FNDDS": {
        "name": "USDA FoodData Central",
        "dataset": "Food and Nutrient Database for Dietary Studies 2021-2023 (FNDDS)",
        "release": "2024-10-31",
        "file": "fdc/FoodData_Central_survey_food_csv_2024-10-31.zip",
        "sha256": "5ccc25ec2777a8982fbb61378a42f415316173eb11e48c9a8ba4cb19f5a4f29c",
        "license": FDC_LICENSE, "citation": FDC_CITATION, "record_label": "FDC ID", "record_url": FDC_URL,
        "carbohydrate_definition": FDC_CARBS,
    },
    "SR": {
        "name": "USDA FoodData Central",
        "dataset": "SR Legacy",
        "release": "2018-04",
        "file": "fdc/FoodData_Central_sr_legacy_food_csv_2018-04.zip",
        "sha256": "b80817294b8850530aaedf2e515c02593b1824f763a0ff356e5c2081643e6fd0",
        "license": FDC_LICENSE, "citation": FDC_CITATION, "record_label": "FDC ID", "record_url": FDC_URL,
        "carbohydrate_definition": FDC_CARBS,
    },
    "BRANDED": {
        "name": "USDA FoodData Central",
        "dataset": "Branded Foods (values declared on the manufacturer's label)",
        "release": "2026-04-30",
        "file": "fdc/branded_selected_2026-04-30.csv",
        "sha256": "d0fdafffd9271dee70bb4391898fbb6135e32f7dfa32ad9cd2b776b6af2c78d9",
        "full_download": "FoodData_Central_branded_food_csv_2026-04-30.zip",
        "full_download_sha256": "26050a5d03197469813754743a21ee0fad4ccf22b6aac2a995846a987719fc49",
        "license": FDC_LICENSE, "citation": FDC_CITATION, "record_label": "FDC ID", "record_url": FDC_URL,
        "carbohydrate_definition": "Total carbohydrate as declared on the manufacturer's label (includes fibre); "
                                   "label-rounded",
    },
    "COFID": {
        "name": "McCance and Widdowson's Composition of Foods Integrated Dataset (CoFID) 2021",
        "dataset": "CoFID 2021, Public Health England",
        "release": "2021",
        "file": "cofid/McCance_Widdowsons_Composition_of_Foods_Integrated_Dataset_2021.xlsx",
        "sha256": "436e9445ef2adb2a75f3d7edd51302de3adad25385f9795fc94ba58bd030e97d",
        "license": "Open Government Licence v3.0 (Crown copyright). Contains public sector information licensed "
                   "under the Open Government Licence v3.0.",
        "citation": "Public Health England (2021) McCance and Widdowson's The Composition of Foods Integrated "
                    "Dataset 2021. https://www.gov.uk/government/publications/composition-of-foods-integrated-dataset-cofid",
        "record_label": "CoFID food code",
        "record_url": "https://www.gov.uk/government/publications/composition-of-foods-integrated-dataset-cofid",
        "carbohydrate_definition": "Available carbohydrate (starch and sugars; excludes fibre)",
    },
}

# FoodData Central nutrient ids: Energy (KCAL), Protein, Carbohydrate by
# difference, Total lipid (fat). CoFID columns: KCALS, PROT, CHO, FAT.
NUTRIENTS = {"calories_kcal": 1008, "protein_g": 1003, "carbohydrates_g": 1005, "fat_g": 1004}
COFID_COLUMNS = {"calories_kcal": "Energy (kcal) (kcal)", "protein_g": "Protein (g)",
                 "carbohydrates_g": "Carbohydrate (g)", "fat_g": "Fat (g)"}
BRANDED_COLUMNS = ["fdc_id", "description", "publication_date", "brand_owner", "brand_name",
                   "branded_food_category", "market_country", "serving_size", "serving_size_unit",
                   "household_serving_fulltext", "ingredients", "discontinued_date",
                   *(f"nutrient_{nid}" for nid in NUTRIENTS.values())]

UNAVAILABLE = "UNAVAILABLE"

# The reviewed research database. Pinned by hash so a changed decision
# always shows up as a deliberate edit here.
RESEARCH = {
    "file": "docs/research/FoodCal_AI_Nutrition_Database.json",
    "sha256": "5ab9762d812dbf844fc15d12215ee0aba799d26d42a9d38c272286cd18aec1a4",
    "report": "docs/research/FoodCal_AI_Nutrition_Report.pdf",
    "workbook": "docs/research/FoodCal_AI_Nutrition_Database.xlsx",
}
RESEARCH_DATASETS = {"FNDDS": "FNDDS", "SR Legacy": "SR", "Branded Foods": "BRANDED"}
# The research marks a record whose value it doubts by starting its limitations with this.
REVIEW_FLAG = "FLAGGED FOR REVIEW"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_path(key, check=True):
    meta = SOURCES[key]
    path = EXTERNAL / meta["file"]
    if not path.is_file():
        raise FileNotFoundError(f"{meta['file']} is missing from {EXTERNAL}; see docs/NUTRITION_DATA.md.")
    if check and sha256(path) != meta["sha256"]:
        raise ValueError(f"{path.name} does not match its recorded SHA-256.")
    return path


def read_zip_csv(zf, prefix, name):
    return csv.DictReader(io.TextIOWrapper(zf.open(prefix + name), encoding="utf-8"))


def load_fdc(key, wanted):
    """{record id: (description, {nutrient key: value})} from an FNDDS or SR Legacy ZIP."""
    path = source_path(key)
    prefix = path.name[:-4] + "/"
    with zipfile.ZipFile(path) as zf:
        # food_nutrient.csv names nutrients by id (SR Legacy, e.g. 1008) or by
        # nutrient number (FNDDS, e.g. 208); nutrient.csv maps both.
        codes = {}
        for r in read_zip_csv(zf, prefix, "nutrient.csv"):
            if r["id"] in {str(n) for n in NUTRIENTS.values()}:
                codes[r["id"]] = codes[str(int(float(r["nutrient_nbr"])))] = r["id"]
        if len(set(codes.values())) != len(NUTRIENTS):
            raise ValueError(f"{path.name}: nutrient.csv lacks one of the required nutrients.")
        names = {r["fdc_id"]: r["description"] for r in read_zip_csv(zf, prefix, "food.csv") if r["fdc_id"] in wanted}
        amounts = {}
        for r in read_zip_csv(zf, prefix, "food_nutrient.csv"):
            if r["fdc_id"] in wanted and r["nutrient_id"] in codes:
                food = amounts.setdefault(r["fdc_id"], {})
                if codes[r["nutrient_id"]] in food:
                    raise ValueError(f"{path.name}: FDC {r['fdc_id']} lists nutrient {codes[r['nutrient_id']]} twice.")
                food[codes[r["nutrient_id"]]] = r["amount"]
    by_key = {str(nid): key for key, nid in NUTRIENTS.items()}
    return {i: (names.get(i), {by_key[n]: v for n, v in amounts.get(i, {}).items()}) for i in wanted}


def load_branded(wanted):
    path = source_path("BRANDED")
    with open(path, newline="", encoding="utf-8") as handle:
        rows = {r["fdc_id"]: r for r in csv.DictReader(handle)}
    return {i: (rows[i]["description"], {k: rows[i][f"nutrient_{nid}"] for k, nid in NUTRIENTS.items()
                                         if rows[i][f"nutrient_{nid}"] != ""})
            for i in wanted if i in rows}


# ---- Minimal .xlsx reader (standard library only) --------------------------

XLSX_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
           "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}


def xlsx_rows(path, sheet_name):
    with zipfile.ZipFile(path) as z:
        shared = [
            "".join(t.text or "" for t in si.iter(f"{{{XLSX_NS['m']}}}t"))
            for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", XLSX_NS)
        ]
        targets = {r.get("Id"): r.get("Target") for r in ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))}
        sheet = next(s for s in ET.fromstring(z.read("xl/workbook.xml")).find("m:sheets", XLSX_NS)
                     if s.get("name") == sheet_name)
        member = "xl/" + targets[sheet.get(f"{{{XLSX_NS['r']}}}id")].lstrip("/").replace("xl/", "", 1)
        for row in ET.fromstring(z.read(member)).iter(f"{{{XLSX_NS['m']}}}row"):
            cells = {}
            for c in row.findall("m:c", XLSX_NS):
                value = c.find("m:v", XLSX_NS)
                text = None if value is None else (shared[int(value.text)] if c.get("t") == "s" else value.text)
                letters = re.match(r"[A-Z]+", c.get("r")).group(0)
                index = 0
                for ch in letters:
                    index = index * 26 + ord(ch) - 64
                cells[index - 1] = text
            yield [cells.get(i) for i in range(max(cells) + 1)] if cells else []


def load_cofid(wanted):
    rows = xlsx_rows(source_path("COFID"), "1.3 Proximates")
    header = next(rows)
    col = {name: header.index(name) for name in ("Food Code", "Food Name", *COFID_COLUMNS.values())}
    found = {}
    for row in rows:
        row += [None] * (len(header) - len(row))
        code = row[col["Food Code"]]
        if code in wanted:
            found[code] = (row[col["Food Name"]], {k: row[col[c]] for k, c in COFID_COLUMNS.items()})
    return found


LOADERS = {"FNDDS": lambda w: load_fdc("FNDDS", w), "SR": lambda w: load_fdc("SR", w),
           "BRANDED": load_branded, "COFID": load_cofid}


def number(value, where):
    """Source values must be plain numbers; 'Tr' (trace), 'N' (unknown) or blanks are refused."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{where}: value {value!r} is not a number.") from None
    if result < 0:
        raise ValueError(f"{where}: negative value {value!r}.")
    return result


def class_names():
    import torch  # only needed for the read-only class-list check
    return list(torch.load(CHECKPOINT, map_location="cpu", weights_only=True)["class_names"])


def load_research(names):
    """The research database, after checking it lists every model class exactly once."""
    path = ROOT / RESEARCH["file"]
    if sha256(path) != RESEARCH["sha256"]:
        raise ValueError(f"{RESEARCH['file']} does not match its recorded SHA-256.")
    research = json.loads(path.read_text(encoding="utf-8"))
    listed = [food["food_class"] for food in research["foods"]]
    if sorted(listed) != sorted(names):
        missing, extra = sorted(set(names) - set(listed)), sorted(set(listed) - set(names))
        raise ValueError(f"The research does not list each checkpoint class once (missing {missing}, extra {extra}).")
    return research


def research_reference(food):
    """(source key, record id) named by a research entry, or None when it is unavailable."""
    name, source = food["food_class"], food["source"]
    if food["match_type"] == UNAVAILABLE:
        if source["record_id"] is not None or any(v is not None for v in food["nutrition_per_100g"].values()):
            raise ValueError(f"{name}: unavailable in the research but it names a record or values.")
        return None
    key = RESEARCH_DATASETS.get(source["dataset"])
    record_id = (source["record_id"] or "").removeprefix("FDC ID ")
    if key is None or not record_id.isdigit():
        raise ValueError(f"{name}: unrecognised research source {source['dataset']!r} {source['record_id']!r}.")
    if source["url"] != FDC_URL.format(id=record_id):
        raise ValueError(f"{name}: research URL {source['url']!r} is not the URL of FDC {record_id}.")
    return key, record_id


def rounds_to(source_value, research_value):
    """True when research_value is source_value rounded to research_value's own precision."""
    research = Decimal(repr(float(research_value)))
    places = max(0, -research.normalize().as_tuple().exponent)
    return abs(Decimal(str(source_value)) - research) <= Decimal(5).scaleb(-places - 1)


def build():
    names = class_names()
    research = load_research(names)
    by_class = {food["food_class"]: food for food in research["foods"]}
    refs = {name: research_reference(by_class[name]) for name in names}

    wanted = {}
    for ref in refs.values():
        if ref:
            wanted.setdefault(ref[0], set()).add(ref[1])
    loaded = {key: LOADERS[key](ids) for key, ids in wanted.items()}

    foods = []
    for name in names:
        food, ref = by_class[name], refs[name]
        entry = {
            "food_class": name,
            "food_name": food["food_name"],
            "preparation_method": food["preparation_method"],
            "match_type": food["match_type"],
            "data_quality": food["data_quality"],
            "reference": None,
            "nutrition_per_100g": None,
            "limitations": food["limitations"],
            "notes": food["notes"],
            "flagged_for_review": food["limitations"].startswith(REVIEW_FLAG),
            "verification_status": food["verification_status"],
        }
        if ref:
            key, record_id = ref
            description, values = loaded[key].get(record_id, (None, {}))
            if description is None:
                raise ValueError(f"{name}: {key} record {record_id} is not in {SOURCES[key]['file']}.")
            missing = [k for k in NUTRIENTS if k not in values]
            if missing:
                raise ValueError(f"{name}: {key} record {record_id} has no value for {', '.join(missing)}.")
            stored = {k: number(values[k], f"{name} {k}") for k in NUTRIENTS}
            for k in NUTRIENTS:
                if not rounds_to(stored[k], food["nutrition_per_100g"][k]):
                    raise ValueError(f"{name}: research {k} {food['nutrition_per_100g'][k]} disagrees with "
                                     f"{key} record {record_id} ({stored[k]}).")
            entry["reference"] = {
                "source_key": key,
                "record_id": record_id,
                "food_description": description,
                "research_description": food["reference_description"],
                "mirror_url": food["source"]["mirror_url"],
                "carbohydrate_definition": food["source"]["carbohydrate_definition"],
            }
            entry["nutrition_per_100g"] = stored
        foods.append(entry)

    return {
        "schema_version": 3,
        "retrieved": RETRIEVED,
        "research": {**RESEARCH, "generated": research["generated"], "access_note": research["access_note"]},
        "sources": SOURCES,
        "nutrients": {"basis": "per 100 g of the food as eaten (edible portion)",
                      "fields": list(NUTRIENTS)},
        "foods": foods,
    }


def extract_branded(full_zip):
    """Copies the selected Branded Foods records verbatim from the full download."""
    meta = SOURCES["BRANDED"]
    if sha256(full_zip) != meta["full_download_sha256"]:
        raise ValueError(f"{full_zip} is not {meta['full_download']} (SHA-256 differs).")
    research = load_research(class_names())
    refs = (research_reference(food) for food in research["foods"])
    wanted = {ref[1] for ref in refs if ref and ref[0] == "BRANDED"}
    prefix = meta["full_download"][:-4] + "/"
    csv.field_size_limit(2**31 - 1)
    rows = {i: {"fdc_id": i} for i in wanted}
    with zipfile.ZipFile(full_zip) as zf:
        for r in read_zip_csv(zf, prefix, "food.csv"):
            if r["fdc_id"] in wanted:
                rows[r["fdc_id"]].update(description=r["description"], publication_date=r["publication_date"])
        for r in read_zip_csv(zf, prefix, "branded_food.csv"):
            if r["fdc_id"] in wanted:
                rows[r["fdc_id"]].update({k: r[k] for k in BRANDED_COLUMNS[3:12]})
        for r in read_zip_csv(zf, prefix, "food_nutrient.csv"):
            if r["fdc_id"] in wanted and r["nutrient_id"] in {str(n) for n in NUTRIENTS.values()}:
                rows[r["fdc_id"]][f"nutrient_{r['nutrient_id']}"] = r["amount"]
    out = EXTERNAL / meta["file"]
    with open(out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=BRANDED_COLUMNS)
        writer.writeheader()
        for i in sorted(rows, key=int):
            writer.writerow({c: rows[i].get(c, "") for c in BRANDED_COLUMNS})
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build the nutrition reference from the downloaded source files.")
    parser.add_argument("--extract-branded", type=Path, metavar="ZIP",
                        help="copy the selected records from the full USDA Branded Foods CSV download first")
    args = parser.parse_args(argv)
    try:
        if args.extract_branded:
            out = extract_branded(args.extract_branded)
            print(f"Wrote {out.relative_to(ROOT).as_posix()} (SHA-256 {sha256(out)})")
            print("Record that SHA-256 in SOURCES['BRANDED'] before building.")
            return 0
        reference = build()
    except (OSError, ValueError, StopIteration) as error:
        print(f"Cannot build the nutrition reference: {error}")
        return 1
    OUTPUT.write_text(json.dumps(reference, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    counts = {}
    for food in reference["foods"]:
        counts[food["match_type"]] = counts.get(food["match_type"], 0) + 1
    print(f"Wrote {OUTPUT.relative_to(ROOT).as_posix()}: {len(reference['foods'])} classes {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
