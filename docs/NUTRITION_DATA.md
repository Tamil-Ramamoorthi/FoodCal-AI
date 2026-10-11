# Nutrition data

FoodCal AI shows calories, protein, carbohydrates and fat for each food
class the model predicts. It reads them from a reviewed local reference
file and scales them to a serving weight the user enters (default 100 g).

**These are typical values for a reference food. They are not measurements
of the photographed meal, and the app never estimates portion size from an
image.**

## Where the data comes from

Two layers, kept separate on purpose:

1. **Decisions: the FoodCal AI nutrition research** (`docs/research/`,
   compiled 2026-10-11). For each of the 34 classes it chooses the USDA
   record, the match type, the data-quality rating, and the food name,
   preparation method, limitations, notes and verification status.
   - `FoodCal_AI_Nutrition_Report.pdf`: the report, with per-class reasoning
     and research gaps.
   - `FoodCal_AI_Nutrition_Database.xlsx`: the same data as a workbook
     (Nutrition Database, Source Register, Data Quality Summary, Calculation
     Guide sheets).
   - `FoodCal_AI_Nutrition_Database.json`: the same data as JSON. The build
     reads this file and pins its SHA-256 (`5ab9762d…c1a4`).

   The three files agree exactly: every value, rating, record id, URL and
   text field.
2. **Numbers: the USDA source files** in `data/external/`. The research read
   its records through a public FoodData Central mirror (getfoodfacts.com),
   because fdc.nal.usda.gov could not be reached from where it was compiled.
   So the build copies every stored number from the downloaded USDA files.
   It refuses to build unless each research value equals the source value
   rounded to the research's precision (for example 21.45 → 21.4). All 116
   values (29 records × 4 nutrients) pass this check.

## Sources and licences

| Key | Source | Licence | File (in `data/external/`) | SHA-256 |
|---|---|---|---|---|
| FNDDS | USDA FoodData Central: FNDDS 2021-2023, CSV release 2024-10-31 | Public domain (CC0 1.0); cite FoodData Central | `fdc/FoodData_Central_survey_food_csv_2024-10-31.zip` | `5ccc25ec…f29c` |
| SR | USDA FoodData Central: SR Legacy, 2018-04 | Public domain (CC0 1.0) | `fdc/FoodData_Central_sr_legacy_food_csv_2018-04.zip` | `b8081729…6fd0` |
| BRANDED | USDA FoodData Central: Branded Foods, 2026-04-30. Values are declared on manufacturers' labels | Public domain (CC0 1.0) | `fdc/branded_selected_2026-04-30.csv` (see below) | `d0fdafff…78d9` |
| COFID | McCance and Widdowson's Composition of Foods Integrated Dataset (CoFID) 2021, Public Health England | Open Government Licence v3.0, Crown copyright | `cofid/McCance_Widdowsons_Composition_of_Foods_Integrated_Dataset_2021.xlsx` | `436e9445…e97d` |

**CoFID is no longer used by any class.** The research uses USDA records
only. The file and loader are kept because the research expects CoFID
records may be added later. CoFID reports *available* carbohydrate, so any
such record must keep its own carbohydrate definition.

**Download.** All files were downloaded on 2026-10-11 from
<https://fdc.nal.usda.gov/download-datasets/> (USDA) and
<https://www.gov.uk/government/publications/composition-of-foods-integrated-dataset-cofid>
(CoFID; the page footer states the Open Government Licence v3.0).

**Branded Foods.** The full download is 449 MB
(`FoodData_Central_branded_food_csv_2026-04-30.zip`, SHA-256
`26050a5d…fc49`), so it is not kept. Instead, the selected records (food,
branded_food and food_nutrient rows) are copied verbatim into
`branded_selected_2026-04-30.csv`:

```
.venv-gpu\Scripts\python.exe -m backend.app.services.build_nutrition_reference --extract-branded <full ZIP>
```

That command checks the full ZIP's hash, then copies the Branded records the
research names: 484203 (kadai_paneer), 1862742 (dal_makhani) and 2677881
(pav_bhaji). The ZIP was downloaded again on 2026-10-11 to add 484203 and
1862742, and its hash matched the one recorded above.

**Licensing notes from the research.** The research marks the FoodData
Central terms and the mirror's terms "to verify", because it could not open
either. The USDA download page states the CC0 licence used above. Nothing
from the mirror is stored except the record URLs. Branded product and brand
names are manufacturer trademarks, and a branded product is never presented
as the universal composition of a dish.

**Runtime.** No API key is used and the app makes no external request.

### Sources considered and not used

- **IFCT 2017 (ICMR-NIN).** Checked by the research: its 543 records cover
  raw and basic foods only, with no prepared dishes.
- **Indian Nutrient Databank (INDB, 2024).** Its data repository shows no
  licence, and the data derives largely from IFCT 2017.
- **Community calorie-counter and recipe sites.** User-contributed, with no
  documented methodology and restrictive terms.

## Nutrients and definitions

Values are per 100 g of the food as eaten (edible portion).

| Field | FoodData Central | Branded Foods |
|---|---|---|
| `calories_kcal` | Energy (KCAL), nutrient id 1008 / number 208 | Same nutrient, from the label |
| `protein_g` | Protein, 1003 / 203 | Same, from the label |
| `carbohydrates_g` | Carbohydrate, **by difference** (includes fibre), 1005 / 205 | **Total carbohydrate as declared on the label** (includes fibre; label-rounded) |
| `fat_g` | **Total** lipid (fat), 1004 / 204 (not saturated fat) | Same, from the label |

- **Carbohydrate definitions.** Each record stores its own
  `carbohydrate_definition`, taken from the research. The API returns it and
  the UI shows it under the values.
- **Data quirks the build handles:** FNDDS's `food_nutrient.csv` refers to
  nutrients by number, while SR Legacy and Branded use the id. The build
  maps both through each release's `nutrient.csv`.

## How the reference is built

- **`backend/app/services/build_nutrition_reference.py`**:
  - checks the SHA-256 of the research JSON and every source file it uses;
  - requires the research to list exactly the production checkpoint's 34
    classes (it reads the checkpoint but never writes it);
  - takes each class's record, match type, quality and text from the
    research, without changing them;
  - copies every number from the source file and stops if any research
    value disagrees with it;
  - sets `flagged_for_review` when the research's limitations start with
    "FLAGGED FOR REVIEW" (only pakode).
- **`backend/app/services/nutrition_reference.json`** is the generated file
  (schema version 3). It is the single source of truth at runtime.
- **`backend/app/services/nutrition.py`** validates the reference at
  startup:
  - 34 classes, no duplicates;
  - non-negative, finite numbers;
  - no values on unavailable records;
  - a source, licence, record id, carbohydrate definition, food name,
    preparation method, limitations, notes and verification status on every
    record.

  It also checks serving sizes and scales the values.

To rebuild after the research or a source file changes:

```
.venv-gpu\Scripts\python.exe -m backend.app.services.build_nutrition_reference
```

`tests/test_nutrition.py` rebuilds the file in memory from the source files
and checks that it is identical to the stored file. It also checks that the
stored decisions and text are the research's, verbatim.

## Match types and data quality

The ratings are the research's own assessments, not formal validation.

| Type | Meaning | `data_quality` |
|---|---|---|
| `DIRECT_MATCH` | A record for the same dish | `HIGH`, `MEDIUM` or `LOW` |
| `APPROXIMATE_MATCH` | A related food, or one component of the dish. Always labelled "Approximate", with the reason | `MEDIUM` when the related food is close (plain naan for butter naan, a filled dosa, a steamed dumpling, milk tea); `LOW` when it is a single branded product or a different dish (paneer makhani for kadai paneer, a generic wrap for a kathi roll). **Never `HIGH`** |
| `UNAVAILABLE` | No sufficiently reliable record. No values are stored or shown, never zeros | `NONE` |

In the app, approximate matches, LOW-quality references and values flagged
for review get a visible warning.

## Coverage: 34 classes

- 21 direct matches
- 8 approximate matches
- 5 unavailable
- Quality: 11 HIGH, 14 MEDIUM, 4 LOW, 5 NONE

Values are per 100 g, as stored from the source file (rounded only for
display).

| Class | Match, quality | Source record | kcal | Protein g | Carbs g | Fat g |
|---|---|---|---|---|---|---|
| Baked Potato | Direct, HIGH | FNDDS 2709383 "Potato, baked, NFS" | 93 | 1.95 | 21.45 | 0.1 |
| Crispy Chicken | Direct, HIGH | SR Legacy 170718 "Fast foods, chicken, breaded and fried, boneless pieces, plain" | 307 | 15.92 | 14.93 | 20.36 |
| Donut | Direct, MEDIUM | FNDDS 2708062 "Doughnut, NFS" | 426 | 5.52 | 49.52 | 22.86 |
| Fries | Direct, HIGH | FNDDS 2709456 "Potato, french fries, NFS" | 225 | 2.5 | 23.23 | 14.07 |
| Hot Dog | Direct, HIGH | FNDDS 2707056 "Hot dog sandwich, NFS, on white bun" | 296 | 10.85 | 23.73 | 17.37 |
| Sandwich | Direct, MEDIUM | FNDDS 2706880 "Sandwich, NFS" | 287 | 12.75 | 33.83 | 12.03 |
| Taco | Direct, MEDIUM | SR Legacy 170689 "Fast foods, taco with beef, cheese and lettuce, hard shell" | 226 | 8.86 | 19.85 | 12.7 |
| Taquito | Direct, MEDIUM | FNDDS 2708601 "Taquito, chicken" | 269 | 12.53 | 31.61 | 10.41 |
| apple_pie | Direct, HIGH | FNDDS 2707995 "Pie, apple" | 296 | 2.7 | 37.55 | 15.28 |
| burger | Direct, HIGH | FNDDS 2706932 "Hamburger, on white bun, 1 medium patty" | 288 | 17.51 | 20.68 | 14.38 |
| butter_naan | **Approx.**, MEDIUM | FNDDS 2707613 "Bread, naan" (plain naan) | 311 | 11.09 | 50.23 | 7.28 |
| chai | **Approx.**, MEDIUM | FNDDS 2710506 "Tea, hot, with milk" | 51 | 1.58 | 8.95 | 1.05 |
| chapati | Direct, MEDIUM | SR Legacy 174075 "Bread, chapati or roti, whole wheat, commercially prepared, frozen" | 299 | 7.85 | 46.13 | 9.2 |
| cheesecake | Direct, HIGH | SR Legacy 172711 "Cake, cheesecake, commercially prepared" | 321 | 5.5 | 25.5 | 22.5 |
| chicken_curry | Direct, MEDIUM | FNDDS 2706437 "Chicken curry" | 107 | 6.48 | 6.54 | 6.48 |
| chole_bhature | Unavailable | — | — | — | — | — |
| dal_makhani | **Approx.**, LOW | Branded 1862742 "DAL MAKHANI" (Mother India) | 140 | 6 | 14 | 6.67 |
| dhokla | Unavailable | — | — | — | — | — |
| fried_rice | Direct, MEDIUM | FNDDS 2708952 "Rice, fried, NFS" | 174 | 3.84 | 32.5 | 3.19 |
| ice_cream | Direct, HIGH | FNDDS 2705630 "Ice cream, vanilla" | 207 | 3.5 | 23.6 | 11 |
| idli | Direct, HIGH | FNDDS 2708346 "Idli" | 128 | 6.36 | 24.98 | 0.35 |
| jalebi | Unavailable | — | — | — | — | — |
| kaathi_rolls | **Approx.**, LOW | FNDDS 2706881 "Sandwich wrap, NFS" (generic wrap) | 232 | 12.77 | 23.3 | 9.4 |
| kadai_paneer | **Approx.**, LOW | Branded 484203 "PANEER MAKHANI" (Haldiram) | 204 | 4.93 | 9.15 | 16.2 |
| kulfi | Unavailable | — | — | — | — | — |
| masala_dosa | **Approx.**, MEDIUM | FNDDS 2709129 "Dosa, with filling" | 184 | 5.46 | 30.8 | 4.27 |
| momos | **Approx.**, MEDIUM | FNDDS 2708708 "Wonton, dumpling or pot sticker, steamed" | 113 | 6.9 | 11.07 | 4.6 |
| omelette | Direct, HIGH | SR Legacy 172185 "Egg, whole, cooked, omelet" | 154 | 10.57 | 0.64 | 11.66 |
| paani_puri | Unavailable | — | — | — | — | — |
| pakode | Direct, MEDIUM, **flagged for review** | FNDDS 2710066 "Pakora" | 125 | 4.92 | 16.24 | 4.77 |
| pav_bhaji | **Approx.**, LOW | Branded 2677881 "AUTHENTIC INDIAN MASHED VEGETABLE CURRY PAV BHAJI" (bhaji only) | 136 | 2.4 | 8 | 10.4 |
| pizza | Direct, HIGH | SR Legacy 173292 "Fast Food, Pizza Chain, 14" pizza, cheese topping, regular crust" | 266 | 11.39 | 33.33 | 9.69 |
| samosa | Direct, MEDIUM | FNDDS 2708730 "Samosa" | 310 | 5.14 | 33.16 | 17.47 |
| sushi | Direct, MEDIUM | FNDDS 2708959 "Sushi, NFS" | 94 | 2.92 | 18.39 | 0.67 |

Each record's preparation method, limitations and notes are in
`nutrition_reference.json` and are shown in the app.

**Energy check.** For every available record, 4·protein + 4·carbs + 9·fat
is within 3.2% of the stored energy (the research's own check). No value
was adjusted to fit it.

## Adopting the research database (2026-10-11)

The research replaced the previous reference (schema version 2), which had
been built from the same USDA files plus UK CoFID. 10 classes are unchanged
(Baked Potato, Donut, fried_rice, pakode, pav_bhaji, sushi, apple_pie,
dhokla, jalebi, paani_puri). The other 24 changed as follows. Each change
follows the research. The reasons are the research's, summarised from its
limitations and notes.

**Source record or availability changed (16 classes)**

| Class | Before | Now | Reason (from the research) |
|---|---|---|---|
| Crispy Chicken | Direct MEDIUM, FNDDS 2705949, 251 kcal | Direct HIGH, SR 170718, 307 kcal | Explicitly breaded, fried, plain boneless pieces, matching "crispy chicken" |
| Taco | Direct MEDIUM, FNDDS 2708514, 246 kcal | Direct MEDIUM, SR 170689, 226 kcal | Hard-shell beef taco; USDA taco records cluster at 206–229 kcal |
| burger | Direct MEDIUM, FNDDS 2706920 | Direct HIGH, FNDDS 2706932 | "Hamburger, on white bun, 1 medium patty"; same 288 kcal |
| butter_naan | Approx. LOW, SR 171845, 291 kcal | Approx. MEDIUM, FNDDS 2707613, 311 kcal | Still plain naan. True butter naan is higher in fat; no source quantifies the gap |
| chapati | Direct HIGH, FNDDS 2707713 | Direct MEDIUM, SR 174075 | Whole-wheat chapati; commercially prepared, so it may overstate fat for home roti. Same 299 kcal |
| cheesecake | Direct MEDIUM, FNDDS 2707861, 399 kcal | Direct HIGH, SR 172711, 321 kcal | Plain commercially prepared cheesecake |
| chicken_curry | Direct MEDIUM, CoFID 19-322 takeaway average, 145 kcal | Direct MEDIUM, FNDDS 2706437, 107 kcal | USDA survey recipe; the research notes it is lower than many restaurant curries |
| chole_bhature | Approx. LOW, CoFID chickpea curry only, 112 kcal | **Unavailable** | Chickpea curry alone must not represent the complete dish; no bhatura record exists |
| dal_makhani | Direct LOW, Branded 1852629, 127 kcal | Approx. LOW, Branded 1862742 (Mother India), 140 kcal | One manufacturer's ready-meal; other products range from 105 to 225 kcal |
| ice_cream | Direct MEDIUM, FNDDS 2705629 "Ice cream, NFS", 221 kcal | Direct HIGH, FNDDS 2705630 "Ice cream, vanilla", 207 kcal | Vanilla, regular fat, as the standard reference flavour |
| kaathi_rolls | Unavailable | **Approx. LOW**, FNDDS 2706881 generic wrap, 232 kcal | No kathi roll record exists; a generic wrap most likely understates fat and energy, so it is labelled indicative only |
| kadai_paneer | Unavailable | **Approx. LOW**, Branded 484203 paneer makhani, 204 kcal | No kadai paneer record exists; paneer makhani probably overstates fat |
| kulfi | Direct LOW, CoFID 12-606 homemade, 457 kcal | **Unavailable** | No verifiable prepared-kulfi record; ice cream is deliberately not substituted |
| masala_dosa | Direct LOW, Branded 2396366 frozen masala dosa, 273 kcal | Approx. MEDIUM, FNDDS 2709129 "Dosa, with filling", 184 kcal | A filled dosa, but the filling is not stated to be potato masala |
| omelette | Direct MEDIUM, FNDDS 2707198, 185 kcal | Direct HIGH, SR 172185, 154 kcal | Plain whole-egg omelette |
| pizza | Direct MEDIUM, FNDDS 2708614 | Direct HIGH, SR 173292 | Pizza-chain cheese pizza, regular crust; same 266 kcal |

**Same record; match type or quality changed (8 classes)**

| Class | Before | Now |
|---|---|---|
| Sandwich | Approx. LOW | Direct MEDIUM (the research treats the generic NFS record as the class reference, with a warning that sandwiches vary from 232 to 382 kcal) |
| Taquito | Approx. LOW | Direct MEDIUM (chicken filling) |
| Fries, Hot Dog, idli | Direct MEDIUM | Direct HIGH |
| samosa | Direct HIGH | Direct MEDIUM (the record does not state the filling) |
| chai, momos | Approx. LOW | Approx. MEDIUM |

**Points that need attention**

- **pakode is flagged for review.** The USDA "Pakora" record gives 125 kcal
  and 4.77 g fat per 100 g, against about 250–320 kcal commonly cited for
  Indian deep-fried pakoda. Its own values are internally consistent, so the
  research treats it as a recipe difference, not a data error. The app shows
  a "Flagged for review" warning.
- **Approximate matches are labelled with their reason** in the app:
  butter_naan (plain naan), chai (not masala chai), dal_makhani,
  kadai_paneer and pav_bhaji (single branded products; pav bhaji is bhaji
  only), kaathi_rolls (generic wrap), masala_dosa (filling not specified)
  and momos (generic steamed dumpling).
- **dal_makhani** (1862742) is filed by USDA under "Pizza Mixes & Other Dry
  Dinners". Its ingredients (black gram, kidney beans, butter, cream) and
  140 kcal per 100 g fit a prepared ready-meal, not a dry mix (dry mixes are
  about 350–400 kcal).
- **Branded per-100 g values** are USDA's, derived from the declared
  serving. The research's figures, scaled from the same serving, agree with
  them after rounding.
- **The research's verification status is kept verbatim** ("read via FDC
  mirror, not inspected on USDA site"). The app also states that the value
  was checked against the downloaded USDA file, which the build guarantees.

## API

- **`GET /api/nutrition`** returns coverage, counts, sources, licences and
  the research file's details for all 34 classes. Each food includes its
  food name, preparation method, match type, quality, review flag and
  limitations.
- **`GET /api/nutrition/{food_class}?grams=<g>`** returns reference values
  per 100 g and for the serving, plus:
  - the food name and preparation method;
  - match type, data quality and `flagged_for_review`;
  - the source record (canonical USDA URL, mirror URL, research and USDA
    descriptions, carbohydrate definition);
  - limitations, notes, verification status and a notice.

  Rules:
  - `grams` is required and must be finite, more than 0 and at most 2000.
    Otherwise the response is 422. The frontend defaults it to 100.
  - A class that isn't one of the 34 returns 404, with
    `"<name>" is not one of the 34 food classes.`
  - A valid class without data returns 200 with `available: false` and
    null values.
  - If the reference file is missing or invalid, both endpoints return 503.
    `GET /` then reports `"nutrition_reference": "unavailable"`, and
    `/api/predict` and the rest of the app keep working.
- All fields added in schema version 3 are additions; no existing field was
  removed or renamed.
- **Calculation:** `serving = per_100g × grams / 100`. It is computed with
  `Decimal` from the unrounded stored value. Calories are rounded half-up to
  whole kcal and grams to 0.1 g. Unavailable values are never scaled.
- **`/api/predict` is unchanged.**

### "Nutrition could not be loaded. Not Found (HTTP 404)"

- **Cause:** FastAPI's generic `{"detail": "Not Found"}` means the running
  backend has no such route. It was started before the nutrition endpoints
  were added, and it runs without `--reload`.
- **What the frontend shows:** "The backend at … has no nutrition endpoints
  (HTTP 404). Restart the FoodCal AI backend so it runs the current code."
- **Fix:** restart the backend.

## Limitations

- **Values describe a typical reference food,** not the photographed meal.
  Recipes, oil, fillings, toppings and restaurant versions differ, sometimes
  greatly. Every class has limitations, listed in the app.
- **Branded records are one manufacturer's label values** (dal_makhani,
  kadai_paneer, pav_bhaji), so they are LOW quality and approximate.
- **Approximations can understate or overstate a dish:**
  - kaathi_rolls likely understates;
  - kadai_paneer likely overstates fat;
  - butter_naan and chai likely understate fat;
  - pav_bhaji excludes the pav.
- **pakode's value is unusually low** for deep-fried pakoda (flagged for
  review).
- **Five classes have no value** (chole_bhature, dhokla, jalebi, kulfi,
  paani_puri). The remaining gaps are listed in the research report's
  section 9.
- **Results depend on the classifier being right.** The UI warns on
  low-confidence predictions and lets the user pick another top prediction.
- **Serving size is entered by the user.** Portion estimation from images
  is out of scope.
