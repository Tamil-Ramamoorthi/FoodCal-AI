# Real-world evaluation

Evaluates the existing EfficientNet-B0 checkpoint
(`models/efficientnet_b0/food_efficientnet_b0_best.pth`) on independent,
manually labeled photos. Nothing here trains the model or modifies the
processed train / validation / test images.

Run every command from the project root with the project's environment,
`.venv-gpu\Scripts\python.exe`. A bare `python` may resolve to a different
environment (on this machine it resolves to `FoodCal-AI\.venv`). The
checkpoint is always passed with `--checkpoint <path>`, never as a
positional argument.

## 1. Add photos

```
data/real_world_test/
    <class name>/photo.jpg      # one folder per class, exact names below
    provenance.csv              # where each photo came from
```

- Folder names must match the checkpoint's 34 class names exactly
  (including case and spaces, e.g. `Baked Potato`, `apple_pie`). The folders
  already exist.
- Supported: `.jpg`, `.jpeg`, `.png`, `.webp`. Other files are reported and
  ignored, and unreadable images are reported and excluded.
- Aim for at least 10 photos per class (`--min-per-class`). With fewer
  images, per-class metrics are flagged as unreliable.
- Use photos that are not from the training source (the Hugging Face
  dataset `ewanlong/Food_Classification_Dataset`), e.g. your own camera.

`provenance.csv` needs the columns `file` and `source`. Extra columns are
allowed:

```csv
file,source,captured_by,date,license,notes
pizza/IMG_0412.jpg,own phone camera,me,2026-10-12,own,
samosa/samosa_01.webp,own phone camera,me,2026-10-12,own,
```

`file` is relative to `data/real_world_test/`. Until every image has a
documented source, the independence status remains `provenance_incomplete`.

## 2. Audit the data (read-only)

```
.venv-gpu\Scripts\python.exe -m evaluation.dataset_audit
```

Writes `reports/real_world/dataset_audit.json` and `.md` with:
- images per class and split
- missing and unexpected folders
- invalid and unsupported files
- exact and near-duplicates within each set
- overlap of real-world photos with the processed splits and the original
  source archive

No file is moved or deleted.

## 3. Evaluate

```
.venv-gpu\Scripts\python.exe -m evaluation.evaluate_real_world
```

Writes `reports/real_world/real_world_evaluation.json` and `.md`. These
are separate from the offline test-set report in
`models/efficientnet_b0/evaluation_report.json`. The dashboard (Model
Evaluation → Source 2) reads the JSON through
`GET /api/evaluation/real-world`. Reload the page after re-running.

Useful options: `--device cpu|cuda`, `--min-per-class N`,
`--near-distance N`, `--source-archive none|auto|<zip path>`, `--no-cache`.

### Another checkpoint (e.g. the v2 candidate) and a comparison

```
.venv-gpu\Scripts\python.exe -m evaluation.evaluate_real_world --checkpoint models/efficientnet_b0_v2/food_efficientnet_b0_v2_best.pth --output-dir reports/real_world_v2
.venv-gpu\Scripts\python.exe -m evaluation.compare_real_world
```

- Any checkpoint other than production needs its own `--output-dir`;
  `reports/real_world/` is reserved for the deployed model's dashboard
  report.
- Every checkpoint must have the production class list (same 34 names, same
  order) and input size; otherwise the evaluation stops.
- Each report records an image manifest (count + SHA-256 of every
  path, label and file hash). `compare_real_world` refuses to compare two
  reports unless they evaluated exactly the same images. It writes
  `reports/real_world_comparison/comparison.md` and `.json` (overall and
  per-class changes, paired McNemar test, bootstrap CIs, frequent
  confusions, changed predictions, limitations) and both confusion matrices
  as CSV, or a **pending** report while either evaluation has no valid
  images.

How it works:
- **Preprocessing** is shared with the API in
  `backend/app/models/classifier.py`: EXIF orientation, RGB, LANCZOS
  resize to 224×224, then ImageNet normalization. This mirrors
  `training/prepare_dataset.py` and `training/evaluate_efficientnet.py`.
- **Correctness** compares the top-1 prediction with the folder label.
  Model confidence is reported separately and never used for accuracy.
- **Undefined values are `null`.** Recall and F1 are `null` for classes with
  no images, and precision is `null` for classes that were never predicted.
  Macro F1 averages only classes with images, and the report says how many.
- **Overlapping images.** If any images overlap the training sources, metrics are also reported
  with those images excluded.

### Cleaned evaluation (duplicates, overlaps, manual review)

```
.venv-gpu\Scripts\python.exe -m evaluation.clean_real_world
```

- **Inputs:** the two reports above and the review sheet
  `data/real_world_review/manual_review.csv`.
- **Outputs, all in `reports/real_world_clean/`:**
  - `clean_manifest.csv`
  - `exclusions.csv`, which gives the reason for each excluded image
  - `clean_evaluation.md` and `.json`, which give both models' metrics on
    the same kept images
- **What it never changes:** images, labels and the original reports. It
  doesn't re-run either model.
- **Automatic exclusions:**
  - Extra exact copies with the same label.
  - Files that are in two class folders. These stay excluded and pending
    until you mark exactly one copy `keep`, or every copy `exclude`.
  - Possible overlaps with model data, until you mark them `keep`.
- **Review decisions:** fill in `decision` (`keep`/`exclude`) and
  `reviewer` in the sheet. The result stays **provisional** until every
  flagged image has a decision.
- **Details:** see `docs/REAL_WORLD_EVALUATION_AUDIT.md`.

## Duplicate detection and its limits

- **Exact duplicates:** SHA-256 of the file bytes. Any re-encode, resize,
  crop or metadata edit produces a different hash.
- **Possible near-duplicates:** 64-bit difference hash (dHash), Hamming
  distance ≤ 3 by default. `python -m evaluation.calibrate_dhash` measured
  this on the project data:
  - 99.9% of copies re-created the way `prepare_dataset.py` made them were
    caught.
  - 99.7% of re-saved copies (≤1280 px, JPEG q80) were caught.
  - 0.38% of processed images have a different-label image this close
    (2.99% at distance 6), so some matches are false alarms.
- **Not detected:** crops, rotations, mirroring, heavy edits, screenshots
  and different photos of the same plate.
- **No action is taken on matches.** They are reported for manual review only.
- **Overlap checks cover only known sources:** the processed splits and the
  cached source ZIP. A photo copied from elsewhere on the web may still have
  been seen in ImageNet pre-training or similar data.

## Tests

```
.venv-gpu\Scripts\python.exe -B -m unittest discover -s tests -v
cd frontend && npm test && npm run lint && npm run build
```

## Reliability diagnostics and the low-confidence warning

```
.venv-gpu\Scripts\python.exe -m evaluation.diagnose_reliability
```

Read-only. Writes `reports/reliability/reliability_diagnostics.json` and
`.md`. Everything goes through the API's own preprocessing, and it checks:
- class mapping
- validation and test accuracy, and the most frequent confusions
- how well confidence matches observed accuracy, per confidence band
- the effect of train/test duplicates
- synthetic domain-shift probes (zoom, blur, lighting, rotation, JPEG)

The API flags a prediction as `low_confidence` when its top-1 confidence is
below `FOODCAL_LOW_CONFIDENCE_THRESHOLD` (percent, default 50). The default
was chosen on the validation split only. It is the largest 5-point step at
which predictions below it were right less than half the time: 48.7% on
validation, and 47.3% on the test split, which was used only to check it.

The flag does not detect foods outside the 34 classes. Those are always
mapped to one of the 34 and can still score high.

## Adding training images (intake)

`evaluation/intake.py` registers and checks **new training/validation
images**. It never moves, edits or deletes an image and never sets a label;
labels come only from the folder you put an image in and the reviewer's
`decision`. Photos for the independent real-world test set do **not** go
here; they go to `data/real_world_test/` (section 1).

```
.venv-gpu\Scripts\python.exe -m evaluation.intake init 2026-10-15_phone_idli
#   put originals in data/intake/2026-10-15_phone_idli/images/<class>/
.venv-gpu\Scripts\python.exe -m evaluation.intake sync 2026-10-15_phone_idli
#   fill in intake.csv (Excel is fine; format capture_date as text)
.venv-gpu\Scripts\python.exe -m evaluation.intake validate 2026-10-15_phone_idli [--model-hint]
```

`sync` appends one row per new image (`file`, `proposed_label` = folder) and
never changes existing rows. Columns to fill in:

| Column | Required | Values |
|---|---|---|
| `source`, `license`, `captured_by` | yes | free text, e.g. `own phone camera`, `own`, your name |
| `capture_date` | yes | `YYYY-MM-DD`, not in the future |
| `session_id` | yes | same dish / same sitting = same id; photos of one plate share one id so they always land in the same split |
| `serving` | tag | single, several, with_sides, thali, partly_eaten |
| `variant` | tag | standard, mini, stuffed_or_rava, cut_or_sliced, other |
| `garnish` | tag | none, herbs, chilli, nuts, saffron, sauce, mixed, other |
| `plate` | tag | white_plate, steel_plate, banana_leaf, dark_plate, bowl, takeaway, other |
| `background` | tag | table, cloth, counter, restaurant, cluttered, other |
| `lighting` | tag | daylight, warm_indoor, dim, flash, harsh_shadow |
| `angle` | tag | top_down, angle_45, side |
| `distance` | tag | close_up, whole_plate, table_scene |
| `reviewer` | with a decision | who looked at the image |
| `decision` | for use | `accept`, `reject` or `relabel:<class>` (exact class name) |
| `notes` | no | anything |

Tags may be left blank (reported as untagged); a filled tag must be one of
the listed values. `validate` writes `reports/intake/<batch_id>/validation.md`
and `.json` and exits with code 1 if anything has an error.

**Errors** (the image cannot be used until fixed or rejected):
- unreadable file, not JPEG/PNG/WEBP, short side under 448 px
- `proposed_label` not a class or not matching its folder; missing provenance;
  bad date, tag, decision or a decision without a reviewer
- an exact copy of any processed image, real-world test photo, image in
  another batch or in this batch
- a possible near-duplicate (dHash <= 3) of a **validation, test or
  real-world test** image: excluded so it cannot leak into an evaluation set
- image files on disk that are not listed in `intake.csv` (batch error)

**Warnings** (a person should look; the image can still be used):
- possible near-duplicate of a training image or an image in another batch
- near-duplicate within the batch with a different label or `session_id`
- possibly blurred, very dark/bright or low contrast. These are uncalibrated
  heuristics; on 600 processed test images 2.7% got one
- untagged presentation axes
- with `--model-hint`: the v2 model's top-1 differs from the label at
  >= 50%. The Phase 1 audit found confident disagreements were usually the
  model's mistake, so this is only a prompt to look again.

Only rows with status **ready** (decision `accept`/`relabel`, no errors)
may go into a future training manifest. Building that manifest and
assigning sessions to train/validation is a separate, approved step.
