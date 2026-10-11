# Real-world evaluation audit

Date: 2026-10-11. Scope: the 263-image real-world evaluation of the
production EfficientNet-B0 and the v2 candidate.

The audit was read-only for everything that already existed:
- no image was moved, renamed, relabelled or deleted
- no report was regenerated
- no checkpoint was touched; SHA-256 is still `39363ab4…` for production and `e34b83ee…` for v2
- no model was re-run

## 1. Summary

| | Production | v2 |
|---|---|---|
| **As reported (263 images)** | | |
| Correct | 144 | 177 |
| Accuracy | 54.75% | 67.30% |
| Macro F1 | 48.46% | 63.15% |
| **Cleaned, provisional (240 images, section 4)** | | |
| Correct | 127 | 157 |
| Accuracy | 52.9% | 65.4% |
| Macro F1 | 47.3% | 62.1% |

- **The reported numbers are correct for the 263 files.** An independent
  recomputation with scikit-learn from the per-image predictions reproduces
  every figure.
- **Both models saw exactly the same images, labels and class mapping.**
- **The 263 files are not 263 independent images.**
  - 15 are byte-identical extra copies; 13 of them are one sushi photo.
  - One file sits in two class folders with different labels.
  - 6 are copies of photos in the model's training or validation data:
    5 of the 6 flagged by the evaluation, plus 1 iStock photo that the
    check missed.
- **After removing those, v2 is still clearly better on this sample.**
  - It fixes 39 images that production gets wrong and breaks 9 (exact
    McNemar p = 1.5e-5).
  - Bootstrap 95% CI: accuracy +7.1 to +18.3 points, macro F1 +8.9 to
    +20.2 points.
- **The cleaned result is provisional.** 16 flagged images still need a
  person to decide (section 4.3), but none of those decisions can
  overturn the conclusion (section 4.4).
- **No pipeline bug was found.** The large-image warning comes from a
  legitimate 121-megapixel photo that the pipeline handled within Pillow's
  default safeguards (section 3).

## 2. Report verification (Task 1)

| Check | Method | Result |
|---|---|---|
| Same images for both models | Compared path, label and SHA-256 of all 263 predictions in both JSON reports; recomputed the manifest digest | Identical. Both digests are `b4ed5ce0a4807567…`, and `comparison.json` records both checks as true |
| Files unchanged since evaluation | Re-hashed all 263 files on disk | All 263 match the reports, and no other file is in `data/real_world_test/` |
| Ground truth = folder | Each `actual` against its parent folder name | 263/263 match |
| Class mapping | `class_mapping` in both reports | Identical: 34 classes, same order. Markdown shows `apple_pie` as "apple pie" (display only) |
| Prediction fields | `correct == (actual == predicted)`; `predicted` / `confidence` equal top-1 | Consistent in both reports |
| Accuracy, macro / weighted F1, macro P / R | Recomputed by hand and with `sklearn.metrics` over the classes with images | Exact match. Production 144/263 = 54.75%, macro F1 48.46%. v2 177/263 = 67.30%, macro F1 63.15% |
| Confusion matrices | Rebuilt from predictions; compared with the JSON and `confusion_matrix_old/new.csv` | Exact match; each sums to 263 |
| Paired test | Recounted discordant pairs; `scipy.stats.binomtest` | 9 vs 42, p = 3.4e-6, as reported. 105 changed predictions, as listed |
| Duplicate / near-duplicate warnings | `quality` section of the JSON | Every warning names concrete files (section 3) |

Notes:
- **Macro precision** counts a class that was never predicted as 0, while the
  per-class table shows "n/a" for it. This is scikit-learn's
  `zero_division=0` convention.
  - Production never predicts Baked Potato, Taquito or chole bhature. Its
    macro precision is 53.0% this way, or 58.1% if those classes were
    skipped instead.
  - Macro F1 and recall are unaffected.
- **Two report details are out of date or misleading:**
  - The reports' "Reproduce" line says `python -m …`. On this machine the
    command must use `.venv-gpu\Scripts\python.exe`.
  - `reports/real_world/dataset_audit.json` is from 2026-10-10, before the
    photos were added, so it is stale. The evaluation JSON carries its own,
    current quality section.

## 3. Data quality (Task 2)

### 3.1 Exact duplicates (SHA-256), 3 groups and 15 extra copies

| Group | Files | Effect |
|---|---|---|
| apple pie | `apple_pie/pexels-hilalbulbul-12782728.jpg` and `… (1).jpg` | One photo counted twice. Both models are wrong on both: production says chapati, v2 says Sandwich |
| sushi | `sushi/pexels-valeriya-28291554.jpg` and `… (1)` to `… (13).jpg` | **One photo counted 14 times.** Sushi has 15 images but only 2 distinct photos. Both models are right on all 14, which inflates both accuracies equally |
| **conflicting labels** | `idli/pexels-ryshy-s-2149956588-35539315.jpg` = `masala_dosa/pexels-ryshy-s-2149956588-35539315.jpg` | One file, two labels. It is a mixed South Indian meal: a dosa, a round podi-coated item, small idlis in sambar and coffee. Both models predict pav bhaji. A single label is ambiguous; **a person must decide** |

There are no near-duplicate pairs within the set (dHash ≤ 3).

### 3.2 Overlap with the model's own data

The evaluation compared every photo with the 23,873 processed
train/validation/test images using SHA-256 and dHash ≤ 3.
- **The source ZIP is no longer in the Hugging Face cache,** so this run did
  not check it.
- **Archive hashes from an earlier run were used instead.** I checked the
  photos against those cached hashes in `reports/real_world/cache/` (23,873
  archive entries) and at a looser dHash ≤ 8. I then looked at every
  same-label candidate side by side.

| Real-world file | Matched file | Distance | Visual check |
|---|---|---|---|
| `Donut/pexels-tijana-drndarski-449691-3338681.jpg` | `processed/train/Donut/000386.jpg` | 0 | Same photo |
| `Hot Dog/pexels-kamila-bairam-80608214-12123657.jpg` | `processed/validation/Hot Dog/000006.jpg` and `000160.jpg` | 0 | Same photo (it is also duplicated inside validation). Validation selected both checkpoints, so this image is not independent |
| `Hot Dog/pexels-polina-tankilevitch-4518656.jpg` | `processed/train/Hot Dog/000501.jpg` | 2 | Same photo |
| `Sandwich/pexels-karola-g2-6467.jpg` | `processed/train/Sandwich/000568.jpg` | 0 | Same photo |
| `momos/pexels-bishop-tamrakar-2246201-3926123.jpg` | `processed/train/momos/000219.jpg` | 1 | Same photo |
| `masala_dosa/pexels-jack-baghel-2199968-20422126.jpg` | `processed/train/Crispy Chicken/000432.jpg` | 2 | **Different photos:** a dosa on a dark background vs a tray of puffed-rice bars. A dHash false alarm |
| `kulfi/istockphoto-657073780-1024x1024.jpg` (**not flagged by the evaluation**) | `processed/train/kulfi/000068.jpg` | 7 | **Same iStock photo** (credit "subodhsathe"), cropped differently. Missed because the crop moved the dHash beyond 3 |

All 5 confirmed overlaps in Pexels-named files are Pexels photos that are
also in the training dataset. Any other Pexels photo may therefore have been
seen in a form the check cannot detect, such as a crop or re-edit.

The other candidates at distances 5 to 8 all had different labels and were
different photos on inspection, for example a burger vs a fried-rice image.

### 3.3 Classes with too few images

- **Before cleaning, 31 of 34 classes have fewer than 10 images.**
  - The exceptions are Hot Dog (10), idli (11) and sushi (15). For sushi,
    14 of the 15 are one photo.
- **After cleaning, 33 of 34 are below 10.** Only idli has 10.
  - Sushi drops to 2 and kulfi to 4.
  - Baked Potato, apple pie and chapati have 5 each.
- **Per-class precision, recall and F1 cannot support conclusions.** One
  image changes a class's recall by 10 to 25 points.

### 3.4 Clustering by photographer

- 191 distinct (class, photographer) groups cover the 240 cleaned images.
- **Several classes are dominated by one shoot:**
  - jalebi: 7 of 9 from one photographer
  - fried rice: 5 of 7
  - Taco, cheesecake and dal makhani: 4 each
- Photos from one shoot are correlated, so the effective sample size is
  smaller than the image count.

### 3.5 The large image

`kulfi/pexels-sahaab-38678811.jpg`:
- **Size:** 9534 × 12712 px (121.2 megapixels), an 8.25 MB baseline JPEG.
  - EXIF orientation is 1.
  - There is no camera make or model, and the DPI is 72.
- **It is a legitimate photograph, not a decompression bomb.**
  - The image is a street scene of a kulfi vendor's cart.
  - The compressed size (about 0.55 bits per pixel) is normal for a real
    photo.
  - A bomb is a tiny file that declares enormous dimensions.
- **It triggers the warning by design.** It exceeds Pillow's default
  `MAX_IMAGE_PIXELS` of 89,478,485 but is below twice that, which is the
  point where Pillow raises an error.
- **Its label is questionable for scope rather than for food class.** The
  kulfi is a small part of the frame. Production predicts Taco and v2
  predicts Taquito.

### 3.6 Provenance and licences

- **`data/real_world_test/provenance.csv` does not exist,** so 263 of 263
  images are undocumented.
- **File names suggest the sources:**
  - 261 are named `pexels-*`.
  - 2 are `istockphoto-*-1024x1024.jpg`. These are watermarked iStock
    previews with the "iStock / Credit: …" overlay on the food. They are
    not licensed copies, and the watermark is an artefact a real user photo
    would not have.
- **None of these are "own camera" photos,** which is what
  `evaluation/README.md` asks for. Treat the set as web stock photography,
  not as user photos.

### 3.7 Possible label problems (not decided)

I looked at the 14 images that both models get wrong, where at least one
model was at least 70% confident. Seven look questionable. **They are
flagged for review, and no label was changed:**

| File | Observation |
|---|---|
| `butter_naan/pexels-yarensphotoworks-30556657.jpg` | Flatbread on a tawa; could be chapati or roti. Both models say chapati |
| `chicken_curry/pexels-drmkhawarnazir-38003495.jpg` | Yellow cubes in curry; no visible chicken (paneer or potato?). Both models say kadai paneer |
| `chicken_curry/pexels-jkreat0r-32986463.jpg` | Brown curry in a bowl; contents unclear at this size |
| `dhokla/pexels-marcelochagas-3324431.jpg` | Pale cubes on a board; not clearly dhokla |
| `idli/pexels-mahesh-kumar-ganesan-2148232589-35514447 (1).jpg` | White halves in curry; could be egg curry |
| `kaathi_rolls/pexels-magda-ehlers-pexels-12758949.jpg` | Looks like spring rolls |
| `kulfi/pexels-sahaab-38678811.jpg` | Street scene (section 3.5) |

The other 249 images were not individually label-reviewed in this audit.

## 4. Clean evaluation (Task 3)

### 4.1 What was created (no source file changed)

- **`evaluation/clean_real_world.py`** reads the two existing reports and a
  review sheet. It writes only to `reports/real_world_clean/`, and it
  refuses to write into the original report folders or `data/`.
  - It re-uses the predictions already stored in the reports, so it is the
    same checkpoints and the same preprocessing, and nothing is re-run.
  - It stops if any image file is missing or has changed since it was
    evaluated.
  - It gives both models the identical cleaned manifest, and checks that
    with the same manifest-digest check used by `compare_real_world`.
- **`data/real_world_review/manual_review.csv`** is the review sheet: one
  row per flagged image, with columns `issue`, `related`,
  `audit_finding`, `default_action`, and then **`decision`, `reviewer`
  and `notes` left blank for a person to fill in**.
  - `decision` is `keep` or `exclude`. There is no relabel option: the
    label is always the class folder.
  - It is a separate folder outside `data/real_world_test/`.
- **Outputs:**
  - `clean_manifest.csv` (path, label, SHA-256)
  - `exclusions.csv` (path, label, reason, who decided)
  - `clean_evaluation.json` and `.md`
  - `confusion_matrix_old.csv` and `confusion_matrix_new.csv`

**Rules.** The label is never chosen by code.
1. **Same-label exact copies:** the copy without a ` (n)` suffix is kept and
   the others are excluded. This is automatic.
2. **One file in several class folders:** all copies are excluded and the
   result stays pending until a reviewer marks exactly one copy `keep`, or
   marks every copy `exclude` for an ambiguous image. If two copies are
   marked `keep`, or only some copies are decided, they stay excluded and
   pending.
3. **Overlaps reported by the evaluation:** excluded until a reviewer marks
   the image `keep`. This applies even to the masala dosa false alarm.
4. **Other review rows:** the reviewer's `decision` is used. While it is
   blank, the audit's `default_action` is used.
5. **Status:** the result is `provisional` while any flagged image lacks a
   decision.

### 4.2 Current result (provisional)

| | Production | v2 |
|---|---|---|
| Images | 240 | 240 |
| Correct | 127 | 157 |
| Accuracy | 52.9% | 65.4% |
| Macro F1 (34 classes) | 47.3% | 62.1% |

- **Paired test:** production right and v2 wrong 9; production wrong and v2
  right 39; exact McNemar p = 1.5e-5.
- **Bootstrap 95% CI (2,000 resamples):** accuracy +7.1 to +18.3 points;
  macro F1 +8.9 to +20.2 points.
- **Excluded: 23 images.**
  - 14 extra copies (13 sushi and 1 apple pie)
  - 2 for the conflicting-label file
  - 6 evaluation-flagged overlaps, including the likely false alarm
  - 1 iStock overlap missed by dHash

The scikit-learn recomputation from the original predictions matches these
numbers.

### 4.3 Images that need a person's decision

These are in `data/real_world_review/manual_review.csv`:

| Image(s) | Why | Audit suggestion |
|---|---|---|
| `idli/…35539315.jpg`, `masala_dosa/…35539315.jpg` | Same file, two labels; mixed meal | Exclude both, or keep the one copy you judge correct |
| Donut, both Hot Dog files, Sandwich, momos (section 3.2) | Same photo as train/validation images | Exclude |
| `kulfi/istockphoto-657073780-1024x1024.jpg` | Same photo as `train/kulfi/000068.jpg`; watermarked | Exclude |
| `masala_dosa/pexels-jack-baghel-2199968-20422126.jpg` | dHash false alarm | Keep |
| `kulfi/istockphoto-2154324654-1024x1024.jpg` | Watermarked stock preview; licence unknown | Your call. Replace with an own photo if possible |
| `kulfi/pexels-sahaab-38678811.jpg` | 121 MP street scene; kulfi barely visible | Your call. Exclude if the test is meant to show plates of food |
| The six label-uncertain images (section 3.7) | Label may be wrong | Check each one: keep it, or exclude it (and move it yourself if the label is wrong) |

### 4.4 Sensitivity

These figures come from a one-off check run during this audit; they are not
a saved report.

| Scenario | Images | Production | v2 | Paired (prod right & v2 wrong / prod wrong & v2 right) |
|---|---|---|---|---|
| Excluding the 8 flagged images kept by default | 232 | accuracy 54.3%, macro F1 47.6% | accuracy 67.2%, macro F1 62.7% | 9 / 39 |
| Keeping the masala dosa false alarm | 241 | accuracy 52.7%, macro F1 47.1% | accuracy 65.6%, macro F1 62.2% | 9 / 40 |

The pending decisions move the numbers by about a point. They do not change
the direction or the significance of the result.

### 4.5 For the next real-world set

- **Independence:**
  - Use own photos, as the README asks.
  - Do not use Pexels or iStock images; the training data already
    contains Pexels photos.
  - Never copy anything from the processed splits.
- **Coverage:** add at least 10 distinct photos per class from several
  sessions or photographers. Use 20 for the weak classes: apple pie, burger,
  butter naan, cheesecake and chicken curry.
- **Duplicates:** keep one file per photo. Delete accidental ` (n)` copies
  yourself; the tools never delete.
- **Provenance:** write `provenance.csv` (`file,source`, plus a licence).

### 4.6 Final result after manual review (2026-10-11)

The review sheet now has a decision and reviewer for every row:
- Both copies of the idli/masala dosa file are excluded, because the dish is
  ambiguous.
- The masala dosa false alarm is kept.
- The second iStock image, the 121 MP street photo and the six
  label-uncertain images are excluded.

Re-running `evaluation.clean_real_world` on the stored predictions gives
manifest status **reviewed**, with 233 images kept and 30 excluded.

| | Production | v2 |
|---|---|---|
| Correct | 126 | 157 |
| Accuracy | 54.1% | 67.4% |
| Macro F1 | 47.4% | 62.8% |

- **Paired test:** production right and v2 wrong 9; production wrong and v2
  right 40; p = 9.3e-6.
- **Bootstrap 95% CI:** accuracy +7.7 to +18.9 points; macro F1 +9.1 to
  +20.9 points.
- **All 34 classes now have fewer than 10 images,** so per-class figures
  remain unreliable.

## 5. Large-image handling (Task 4)

- **Pillow's protection is intact.**
  - No project code changes `Image.MAX_IMAGE_PIXELS`.
  - Above twice the limit (about 179 MP), Pillow raises
    `DecompressionBombError`. Every loader catches it and reports the image
    as invalid instead of crashing:
    - `evaluation/dataset_audit.py`
    - `evaluation/evaluate_real_world.py`
    - `evaluation/hashing.py`
    - `evaluation/intake.py`
    - the backend, which returns HTTP 400
- **Between the limit and twice the limit, Pillow only warns,** and the
  image is fully decoded.
  - For this photo that is about 364 MB as RGB.
  - Peak memory is roughly 2 to 3 times that during preprocessing, because
    `exif_transpose` and `convert("RGB")` each make a copy. This is an
    estimate from the dimensions, not a measurement.
  - Images are processed one at a time, so memory is bounded at roughly
    1.5 GB in the worst case. The evaluation finished, and the photo was
    scored by both models.
- **There is no confirmed bug, so no fix was made.**
  - A reduced-size JPEG decode (`Image.draft`) would lower memory use, but
    it changes pixel values relative to the API's preprocessing. That would
    break the "same preprocessing as production" guarantee the comparison
    relies on.
- **One observation outside this task's scope, not changed:** the backend's
  10 MB upload limit does not bound pixel count. A ≤10 MB JPEG of up to
  about 179 MP would be decoded in full, using about 1.5 GB of transient
  memory per request. If the API is ever exposed publicly, consider a pixel
  limit before decoding. This is a backend change and was not made.

## 6. Model decision (Task 5)

**Supported:**
- **v2 is better than production on this collection of web food photos.**
  It is ahead on accuracy (+12.5 points) and macro F1 (+14.7 points) on the
  original 263 images.
  - The gain persists after duplicates and overlaps are removed: +12.5
    points accuracy and +14.8 points macro F1 on 240 images.
  - The paired test is decisive (39 vs 9).
- **This agrees with the offline clean test set,** which v2 passed with
  91.9% accuracy against 81.6% for production.
- **Both models fall a long way from their offline numbers on these
  photos:** production about 53% vs 82% offline, and v2 about 65% vs 92%
  offline.

**Not proven:**
- **Real-user accuracy is not established.**
  - The photos are professional stock photography, not user phone photos.
  - Their sources are undocumented.
  - Pexels is evidently one of the training sources, so undetected overlap
    is possible.
- **No per-class claim holds.** All but one class has fewer than 10 images.
  Per-class "regressions" such as fried rice (−22.9 F1) and kaathi rolls
  (−16.7) are each 1 or 2 images.
- **Calibration isn't validated.** The 40% threshold suggested for v2 was not
  checked on these photos.
- **Nothing about foods outside the 34 classes.**

**Recommendation:**
- **Do not promote v2 yet.** Promotion still needs your decision.
- The evidence now consistently favours v2, both offline and on this set.
- **Before deciding:**
  1. Complete the manual review.
  2. Ideally, add a small set of your own phone photos with provenance, at
     least 10 per class.
  3. Re-run the three evaluation commands plus `clean_real_world`.

No retraining, checkpoint change, backend change or UI change was made.

## 7. Files

**Added:**
- `evaluation/clean_real_world.py`: builds the cleaned manifest and scores
  both models on it.
- `tests/test_real_world_clean.py`: 8 tests.
- `data/real_world_review/manual_review.csv`: the review sheet. It has 17
  rows, and its decision columns are blank.
- `reports/real_world_clean/` (generated): `clean_manifest.csv`,
  `exclusions.csv`, `clean_evaluation.json` and `.md`, and both confusion
  matrices.
- This document.

**Changed:**
- `evaluation/compare_real_world.py`: the body of `compare()` that runs
  after loading moved unchanged into `compare_loaded()`, so the cleaned
  evaluation can reuse it. The outputs are the same.
- `evaluation/README.md`: a short "Cleaned evaluation" section.

**Unchanged:**
- all 263 images
- `reports/real_world*`
- both checkpoints
- the processed train, validation and test data
- the feedback database
- the backend and frontend
