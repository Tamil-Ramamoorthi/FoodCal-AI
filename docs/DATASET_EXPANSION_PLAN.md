# FoodCal AI — Training status and dataset expansion plan

Date: 2026-10-11. Status: **proposal; partly approved** (see section 8).
Nothing in this plan has been trained, relabelled, moved or deleted.

## 1. Phase 1 — status of the v2 training run

**The v2 run did not finish normally. It was interrupted.**

- `models/efficientnet_b0_v2/train_log.txt` stops after epoch 22. There is
  no final `Best: ...` line, which the script prints when it completes.
- The run could not have early-stopped at epoch 22. The best epoch was 21
  and patience is 5, so stopping was impossible before epoch 26. The run
  allowed up to 28 epochs (3 head + 25 full).
- Windows went to sleep at 2026-10-10 22:01:50 (Kernel-Power event 42),
  about a minute after epoch 22 was logged at 22:00:51. No Python process
  is running now.
- The best checkpoint (epoch 21, validation macro-F1 93.08%) was written
  completely at 21:59:32, before the interruption. It loads in the API's
  classifier code.
- Unknown: how much the 6 epochs that never ran would have changed the
  result. The learning rate was already decaying (cosine schedule), and
  validation macro-F1 had changed by less than 1 point over the last 5
  epochs.
- Following the rules, the run has **not** been resumed or restarted. The
  epoch-21 checkpoint is evaluated below as the "v2 candidate".
- **Decision (2026-10-11): the epoch-21 checkpoint is accepted as v2.** No
  re-run.

**Checkpoints (SHA-256)**

| Checkpoint | SHA-256 | State |
|---|---|---|
| Production `models/efficientnet_b0/food_efficientnet_b0_best.pth` | `39363ab4…0084f41` | Unchanged; identical to `backups/production_checkpoint_20261010_211552/` |
| v2 candidate `models/efficientnet_b0_v2/food_efficientnet_b0_v2_best.pth` | `e34b83ee…879c0ebf7` | Epoch 21 of an interrupted run |

The backend still loads the production checkpoint.

### Evaluation on the same clean test set

`data/manifests/v2/test_clean.csv` has 3,304 test images with no exact or
near copy in train or validation. Both models go through the API's own
preprocessing. Full report: `reports/model_comparison/comparison.md`
(plus `.json` and both confusion matrices as CSV).

| Metric | Production (epoch 4) | v2 candidate (epoch 21) | Change |
|---|---|---|---|
| Accuracy | 81.6% | 91.9% | +10.3 pts |
| Macro precision | 81.1% | 93.5% | +12.3 pts |
| Macro recall | 84.3% | 93.0% | +8.7 pts |
| Macro F1 | 81.9% | 93.2% | +11.3 pts |
| Weighted F1 | 81.8% | 91.9% | +10.1 pts |

**Is the difference real?**
- Paired McNemar test: 411 images only v2 gets right vs 72 only production
  gets right, p = 1.1e-58.
- Bootstrap 95% CI for the accuracy gain: +9.1 to +11.4 pts.
- Kulfi precision: 50.0% → 92.6%.
- Images wrongly predicted as kulfi: 23 → 2.

**Robustness and calibration**

Source: `reports/reliability_v2/`, same diagnostics as
`reports/reliability/`. The image shifts are synthetic, not real photos.

| Shift | Production | v2 candidate |
|---|---|---|
| Original | 82.6% | 92.2% |
| Zoom out onto background | 51.8% | 78.5% |
| Blur | 52.6% | 85.4% |
| Rotated 90° | 72.0% | 85.0% |
| JPEG quality 30 | 72.3% | 79.8% |

- v2's confidence is **under**-confident: expected calibration error is
  about 10%, from label smoothing.
- So the API's 50% low-confidence threshold does not transfer. For v2, the
  same rule applied on validation gives **40%**.
- If v2 is ever deployed, the threshold must change with it.

### What this evaluation does *not* show

**The real-world test set is still empty**
- All 34 folders in `data/real_world_test/` contain 0 images.
- The clean test set comes from the same web source as the training data.
- So real-world accuracy is **unknown** for both models.

**The reported idli photo is still misclassified by v2**
- Upload `bb5b6922…`: idli domes with dried chilli and coriander.

| Model | Top prediction | Runner-up | idli in top 3? |
|---|---|---|---|
| Production | kulfi 84.9% | — | — |
| v2 | ice_cream 29.5% | kulfi 20.9% | No |

- v2 is still wrong, but no longer confidently wrong. Below 40% it would be
  flagged as low confidence.
- This is one image, not a metric.

**Recommendation:** do not switch production to v2 yet. Collect the
real-world test set first (section 4), evaluate both models on it, then
decide. v2 is the baseline every future model must beat.

## 2. Phase 2 — class audit (v2 candidate, clean test set)

Classes with F1 below 91%, or a known real-world failure. Classes with
fewer than 50 test images have wide uncertainty: one image is 2–5 points.

Confusions are image counts from `confusion_matrix_new.csv`. "Missed as"
means images of this class predicted as another. "Taken from" means other
classes predicted as this one.

| Class | Train | Test | P / R / F1 | Missed as | Taken from |
|---|---|---|---|---|---|
| Taco | 1,028 | 214 | 83.3 / 88.3 / 85.7 | Taquito 8, Hot Dog 5, Crispy Chicken 4 | Taquito 11, Crispy Chicken 7, Hot Dog 6 |
| kaathi_rolls | 187 | 41 | 87.8 / 87.8 / 87.8 | Taquito 2, samosa 1, dhokla 1 | samosa 2, burger 1, Taquito 1 |
| chapati | 223 | 48 | 86.0 / 89.6 / 87.8 | butter_naan 2 | butter_naan 6 |
| Taquito | 942 | 191 | 89.4 / 88.5 / 88.9 | Taco 11, Sandwich 3 | Taco 8, Sandwich 3, Fries 3 |
| apple_pie | 697 | 149 | 92.8 / 86.6 / 89.6 | cheesecake 7, ice_cream 6 | omelette 4 |
| cheesecake | 699 | 150 | 86.9 / 92.7 / 89.7 | ice_cream 7 | ice_cream 7, apple_pie 7, sushi 4 |
| ice_cream | 698 | 150 | 86.9 / 92.7 / 89.7 | cheesecake 7 | cheesecake 7, apple_pie 6, sushi 5 |
| butter_naan | 208 | 46 | 95.1 / 84.8 / 89.7 | chapati 6 | chapati 2 |
| Crispy Chicken | 1,015 | 203 | 89.3 / 90.6 / 90.0 | Taco 7, Baked Potato 4 | Fries 5, Taco 4 |
| kulfi | 139 | 28 | 92.6 / 89.3 / 90.9 | Donut 2, momos 1 | Taquito 1, Fries 1 |
| idli | 209 | 45 | 100 / 97.8 / 98.9 | chicken_curry 1 | — (but fails on the garnished close-up upload) |
| kadai_paneer | 231 | 48 | 93.8 / 93.8 / 93.8 | chicken_curry 2 | chicken_curry 3 |

**Confusion clusters**

These groups of visually similar foods should be collected together:
- **A. Pale and cream foods:** idli, kulfi, ice_cream, cheesecake,
  dhokla, apple_pie. On the clean test set the desserts confuse each
  other. The real-world idli upload falls into the same group.
- **B. Wrapped and fried items:** Taco, Taquito, kaathi_rolls, Hot Dog,
  Crispy Chicken.
- **C. Flatbreads:** chapati, butter_naan, chole_bhature.
- **D. Gravies:** kadai_paneer, chicken_curry, dal_makhani.

**Still pending**
- None of the 204 rows in `reports/phase1_audit/label_review.csv` has a
  decision yet.
- Known wrong labels (e.g. pizza in Sandwich and Donut, burgers in
  Sandwich, mixed Taquito images) are therefore still in training.

## 3. Phase 2 — data requirements

### Presentation coverage

Every priority class needs variety on each axis. Each image is tagged with
one value per axis at intake, so gaps show up in the audit.

| Axis | Values to cover |
|---|---|
| Serving form | single piece, several pieces, with sides (sambar, chutney), in a thali, partly eaten |
| Shape and variant | standard, mini, rava/stuffed variant, cut or sliced |
| Garnish | none, herbs (coriander), chilli, nuts (pistachio), saffron, sauce drizzle |
| Plate | white plate, steel plate or thali, banana leaf, dark plate, bowl, takeaway box or paper |
| Background | table, cloth, kitchen counter, restaurant, cluttered |
| Lighting | daylight, warm indoor, dim, flash, strong shadow |
| Camera angle | top-down, 45°, side-on |
| Distance | close-up (fills the frame), whole plate, plate on table |
| Capture | phone camera; no screenshots, collages, stock watermarks or text overlays in the test set |

The reported idli failure is a close-up of garnished pieces at 45°. That
combination is rare in the current data, so it is the first gap to fill.

### Target counts for new, verified images

Counts are *additional* images.

| Priority | Classes | New train | New real-world test | Why |
|---|---|---|---|---|
| 1 | idli, kulfi | 150 each | 30 each | Reported failure. kulfi has only 139 train images. |
| 1 | kadai_paneer | 100 | 30 | Requested; gravy cluster. |
| 1 | ice_cream, cheesecake, dhokla | 50 each | 20 each | Look-alikes of idli and kulfi (cluster A) |
| 2 | Taco, Taquito, kaathi_rolls | 100 each | 20 each | Lowest F1 (cluster B); Taquito labels also need cleaning |
| 2 | chapati, butter_naan | 100 each | 20 each | Cluster C |
| 2 | paani_puri, samosa, pakode | 100 each | 20 each | Smallest classes (96–186 train) |
| 3 | all other classes | 0 | ≥ 10 each | Needed only to measure real-world performance |

- The real-world test column is the most urgent: about 590 photos.
- Without it, no model can be judged on real photos.
- Collect it **before** any new training images, and from different
  sources or sessions.

## 4. Phase 2 — intake workflow (built 2026-10-11)

Implemented in `evaluation/intake.py` (`init`, `sync`, `validate`), with
tests in `tests/test_intake.py`. Usage and the full list of errors and
warnings: `evaluation/README.md`, section "Adding training images". The
manifest builder (`data/manifests/v3/`) is not built yet; it belongs to the
retraining step.

```
data/intake/<batch_id>/                 # e.g. 2026-10-15_phone_idli
    images/<class>/<file>.jpg           # originals, full resolution, never edited
    intake.csv                          # one row per image
reports/intake/<batch_id>/validation.md # written by the validator
data/manifests/v3/                      # train / validation / test manifests built from verified rows only
```

**`intake.csv` columns**
- `file`, `proposed_label`
- `source`, `license`, `captured_by`, `capture_date`
- `session_id`: same dish or same sitting = same session
- one column per coverage axis: `serving`, `variant`, `garnish`, `plate`,
  `background`, `lighting`, `angle`, `distance`
- `reviewer`, `decision` (`accept`, `relabel:<class>`, `reject`), `notes`

**Validator (read-only, reports only)**
1. **Readability:** decodes the image, checks format (jpg, png, webp), and
   reads EXIF orientation.
2. **Quality:** rejects images under 448 px on the short side, so they can
   be crop-augmented. Flags blurry (variance of Laplacian), very dark or
   very bright images as warnings. These heuristics are uncalibrated; on
   600 processed test images, 2.7% were flagged.
3. **Labels:** `proposed_label` must be one of the checkpoint's 34 classes
   (the taxonomy in section 5 later) and match its folder. The coverage
   tags must use the allowed values.
4. **Duplicates:** exact (SHA-256) and near (dHash ≤ 3) against all
   processed splits, `real_world_test/`, every earlier intake batch, and
   the rest of this batch. Reuses `evaluation/hashing.py`. An exact copy
   of anything is an error. A near copy of a validation, test or
   real-world test image is also an error (leak protection). Other near
   copies are warnings.
5. **Provenance:** source and licence must be filled in.

**Human verification is required**
- A model prediction may be shown as a hint for spotting mistakes. It
  never sets or changes a label.
- Only rows with `decision = accept` or `relabel:<class>` enter a
  manifest.
- This follows the Phase 1 finding that many confident model
  disagreements were the model's error, not the label's.

**Keeping splits independent**
- Split assignment is by `session_id`, not by image. Photos of the same
  plate never land in different splits.
- The real-world test set is a separate collection and is never used for
  training or threshold tuning.
- New verified images go to train and validation only. The existing clean
  test set stays frozen, so v1, v2 and v3 stay comparable.
- Original files are never moved, renamed or deleted. Manifests point at
  them.

## 5. Phase 3 — adding food classes (investigation)

**What the code supports today**
- The backend reads `class_names` from the checkpoint and sizes the output
  layer from it (`backend/app/models/classifier.py:41-44`).
- No class list is hard-coded in the backend or frontend.
- Nutrition is not connected yet.
- So a checkpoint with more classes loads without code changes.

**What could break**
- Stored history: earlier predictions keep their old labels.
- The evaluation scripts check that both models in a comparison have
  identical class lists. A 34-class model and a 40-class model need a
  comparison restricted to the 34 shared classes, plus separate metrics
  for the new ones.

**Proposed rules**
1. A versioned `data/taxonomy/classes_v3.json` lists each class with:
   - `id` and `display_name`
   - `aliases` (e.g. idli = idly)
   - `includes` / `excludes` (e.g. kadai_paneer *excludes* shahi paneer)
   - `confusable_with`
   - date added
2. Existing class indices never change. New classes are appended at the
   end.
3. Minimum per new class before it can be trained:
   - ≥ 200 verified train images
   - ≥ 40 validation images
   - ≥ 40 independent test images
   - the coverage axes above
4. Candidates must come from your actual usage. The Phase 1 audit names
   grilled-chicken salad and egg curry as examples of foods that would be
   forced into one of the 34. Your choice of candidates is needed.
5. The output layer and training pipeline change only after you approve
   the taxonomy.

## 6. Phase 4 — reliability and unknown foods (investigation)

**Metrics on every evaluation**
- accuracy, macro-F1, weighted F1
- per-class precision, recall and F1, with support
- confusion matrix
- calibration (ECE)
- all reported for both the clean test set and the real-world set

**Uncertain predictions**
- The API already returns a `low_confidence` flag.
- Proposed: the UI shows the top 3 as "possible matches" whenever that flag
  is set.
- The threshold is recalibrated on validation for each checkpoint (v2:
  40%).

**Unknown-food detection**
- Validated on a held-out set of real foods *outside* the 34 classes,
  collected separately (proposed: ≥ 200 images, ≥ 20 dishes).
- Candidate methods, compared by AUROC and by the share of in-class images
  wrongly rejected:

| Method | Note |
|---|---|
| Maximum softmax probability | current baseline |
| Energy score | from the logits; no retraining needed |
| Embedding distance | distance to class centroids in the penultimate layer; no retraining needed |
| Explicit "other food" class | needs real, labelled training data |

- Nothing is deployed until it is measured on that set.

## 7. Phase 5 — gate for retraining and integration

Retraining (v3) starts only after:
- this plan is approved
- the label review is done
- the real-world test set exists
- the priority intake batches are verified

v3 is saved to `models/efficientnet_b0_v3/`. Production and v2 checkpoints
stay untouched.

v3 replaces production only if, compared with production and v2 on
**both** the clean test set and the real-world set:
1. macro-F1 is higher, with a paired McNemar test and bootstrap CI.
2. no class with ≥ 20 test images drops by more than 5 F1 points.
3. the low-confidence threshold has been recalibrated for it.

Before any training run:
- Disable system sleep (`powercfg /change standby-timeout-ac 0`), or
  stay plugged in with sleep off. The v2 run was lost to sleep.
- Add resume-from-checkpoint support, so an interrupted run can be
  continued instead of repeated.

## 8. Decisions needed from you

1. ~~**v2 status**~~: **decided 2026-10-11:** the epoch-21 checkpoint is
   accepted as v2. No re-run.
2. **Real-world test photos:** start collecting your own phone photos
   (section 3, "New real-world test" column) into `data/real_world_test/`,
   with `provenance.csv`.
3. **Label review:** fill in `decision` in
   `reports/phase1_audit/label_review.csv`.
4. ~~**Intake validator**~~: **approved and built 2026-10-11** (section 4).
5. **New classes:** which dishes you want added, if any.
