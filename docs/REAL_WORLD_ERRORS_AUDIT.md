# FoodCal AI — Audit of reported real-world errors

Date: 2026-10-11. This is an audit only. **No training was started**, and no
image, label, split, checkpoint or database row was changed.

## 1. Training and model status

- No training process is running. The only Python processes are the
  FastAPI backend (uvicorn on port 8000).
- **Production model:** `models/efficientnet_b0/food_efficientnet_b0_best.pth`,
  SHA-256 `39363ab4f70d68fb579904f6424fdd59f6c69197822eeba7d544a2b7f0084f41`.
  - It is unchanged and identical to its backup.
  - The backend serves this model, so every prediction in your history
    (47 of 47) came from it.
- **v2 candidate:** `models/efficientnet_b0_v2/food_efficientnet_b0_v2_best.pth`.
  - This is the epoch-21 checkpoint, accepted on 2026-10-11.
  - It is not deployed.
- No newer candidate exists.

## 2. Preprocessing: inference matches training

| Step | Training data (`prepare_dataset.py`) | v2 evaluation transform | API (`classifier.py`) |
|---|---|---|---|
| EXIF orientation, RGB | yes | — (already applied) | yes |
| Resize | LANCZOS to 224×224, no crop | Resize((224,224)), a no-op | LANCZOS to 224×224, then Resize((224,224)) |
| Normalisation | — | ImageNet mean/std | ImageNet mean/std |

One known difference:
- The training images were re-saved as JPEG after resizing. The API does
  not re-encode.
- `reports/reliability/` measured the effect: 97.1% top-1 agreement on
  1,000 originals.
- So preprocessing is not the cause of the reported errors.

## 3. Image counts per class

**Totals**

| Split | Images |
|---|---|
| Train, original | 16,699 |
| Train, v2 leakage-free manifest | 15,949 |
| Validation | 3,572 |
| Clean test | 3,304 |
| Real-world test (`data/real_world_test/`) | **0** |

**New this session**
- Intake batch `feedback_20261011_incorrect`: 6 images, all **pending
  review**, so none is used anywhere.
- The dataset counts are therefore unchanged.

The table is sorted by v2 train count. F1 is on the clean test set, from
`reports/model_comparison/comparison.md`.

| Class | Train (original) | Train (v2 manifest) | Validation | Clean test | Production F1 | v2 F1 |
|---|---|---|---|---|---|---|
| paani_puri | 100 | 96 | 21 | 21 | 72.7% | 100.0% |
| kulfi | 149 | 139 | 32 | 28 | 62.2% | 90.9% |
| samosa | 170 | 165 | 36 | 36 | 77.6% | 94.3% |
| dhokla | 171 | 168 | 36 | 37 | 88.6% | 92.1% |
| masala_dosa | 191 | 184 | 40 | 37 | 86.8% | 98.7% |
| pakode | 188 | 186 | 40 | 40 | 65.0% | 96.1% |
| kaathi_rolls | 195 | 187 | 41 | 41 | 70.5% | 87.8% |
| pizza | 192 | 188 | 41 | 42 | 83.5% | 96.4% |
| dal_makhani | 206 | 197 | 44 | 40 | 80.0% | 96.3% |
| jalebi | 201 | 197 | 43 | 40 | 92.3% | 94.9% |
| pav_bhaji | 213 | 202 | 45 | 43 | 89.4% | 95.3% |
| butter_naan | 214 | 208 | 46 | 46 | 82.9% | 89.7% |
| idli | 211 | 209 | 45 | 45 | 97.8% | 98.9% |
| momos | 226 | 221 | 48 | 47 | 90.7% | 94.9% |
| chapati | 228 | 223 | 49 | 48 | 73.3% | 87.8% |
| burger | 231 | 227 | 49 | 51 | 91.7% | 96.1% |
| kadai_paneer | 233 | 231 | 50 | 48 | 76.9% | 93.8% |
| chai | 240 | 236 | 51 | 50 | 92.2% | 97.0% |
| fried_rice | 244 | 242 | 52 | 54 | 84.5% | 97.2% |
| chole_bhature | 263 | 256 | 56 | 55 | 88.5% | 98.2% |
| apple_pie | 700 | 697 | 150 | 149 | 73.6% | 89.6% |
| ice_cream | 700 | 698 | 150 | 150 | 84.2% | 89.7% |
| omelette | 700 | 698 | 150 | 150 | 81.5% | 92.3% |
| cheesecake | 700 | 699 | 150 | 150 | 80.8% | 89.7% |
| chicken_curry | 700 | 699 | 150 | 150 | 80.4% | 91.3% |
| sushi | 700 | 700 | 150 | 150 | 83.5% | 92.1% |
| Fries | 1050 | 879 | 225 | 170 | 87.7% | 91.7% |
| Taquito | 1050 | 942 | 225 | 191 | 73.8% | 88.9% |
| Baked Potato | 1050 | 943 | 225 | 182 | 84.1% | 93.4% |
| Hot Dog | 1083 | 973 | 232 | 181 | 81.9% | 91.1% |
| Donut | 1050 | 1004 | 225 | 205 | 92.1% | 95.5% |
| Sandwich | 1050 | 1012 | 225 | 210 | 82.7% | 91.3% |
| Crispy Chicken | 1050 | 1015 | 225 | 203 | 80.2% | 90.0% |
| Taco | 1050 | 1028 | 225 | 214 | 71.3% | 85.7% |

## 4. Reported errors, checked against the saved history

| Reported | In the history database | Label check (my reading of the thumbnail; not a verification) |
|---|---|---|
| Apple Pie → Chapati | feedback saved: chapati 41.6% → apple_pie | Whole lattice apple pie, top-down. Label looks right. |
| Burger → Hot Dog | Hot Dog 27.7%, **no feedback saved** | Burger **and fries** on a tray: two foods in one photo |
| Butter Naan → Fried Rice | feedback saved: fried_rice 72.3% → butter_naan | Naan with herbs in a basket. Label looks right. |
| Cheesecake → Pizza | feedback saved: pizza 27.0% → cheesecake | Whole **Basque burnt** cheesecake on a stand. Label looks right; the style is rare in the training data. |
| Chicken Curry → Kadai Paneer | kadai_paneer 79.4% (2 uploads), **no feedback saved** | Home-style chicken curry with potatoes and peas. Looks like chicken_curry. |

Other corrections in the history:
- dal_makhani → **Sandwich**: a club sandwich; label looks right.
- chapati → **momos**: momos with chutney; label looks right.
- jalebi → **Baked Potato**: the photo shows a baked **sweet potato**.
  Whether that belongs in `Baked Potato` is a taxonomy decision for you.

Most of these thumbnails look like professional or stock photographs, not
phone photos. Their licence must be confirmed before any is used for
training. The export marks them "uploaded by the app user (local, not
redistributable)" and the `license` field should be corrected during
review.

### Production vs v2 on the same stored thumbnails

Covers 17 distinct images:
- 15 with saved feedback
- 2 whose labels you stated in your request (burger, chicken curry)

**This is anecdotal, not a metric.** n = 17, the images are 256 px
thumbnails rather than the uploads, and production's thumbnail predictions
can differ from its original predictions. For example, your chapati upload
was right at full size but becomes momos as a thumbnail.

| Image | Label | Production | v2 |
|---|---|---|---|
| club sandwich | Sandwich | Hot Dog 35% ✗ | Sandwich 82% ✓ |
| momos | momos | Donut 70% ✗ | momos 79% ✓ |
| naan | butter_naan | kulfi 44% ✗ | butter_naan 51% ✓ |
| chapati | chapati | momos 29% ✗ | chapati 65% ✓ |
| apple pie | apple_pie | chapati 44% ✗ | Sandwich 51% ✗ |
| Basque cheesecake | cheesecake | Sandwich 33% ✗ | Sandwich 21% ✗ |
| baked sweet potato | Baked Potato | pakode 16% ✗ | butter_naan 21% ✗ |
| burger + fries | burger (stated) | Fries 42% ✗ | Fries 28% ✗ |
| chicken curry | chicken_curry (stated) | kadai_paneer 36% ✗ | Crispy Chicken 59% ✗ |
| 8 others | kulfi, Taco, chai, dhokla, Donut, fried_rice, masala_dosa, Hot Dog | all ✓ | all ✓ |

**Totals:** production 8/17 right, v2 12/17 right. v2 fixed 4 and broke 0.
On a paired exact test that gives p = 0.125, which is **not
statistically significant**.

### The same confusions on the clean test set

| Actual → predicted | Test images | Production | v2 |
|---|---|---|---|
| apple_pie → chapati | 149 | 4 | 0 |
| burger → Hot Dog | 51 | 0 | 0 |
| butter_naan → fried_rice | 46 | 0 | 0 |
| cheesecake → pizza | 150 | 0 | 0 |
| chicken_curry → kadai_paneer | 150 | 8 | 3 |

- These pairs are rare or absent on the clean test set.
- Your errors therefore come from the **gap between your photos and the
  dataset**, which the clean test set cannot measure.
- Only an independent real-world test set can show whether a change fixes
  them.

## 5. Do the training images represent real photos?

I looked at 24 random training images per class.

| Class | Training images mostly show | Missing or rare (seen in your photos) |
|---|---|---|
| apple_pie | slices on plates, often with ice cream; dim restaurant light (Food-101) | whole pies, lattice tops, top-down, bright home settings |
| cheesecake | New York–style slices with fruit or chocolate topping (Food-101) | whole cakes, Basque burnt cheesecake, cake stands |
| burger | studio and advertising shots, logos, text overlays (web-scraped) | ordinary café or home photos, burger with fries on a tray, side view |
| chicken_curry | very mixed, including Thai red and green curries (Food-101) | Indian home-style curry with potatoes, bone-in pieces, bowl top-down |

So the class definitions in the data do not match how you photograph the
food. More of the same kind of image will not fix this; the new images
must cover the missing presentations.

## 6. Data needed (proposal)

All new images go through `evaluation.intake` with a human review decision.
New photos are **in addition to** a separate real-world test set; the two
sets must never mix.

Sources:
- your own photos (best: they match the app's real use)
- or images with a recorded licence that allows this use, such as
  Wikimedia Commons or Open Images with CC licences
- not search-engine scrapes

| Priority | Class | Train now (v2) | Why | New verified train images | Real-world test photos |
|---|---|---|---|---|---|
| 1 | apple_pie | 697 | reported error; whole and lattice pies missing | 150 | 20 |
| 1 | cheesecake | 699 | reported error; whole and Basque cakes missing | 150 | 20 |
| 1 | burger | 227 | reported error; few ordinary photos; small class | 200 | 20 |
| 1 | butter_naan | 208 | reported error; v2 F1 89.7%; confused with chapati | 150 | 20 |
| 1 | chicken_curry | 699 | reported error; class is very mixed | 150 | 20 |
| 2 | Taco | 1,028 | lowest v2 F1 (85.7%), confused with Taquito | 100 | 20 |
| 2 | kaathi_rolls, chapati | 187, 223 | v2 F1 87.8% each; small classes | 150 each | 20 each |
| 2 | Taquito, ice_cream | 942, 698 | v2 F1 88.9% and 89.7% | 100 each | 20 each |
| — | Baked Potato | 943 | needs a decision: does sweet potato belong? | 0 until decided | 10 |
| 3 | all other classes | — | only to measure real-world performance | 0 | ≥ 10 each |

**How the targets were chosen**

These are judgement calls, not measured requirements.
- Small classes (about 200 images) get +150–200, roughly doubling them.
- Large classes (about 700 or more) get +100–150 aimed only at the missing
  presentations, about 15–20%.
- The 6 feedback images are far too few on their own. Adding them without
  oversampling will not measurably change a model trained on 15,949 images.
  Oversampling them risks overfitting to six photos.

## 7. Why there is no new training run yet

The dataset and label audit is **not complete**. Training now would use
the same data as v2.
1. **Label review:** 204 rows in `reports/phase1_audit/label_review.csv`
   still have no decision. Known wrong labels (e.g. pizza in Sandwich and
   Donut, burgers in Sandwich) are still in training.
2. **Feedback:** the 6 images in `data/intake/feedback_20261011_incorrect/`
   are unreviewed and unlicensed for training.
3. **New photos:** no new verified, diverse photos exist for the priority
   classes.
4. **Real-world test set:** `data/real_world_test/` is empty. The reported
   errors cannot be measured, so any "improvement" on them could not be
   shown.

## 8. Recommendation on the production model

**v2 passes the offline promotion gate** on the 3,304-image clean test
set:
- accuracy 81.6% → 91.9%
- macro-F1 81.9% → 93.2%, 95% CI of the difference +10.0 to +12.8 pts
- McNemar p = 1.1e-58
- no class with 20 or more test images got worse by more than 5 F1 points

On your 17 stored photos it was right 12 times against 8 and broke none.
That count is too small to be significant.

The independent real-world evaluation required before switching does not
exist yet. The fastest way to a verified decision:
1. Take about 10 photos per class, 20 for the priority classes, into
   `data/real_world_test/` with `provenance.csv`.
2. Evaluate both checkpoints on those photos, then compare them.
   (Correction: an earlier version of this document wrongly said the
   evaluator had no `--checkpoint` option.)
   ```
   .venv-gpu\Scripts\python.exe -m evaluation.evaluate_real_world
   .venv-gpu\Scripts\python.exe -m evaluation.evaluate_real_world --checkpoint models/efficientnet_b0_v2/food_efficientnet_b0_v2_best.pth --output-dir reports/real_world_v2
   .venv-gpu\Scripts\python.exe -m evaluation.compare_real_world
   ```
3. Switch to v2 if it holds up. Set `FOODCAL_LOW_CONFIDENCE_THRESHOLD=40`
   with it, as calibrated on validation for v2.
