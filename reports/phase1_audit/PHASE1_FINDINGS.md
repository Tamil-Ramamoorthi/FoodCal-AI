# FoodCal AI — Phase 1 audit findings and proposed corrections

Date: 2026-10-10. Read-only audit: no image, label, split or checkpoint was
changed.

Detailed numbers are in `class_audit.md` / `class_audit.json`. The
images behind every finding are in `review/`. Reproduce with:

```
.venv-gpu\Scripts\python.exe -m evaluation.class_audit
.venv-gpu\Scripts\python.exe -m evaluation.diagnose_reliability
```

## 1. Dataset structure

**Classes and splits**
- All 34 class folders are present in train, validation and test, and match
  the checkpoint's class list and index order.
- Split sizes are 16,699 / 3,572 / 3,602 images (70 / 15 / 15% per class),
  made by `training/prepare_dataset.py` (seed 42).

**Provenance**
- Replaying that script recovers the source image of every processed file.
  23,871 of 23,873 were verified by dHash.
- The source ZIP is no longer in the Hugging Face cache. The replay used
  the hash cache written earlier.

**Three groups of classes, by source and size**

| Group | Classes | Train images per class | Notes |
|---|---|---|---|
| Web-scraped fast food | Baked Potato, Crispy Chicken, Donut, Fries, Hot Dog, Sandwich, Taco, Taquito | 1,050–1,083 | Many white-background product shots, logos, cartoons and adverts |
| Food-101-like | apple_pie, cheesecake, chicken_curry, ice_cream, omelette, sushi | 700 | — |
| Indian dishes | the remaining 20 | 100–263 | — |

- Train imbalance is 10.8 : 1 (Hot Dog 1,083 vs paani_puri 100).

## 2. Duplicates and leakage

**Exact duplicates (SHA-256)**
- 480 extra copies in total.
- Concentrated in Fries (138), Baked Potato (79), Taquito (78) and Hot Dog (37).
- 114 test images are byte-identical to a train or validation image.

**Exact or near duplicates (dHash ≤ 3)**
- 298 test images and 243 validation images have a copy in train.
- On white-background product photos, dHash links unrelated images (see
  `review/cross_label_groups.jpg`). So near-duplicate counts for the
  fast-food classes are overstated.

**Effect on the measured score**

| Test set | Images | Accuracy | Macro F1 |
|---|---|---|---|
| Full test split | 3,602 | 82.6% | 82.7% |
| Leakage-free subset (no exact or near copy in train/val) | 3,304 | 81.6% | 81.9% |

**Independence**
- Both test sets come from the same web-image source as the training
  data. Neither is an independent real-world test set.
- `data/real_world_test/` is still empty.

## 3. Label problems

These were confirmed by viewing the images.

**Wrong labels**

| Image | Labelled as | Actually shows |
|---|---|---|
| `train/pizza/000040` = `test/pav_bhaji/000031` | pav_bhaji | pizza |
| `train/Donut/000102`, `train/Sandwich/000095`, `train/Sandwich/000662` | Donut, Sandwich | pizza |
| `test/Sandwich/000018`, `train/Sandwich/000402`, `train/Sandwich/000413`, `validation/Hot Dog/000217` | Sandwich, Hot Dog | burger |
| `train/Taco/000564`, `train/Taco/000253` | Taco | fries |
| Several Taquito images (`review/suspects_Taquito.jpg`) | Taquito | tacos, burritos, a pizza advert, a pastry hot dog |

**Wrong variants and non-photos**
- Near-variant dishes: e.g. `train/kadai_paneer/000073` is "Shahi Paneer".
- Logos, cartoons and adverts appear in the fast-food classes, e.g. a
  "Fully Loaded Baked Potatoes" logo, a cartoon taco, a "Big Stuff
  Burrito $5.99" advert.

**Not label errors**
- Many images the model contradicts with ≥90% confidence are correctly
  labelled and the model is simply wrong. Examples: real tacos predicted
  as Hot Dog, cheesecakes predicted as ice cream or apple pie.
- So model disagreement must never be used to relabel automatically.

**Review queue**
- `label_review.csv` lists 204 images for manual review:
  - 90 with high-confidence disagreement
  - all members of the 30 mixed-label duplicate groups
- The `decision` column is empty for you to fill in.

## 4. Why idli photos become kulfi

**The idli class is clean and well learned**
- 211 training images, correctly labelled. The rava idli variants
  included are legitimate.
- Leakage-free test recall and precision are both 97.8%. Not one
  validation or test idli is predicted as kulfi.

**The problem is presentation (domain shift)**
- Dataset idli images are mostly flat white discs, usually several, often
  with sambar or chutney, and frequently web recipe photos with
  watermarks or text.
- The user's photo is a close-up of cream-coloured domes with green
  (coriander) and red (chilli) garnish. That is the visual pattern of the
  kulfi class: cream or pale-yellow pieces with pistachio and saffron
  flecks.

**Kulfi attracts other pale foods**
- 149 training images, the second-fewest.
- Leakage-free test precision is 50.0% against 82.1% recall.
- `review/other_classes_predicted_as_kulfi.jpg` shows pale cream and yellow
  foods from many classes (cheesecake, ice cream, crepes, taquitos,
  fries, potato fillings) pulled into kulfi.

## 5. Training problems (actionable without new data)

**1. The deployed model was never successfully fine-tuned**
- The checkpoint is epoch 4, the last epoch of head-only training on a
  frozen ImageNet backbone.
- When fine-tuning starts, `run_epoch` puts every BatchNorm layer
  (including frozen blocks) into training mode with batches of 16.
- **Evidence:** with identical weights, only letting BatchNorm statistics
  update on 300 training batches drops validation accuracy from 83.1% to
  47.5%.
- Fine-tuning accuracy duly collapsed (train 91.0% → 74.4% at epoch 5).

**2. Early stopping ended fine-tuning prematurely**
- The patience counter was not reset between stages.
- Training stopped after 4 fine-tuning epochs, while validation was still
  rising (77.4 → 79.9 → 82.0 → 82.9%).
- Only the last two feature blocks were ever unfrozen.

**3. Inverse-frequency sampling over-predicts small classes**
- `WeightedRandomSampler` with weights 1/n shows each kulfi or paani_puri
  image about 7–11× per epoch more often than a fast-food image.

| Classes | Mean recall | Mean precision |
|---|---|---|
| 20 classes with < 300 training images | 88.0% | 78.9% |
| 14 classes with ≥ 700 training images | 79.2% | 84.3% |

- The correlation between log training count and (recall − precision) is −0.49.
- The worst cases are paani_puri, kaathi_rolls, kulfi, chapati and
  dal_makhani, whose recall exceeds their precision by 26–36 points.

**4. Augmentation lacks scale and framing variation**
- There is no random crop or zoom.
- Synthetic test shifts:
  - zoom-out: accuracy 51.8%
  - blur: 52.6%
  - rotation: 72.0%
  - JPEG quality 30: 72.3%
- All training images are stored at 224 px, which limits crop
  augmentation. Re-exporting at higher resolution needs the source ZIP to
  be downloaded again.

## 6. Proposed corrections, in order

**A. Training fixes. No new data needed; first experiment.**
1. Keep BatchNorm layers of frozen blocks in eval mode.
2. Reset patience per stage.
3. Fine-tune the whole backbone with a lower learning rate for early
   blocks, a warm-up and a cosine schedule, and enough epochs.
4. Select the checkpoint on validation macro-F1.
5. Replace 1/n sampling with milder balancing: weights ∝ 1/√n, or
   unweighted shuffling. Choose on validation.
6. Add realistic augmentation: random resized crop (scale about 0.6–1),
   flips, rotation, colour and brightness jitter, mild blur.
7. Keep the 34-class schema, preprocessing and checkpoint format, so the
   FastAPI backend works unchanged.
8. Save to a new file, e.g.
   `models/efficientnet_b0_v2/food_efficientnet_b0_best.pth`. The
   production checkpoint is backed up and not replaced.

**B. Leakage-free data handling. Implemented as manifests, without deleting or moving files.**
1. Exclude from the new training run every training image that has an
   exact or near copy in validation or test.
2. Keep one copy of exact duplicates within train.
3. Freeze a leakage-free test manifest (`leakage_free_test_paths` in
   `class_audit.json`, 3,304 images). Evaluate both the old and new models
   on it, and also report the full test split.
4. Never copy validation or test images into training.

**C. Label review. Only with your decisions.**
1. Fill in `decision` in `label_review.csv`: `keep`, `relabel:<class>`, or
   `exclude` (for non-food images, logos, adverts, ambiguous multi-dish
   photos).
2. Training applies only verified decisions, through a manifest. Original
   files stay where they are.

**D. Additional data.** A documented intake process:
- New images go into a separate folder with a provenance record.
- Validation checks format, corruption, class name, and exact and near
  duplicates against all existing splits.
- New images are added to train only. New photos of your own can form a
  separate real-world test set.
- Priority classes:
  - kulfi, paani_puri, pakode, kaathi_rolls, chapati and butter_naan
    (few images, low precision)
  - Taco vs Taquito
  - cheesecake vs apple_pie
  - chicken_curry vs dal_makhani and kadai_paneer
  - idli in varied presentations

**Will not be fixed by any of this**
- Foods outside the 34 classes, such as grilled-chicken salad or egg
  curry, will still be forced into one of the 34.
- That needs a documented schema change, e.g. new classes or an
  "other" class with real training data.
