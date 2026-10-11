# FoodCal AI reliability diagnostics

Generated 2026-10-10T14:53:21+00:00 from `models/efficientnet_b0/food_efficientnet_b0_best.pth`. Read-only; nothing was trained.

## Class mapping
- train: ImageFolder class_to_idx matches checkpoint: **True** (16699 images)
- validation: ImageFolder class_to_idx matches checkpoint: **True** (3572 images)
- test: ImageFolder class_to_idx matches checkpoint: **True** (3602 images)

## Accuracy through the API preprocessing path
| Split | Images | Accuracy | Macro F1 | Weighted F1 | ECE |
|---|---|---|---|---|---|
| validation | 3572 | 83.1% | 83.2% | 83.3% | 7.6% |
| test | 3602 | 82.6% | 82.7% | 82.8% | 7.6% |

## Most frequent test confusions
| Actual | Predicted | Count |
|---|---|---|
| Taquito | Taco | 31 |
| Baked Potato | Taco | 18 |
| cheesecake | apple_pie | 16 |
| Hot Dog | Taco | 14 |
| Crispy Chicken | Taco | 14 |
| Sandwich | Hot Dog | 10 |
| sushi | cheesecake | 9 |
| chicken_curry | dal_makhani | 9 |
| Taquito | kaathi_rolls | 9 |
| chicken_curry | kadai_paneer | 8 |
| apple_pie | omelette | 8 |
| Taquito | Hot Dog | 8 |
| Taco | kaathi_rolls | 8 |
| Taco | Hot Dog | 8 |
| Crispy Chicken | pakode | 8 |

## Confidence vs observed accuracy (validation / test)
| Threshold | Val share below | Val acc below | Val acc at/above | Test share below | Test acc below | Test acc at/above | Test errors below |
|---|---|---|---|---|---|---|---|
| 0.30 | 4.8% | 33.7% | 85.6% | 5.0% | 30.4% | 85.4% | 20.1% |
| 0.35 | 7.8% | 37.1% | 87.0% | 8.5% | 38.7% | 86.7% | 29.8% |
| 0.40 | 11.4% | 41.6% | 88.4% | 12.5% | 42.5% | 88.3% | 41.1% |
| 0.45 | 15.6% | 44.4% | 90.3% | 16.2% | 44.8% | 89.9% | 51.5% |
| 0.50 | 19.7% | 48.7% | 91.5% | 20.3% | 47.3% | 91.6% | 61.4% |
| 0.55 | 23.7% | 52.2% | 92.7% | 24.2% | 50.0% | 93.0% | 69.4% |
| 0.60 | 27.5% | 54.9% | 93.9% | 28.3% | 53.9% | 93.9% | 74.8% |
| 0.65 | 30.9% | 57.1% | 94.8% | 31.9% | 56.1% | 95.0% | 80.5% |
| 0.70 | 34.7% | 59.6% | 95.6% | 36.3% | 58.9% | 96.1% | 85.6% |
| 0.75 | 39.8% | 62.4% | 96.8% | 40.7% | 61.8% | 96.9% | 89.3% |
| 0.80 | 45.1% | 65.7% | 97.5% | 45.4% | 64.7% | 97.5% | 92.0% |
| 0.85 | 51.0% | 68.8% | 98.1% | 51.4% | 67.7% | 98.3% | 95.2% |
| 0.90 | 58.5% | 72.2% | 98.6% | 59.8% | 71.6% | 99.0% | 97.6% |
| 0.95 | 70.6% | 76.4% | 99.1% | 71.0% | 75.7% | 99.5% | 99.2% |

**Low-confidence threshold: 0.5.** Largest threshold t (step 0.05) for which validation predictions with confidence < t have accuracy < 50%, i.e. the top-1 answer is more likely wrong than right. Chosen on validation only.

## Perturbation probe (test images, synthetic shifts; not real photos)
| Shift | Accuracy | Mean confidence | Mean confidence when wrong | Wrong with >=70% confidence |
|---|---|---|---|---|
| original | 82.6% | 75.0% | 47.3% | 2.5% |
| zoom_in_60pct_centre | 76.5% | 68.5% | 43.5% | 2.2% |
| zoom_out_on_background | 51.8% | 51.1% | 36.3% | 3.2% |
| darker_50pct | 78.7% | 71.2% | 44.5% | 2.7% |
| warm_colour_cast | 80.7% | 73.5% | 47.5% | 3.2% |
| blur_radius_2 | 52.6% | 57.2% | 42.6% | 4.8% |
| rotated_90 | 72.0% | 68.1% | 47.1% | 3.9% |
| jpeg_quality_30 | 72.3% | 68.5% | 46.4% | 3.4% |

## Effect of train/test duplicates
- 247 test images have an exact or possible near-duplicate (dHash <= 3) in train: accuracy 94.7%
- 3355 test images without one: accuracy 81.7%

## Stored history thumbnails (256 px re-encodes of uploads, not the originals)
- `9033dfaabba5`: kadai_paneer 40.0%, Taco 12.8%, Crispy Chicken 12.7%
- `a1f8104a0f1b`: chicken_curry 66.7%, paani_puri 6.9%, kadai_paneer 3.4%
- `bb5b6922e384`: kulfi 84.9%, ice_cream 6.0%, cheesecake 3.6%

## JPEG re-encode step
- 1000 source originals: top-1 agreement 97.1%, mean |confidence difference| 3.8%

