# FoodCal AI reliability diagnostics

Generated 2026-10-11T02:17:33+00:00 from `models/efficientnet_b0_v2/food_efficientnet_b0_v2_best.pth`. Read-only; nothing was trained.

## Class mapping
- train: ImageFolder class_to_idx matches checkpoint: **True** (16699 images)
- validation: ImageFolder class_to_idx matches checkpoint: **True** (3572 images)
- test: ImageFolder class_to_idx matches checkpoint: **True** (3602 images)

## Accuracy through the API preprocessing path
| Split | Images | Accuracy | Macro F1 | Weighted F1 | ECE |
|---|---|---|---|---|---|
| validation | 3572 | 92.8% | 93.1% | 92.8% | 10.1% |
| test | 3602 | 92.2% | 93.3% | 92.2% | 9.7% |

## Most frequent test confusions
| Actual | Predicted | Count |
|---|---|---|
| Taquito | Taco | 13 |
| Taco | Taquito | 9 |
| ice_cream | cheesecake | 7 |
| cheesecake | ice_cream | 7 |
| apple_pie | cheesecake | 7 |
| Crispy Chicken | Taco | 7 |
| butter_naan | chapati | 6 |
| apple_pie | ice_cream | 6 |
| Taco | Hot Dog | 6 |
| Hot Dog | Taco | 6 |
| sushi | ice_cream | 5 |
| chicken_curry | omelette | 5 |
| Sandwich | Donut | 5 |
| Hot Dog | Sandwich | 5 |
| Fries | Crispy Chicken | 5 |

## Confidence vs observed accuracy (validation / test)
| Threshold | Val share below | Val acc below | Val acc at/above | Test share below | Test acc below | Test acc at/above | Test errors below |
|---|---|---|---|---|---|---|---|
| 0.30 | 1.9% | 36.8% | 93.9% | 1.6% | 42.9% | 92.9% | 11.3% |
| 0.35 | 3.1% | 44.1% | 94.4% | 3.0% | 46.7% | 93.6% | 20.2% |
| 0.40 | 4.4% | 43.6% | 95.1% | 4.4% | 49.0% | 94.1% | 28.4% |
| 0.45 | 5.6% | 50.2% | 95.3% | 6.0% | 52.1% | 94.7% | 36.9% |
| 0.50 | 7.5% | 52.2% | 96.1% | 7.6% | 52.9% | 95.4% | 45.4% |
| 0.55 | 9.0% | 55.8% | 96.5% | 9.2% | 55.2% | 95.9% | 52.5% |
| 0.60 | 11.1% | 58.5% | 97.1% | 11.4% | 56.4% | 96.8% | 63.5% |
| 0.65 | 13.3% | 62.4% | 97.4% | 13.4% | 61.0% | 97.0% | 67.0% |
| 0.70 | 15.6% | 66.6% | 97.6% | 15.8% | 64.0% | 97.5% | 72.7% |
| 0.75 | 19.3% | 71.4% | 97.9% | 19.6% | 68.2% | 98.0% | 79.4% |
| 0.80 | 24.1% | 75.3% | 98.4% | 25.3% | 73.2% | 98.6% | 86.5% |
| 0.85 | 35.4% | 81.9% | 98.8% | 35.8% | 80.0% | 99.0% | 91.5% |
| 0.90 | 56.2% | 87.8% | 99.2% | 57.4% | 87.1% | 99.0% | 94.7% |
| 0.95 | 88.8% | 92.1% | 98.8% | 89.3% | 91.3% | 99.7% | 99.6% |

**Low-confidence threshold: 0.4.** Largest threshold t (step 0.05) for which validation predictions with confidence < t have accuracy < 50%, i.e. the top-1 answer is more likely wrong than right. Chosen on validation only.

## Perturbation probe (test images, synthetic shifts; not real photos)
| Shift | Accuracy | Mean confidence | Mean confidence when wrong | Wrong with >=70% confidence |
|---|---|---|---|---|
| original | 92.2% | 82.4% | 54.4% | 2.1% |
| zoom_in_60pct_centre | 83.8% | 75.3% | 50.5% | 3.6% |
| zoom_out_on_background | 78.5% | 71.8% | 45.0% | 3.5% |
| darker_50pct | 91.9% | 82.2% | 53.2% | 2.0% |
| warm_colour_cast | 91.4% | 81.7% | 54.1% | 2.2% |
| blur_radius_2 | 85.4% | 77.9% | 49.5% | 2.8% |
| rotated_90 | 85.0% | 75.8% | 49.9% | 2.7% |
| jpeg_quality_30 | 79.8% | 74.1% | 50.8% | 4.2% |

## Effect of train/test duplicates
- 247 test images have an exact or possible near-duplicate (dHash <= 3) in train: accuracy 95.5%
- 3355 test images without one: accuracy 91.9%

## Stored history thumbnails (256 px re-encodes of uploads, not the originals)
- `4a29edc0e51c`: ice_cream 30.0%, kulfi 20.0%, Baked Potato 14.4%
- `8fce3d6d988b`: Crispy Chicken 42.1%, masala_dosa 11.5%, sushi 5.2%
- `9033dfaabba5`: Crispy Chicken 38.1%, masala_dosa 11.9%, sushi 5.7%
- `917074f7899c`: idli 64.4%, Fries 2.3%, ice_cream 1.5%
- `a1f8104a0f1b`: chicken_curry 40.5%, pakode 14.2%, Crispy Chicken 8.4%
- `bb5b6922e384`: ice_cream 29.5%, kulfi 20.9%, Baked Potato 14.8%
- `c904d189365c`: kulfi 100.0%, burger 0.0%, chole_bhature 0.0%
- `e3ad93938528`: idli 22.9%, pakode 15.1%, Crispy Chicken 14.8%
- `e5fbe591dc05`: chicken_curry 42.6%, pakode 12.0%, Crispy Chicken 9.0%

