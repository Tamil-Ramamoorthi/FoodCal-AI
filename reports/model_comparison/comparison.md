# Old vs new model on the leakage-free test set

Generated 2026-10-11T03:09:59+00:00. 3304 images from `data/manifests/v2/test_clean.csv`; preprocessing: backend/app/models/classifier.py (same as the API). Class lists identical; every label matches its folder and class index.

- Old: `models/efficientnet_b0/food_efficientnet_b0_best.pth` (epoch 4)
- New: `models/efficientnet_b0_v2/food_efficientnet_b0_v2_best.pth` (epoch 21)

## Overall
| Metric | Old | New | Change |
|---|---|---|---|
| Accuracy | 81.6% | 91.9% | +10.3 pts |
| Macro precision | 81.1% | 93.5% | +12.3 pts |
| Macro recall | 84.3% | 93.0% | +8.7 pts |
| Macro F1 | 81.9% | 93.2% | +11.3 pts |
| Weighted F1 | 81.8% | 91.9% | +10.1 pts |

Correct: old 2696 / new 3035 of 3304.
Paired exact McNemar test: old right & new wrong 72, old wrong & new right 411, p = 1.1e-58.
Bootstrap 95% CI (2000 resamples): accuracy difference +9.1 to +11.4 pts; macro-F1 difference +10.0 to +12.8 pts.

## Promotion gate
**PASSED** (offline test set only). Offline gate only. It deploys nothing; real-world photos (data/real_world_test) must also be evaluated before the production checkpoint is changed.

## Idli → kulfi
| | Old | New |
|---|---|---|
| Idli images predicted as kulfi | 0 of 45 (0.0%) | 0 of 45 (0.0%) |
| Non-kulfi images predicted as kulfi | 23 | 2 |
| Kulfi precision | 50.0% | 92.6% |
| Kulfi recall | 82.1% | 89.3% |

Note: the clean test set contains only dataset-style idli photos; it says nothing about differently presented real photos.

## Weakest 8 classes of the old model (F1)
| Class | Images | Old P / R / F1 | New P / R / F1 |
|---|---|---|---|
| kulfi | 28 | 50.0% / 82.1% / 62.2% | 92.6% / 89.3% / 90.9% |
| pakode | 40 | 65.0% / 65.0% / 65.0% | 100.0% / 92.5% / 96.1% |
| kaathi_rolls | 41 | 57.8% / 90.2% / 70.5% | 87.8% / 87.8% / 87.8% |
| Taco | 214 | 62.5% / 83.2% / 71.3% | 83.3% / 88.3% / 85.7% |
| paani_puri | 21 | 58.8% / 95.2% / 72.7% | 100.0% / 100.0% / 100.0% |
| chapati | 48 | 61.1% / 91.7% / 73.3% | 86.0% / 89.6% / 87.8% |
| apple_pie | 149 | 78.6% / 69.1% / 73.6% | 92.8% / 86.6% / 89.6% |
| Taquito | 191 | 85.5% / 64.9% / 73.8% | 89.4% / 88.5% / 88.9% |

## Most frequent confusions of the old model
| Actual → predicted | Actual images | Old | New |
|---|---|---|---|
| Taquito → Taco | 191 | 28 | 11 |
| Baked Potato → Taco | 182 | 17 | 5 |
| cheesecake → apple_pie | 150 | 16 | 1 |
| Crispy Chicken → Taco | 203 | 14 | 7 |
| Hot Dog → Taco | 181 | 14 | 6 |
| Sandwich → Hot Dog | 210 | 10 | 4 |
| chicken_curry → dal_makhani | 150 | 9 | 1 |
| sushi → cheesecake | 150 | 9 | 4 |
| apple_pie → omelette | 149 | 8 | 1 |
| chicken_curry → kadai_paneer | 150 | 8 | 3 |
| Taquito → kaathi_rolls | 191 | 8 | 1 |
| Taquito → Hot Dog | 191 | 8 | 0 |
| Taco → Hot Dog | 214 | 8 | 5 |
| Crispy Chicken → pakode | 203 | 8 | 0 |
| Taco → kaathi_rolls | 214 | 7 | 1 |

## All classes
| Class | Images | Old P / R / F1 | New P / R / F1 | F1 change |
|---|---|---|---|---|
| idli | 45 | 97.8% / 97.8% / 97.8% | 100.0% / 97.8% / 98.9% | +1.1 |
| jalebi | 40 | 94.7% / 90.0% / 92.3% | 97.4% / 92.5% / 94.9% | +2.6 |
| Donut | 205 | 90.6% / 93.7% / 92.1% | 93.1% / 98.0% / 95.5% | +3.4 |
| dhokla | 37 | 83.3% / 94.6% / 88.6% | 89.7% / 94.6% / 92.1% | +3.5 |
| Fries | 170 | 87.2% / 88.2% / 87.7% | 95.5% / 88.2% / 91.7% | +4.0 |
| momos | 47 | 88.0% / 93.6% / 90.7% | 90.4% / 100.0% / 94.9% | +4.2 |
| burger | 51 | 86.2% / 98.0% / 91.7% | 96.1% / 96.1% / 96.1% | +4.3 |
| chai | 50 | 90.4% / 94.0% / 92.2% | 96.1% / 98.0% / 97.0% | +4.9 |
| ice_cream | 150 | 85.0% / 83.3% / 84.2% | 86.9% / 92.7% / 89.7% | +5.5 |
| pav_bhaji | 43 | 90.5% / 88.4% / 89.4% | 95.3% / 95.3% / 95.3% | +5.9 |
| butter_naan | 46 | 94.4% / 73.9% / 82.9% | 95.1% / 84.8% / 89.7% | +6.7 |
| Sandwich | 210 | 91.9% / 75.2% / 82.7% | 92.6% / 90.0% / 91.3% | +8.6 |
| sushi | 150 | 95.7% / 74.0% / 83.5% | 95.0% / 89.3% / 92.1% | +8.6 |
| cheesecake | 150 | 84.7% / 77.3% / 80.8% | 86.9% / 92.7% / 89.7% | +8.8 |
| Hot Dog | 181 | 79.0% / 85.1% / 81.9% | 91.6% / 90.6% / 91.1% | +9.2 |
| Baked Potato | 182 | 90.5% / 78.6% / 84.1% | 92.9% / 94.0% / 93.4% | +9.3 |
| chole_bhature | 55 | 93.9% / 83.6% / 88.5% | 98.2% / 98.2% / 98.2% | +9.7 |
| Crispy Chicken | 203 | 83.9% / 76.8% / 80.2% | 89.3% / 90.6% / 90.0% | +9.8 |
| omelette | 150 | 82.3% / 80.7% / 81.5% | 93.2% / 91.3% / 92.3% | +10.8 |
| chicken_curry | 150 | 83.0% / 78.0% / 80.4% | 91.3% / 91.3% / 91.3% | +10.9 |
| masala_dosa | 37 | 84.6% / 89.2% / 86.8% | 97.4% / 100.0% / 98.7% | +11.8 |
| fried_rice | 54 | 79.0% / 90.7% / 84.5% | 98.1% / 96.3% / 97.2% | +12.7 |
| pizza | 42 | 77.6% / 90.5% / 83.5% | 97.6% / 95.2% / 96.4% | +12.9 |
| Taco | 214 | 62.5% / 83.2% / 71.3% | 83.3% / 88.3% / 85.7% | +14.4 |
| chapati | 48 | 61.1% / 91.7% / 73.3% | 86.0% / 89.6% / 87.8% | +14.4 |
| Taquito | 191 | 85.5% / 64.9% / 73.8% | 89.4% / 88.5% / 88.9% | +15.1 |
| apple_pie | 149 | 78.6% / 69.1% / 73.6% | 92.8% / 86.6% / 89.6% | +16.0 |
| dal_makhani | 40 | 69.1% / 95.0% / 80.0% | 95.1% / 97.5% / 96.3% | +16.3 |
| samosa | 36 | 83.9% / 72.2% / 77.6% | 97.1% / 91.7% / 94.3% | +16.7 |
| kadai_paneer | 48 | 71.4% / 83.3% / 76.9% | 93.8% / 93.8% / 93.8% | +16.8 |
| kaathi_rolls | 41 | 57.8% / 90.2% / 70.5% | 87.8% / 87.8% / 87.8% | +17.3 |
| paani_puri | 21 | 58.8% / 95.2% / 72.7% | 100.0% / 100.0% / 100.0% | +27.3 |
| kulfi | 28 | 50.0% / 82.1% / 62.2% | 92.6% / 89.3% / 90.9% | +28.7 |
| pakode | 40 | 65.0% / 65.0% / 65.0% | 100.0% / 92.5% / 96.1% | +31.1 |

Full 34×34 confusion matrices: `confusion_matrix_old.csv`, `confusion_matrix_new.csv` (rows actual, columns predicted).
