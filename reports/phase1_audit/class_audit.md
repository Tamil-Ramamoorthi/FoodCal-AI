# Phase 1 class-wise dataset audit

Generated 2026-10-10T15:39:30+00:00. Read-only: no image was moved, renamed, deleted or relabelled.

## Summary
- Class folders match checkpoint classes: {'train': True, 'validation': True, 'test': True}
- Source images recovered for 23873 of 23873 processed files; 23871 verified by dHash.
- Train imbalance: Hot Dog 1083 vs paani_puri 100 images (ratio 10.8).
- Duplicate groups (exact or dHash <= 3): 1014 groups, 2234 images; 30 groups mix labels.
- Leakage: 298 test images have a copy in train or validation; 243 validation images have a copy in train.
- Images the current model contradicts with >= 90% confidence: 90 (candidates for manual review, not confirmed label errors).
- Current model, full test split: accuracy 82.6%, macro F1 82.7% (3602 images)
- Current model, leakage-free test subset: accuracy 81.6%, macro F1 81.9% (3304 images)

## Per class
| Class | Train | Val | Test | Train vs largest | Dup images | Test leaked | Cross-label dup | Model disagrees >=90% (train/val/test) | Median orig. short side | Clean-test recall | Clean-test precision | Clean-test F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| kulfi | 149 | 32 | 33 | 0.14 | 29 | 5 | 0 | 0/1/0 | n/a | 82.1% | 50.0% | 62.2% |
| pakode | 188 | 40 | 41 | 0.17 | 4 | 1 | 0 | 0/1/0 | n/a | 65.0% | 65.0% | 65.0% |
| kaathi_rolls | 195 | 41 | 43 | 0.18 | 20 | 2 | 0 | 0/0/0 | n/a | 90.2% | 57.8% | 70.5% |
| Taco | 1050 | 225 | 225 | 0.97 | 83 | 11 | 6 | 7/2/0 | n/a | 83.2% | 62.5% | 71.3% |
| paani_puri | 100 | 21 | 23 | 0.09 | 8 | 2 | 0 | 0/0/0 | n/a | 95.2% | 58.8% | 72.7% |
| chapati | 228 | 49 | 50 | 0.21 | 16 | 2 | 2 | 0/0/0 | n/a | 91.7% | 61.1% | 73.3% |
| apple_pie | 700 | 150 | 150 | 0.65 | 7 | 1 | 1 | 3/1/0 | n/a | 69.1% | 78.6% | 73.6% |
| Taquito | 1050 | 225 | 225 | 0.97 | 315 | 34 | 3 | 8/3/0 | n/a | 64.9% | 85.5% | 73.8% |
| kadai_paneer | 233 | 50 | 51 | 0.22 | 18 | 3 | 2 | 0/2/0 | n/a | 83.3% | 71.4% | 76.9% |
| samosa | 170 | 36 | 38 | 0.16 | 12 | 2 | 0 | 0/1/0 | n/a | 72.2% | 83.9% | 77.6% |
| dal_makhani | 206 | 44 | 45 | 0.19 | 28 | 5 | 0 | 0/0/0 | n/a | 95.0% | 69.1% | 80.0% |
| Crispy Chicken | 1050 | 225 | 225 | 0.97 | 109 | 22 | 5 | 7/1/3 | n/a | 76.8% | 83.9% | 80.2% |
| chicken_curry | 700 | 150 | 150 | 0.65 | 2 | 0 | 2 | 1/1/1 | n/a | 78.0% | 83.0% | 80.4% |
| cheesecake | 700 | 150 | 150 | 0.65 | 2 | 0 | 2 | 8/3/2 | n/a | 77.3% | 84.7% | 80.8% |
| omelette | 700 | 150 | 150 | 0.65 | 4 | 0 | 0 | 2/0/1 | n/a | 80.7% | 82.3% | 81.5% |
| Hot Dog | 1083 | 232 | 233 | 1.00 | 394 | 52 | 11 | 4/0/0 | n/a | 85.1% | 79.0% | 81.9% |
| Sandwich | 1050 | 225 | 225 | 0.97 | 98 | 15 | 8 | 2/1/3 | n/a | 75.2% | 91.9% | 82.7% |
| butter_naan | 214 | 46 | 47 | 0.20 | 23 | 1 | 1 | 1/1/1 | n/a | 73.9% | 94.4% | 82.9% |
| sushi | 700 | 150 | 150 | 0.65 | 0 | 0 | 0 | 2/1/0 | n/a | 74.0% | 95.7% | 83.5% |
| pizza | 192 | 41 | 42 | 0.18 | 8 | 0 | 2 | 0/0/0 | n/a | 90.5% | 77.6% | 83.5% |
| Baked Potato | 1050 | 225 | 225 | 0.97 | 294 | 43 | 6 | 4/1/0 | n/a | 78.6% | 90.5% | 84.1% |
| ice_cream | 700 | 150 | 150 | 0.65 | 2 | 0 | 2 | 4/0/1 | n/a | 83.3% | 85.0% | 84.2% |
| fried_rice | 244 | 52 | 54 | 0.23 | 3 | 0 | 1 | 0/0/1 | n/a | 90.7% | 79.0% | 84.5% |
| masala_dosa | 191 | 40 | 42 | 0.18 | 25 | 5 | 0 | 0/0/0 | n/a | 89.2% | 84.6% | 86.8% |
| Fries | 1050 | 225 | 225 | 0.97 | 461 | 55 | 11 | 1/0/1 | n/a | 88.2% | 87.2% | 87.7% |
| chole_bhature | 263 | 56 | 57 | 0.24 | 24 | 2 | 0 | 0/0/0 | n/a | 83.6% | 93.9% | 88.5% |
| dhokla | 171 | 36 | 38 | 0.16 | 10 | 1 | 0 | 0/0/0 | n/a | 94.6% | 83.3% | 88.6% |
| pav_bhaji | 213 | 45 | 47 | 0.20 | 32 | 4 | 2 | 0/0/1 | n/a | 88.4% | 90.5% | 89.4% |
| momos | 226 | 48 | 49 | 0.21 | 23 | 2 | 1 | 0/0/0 | n/a | 93.6% | 88.0% | 90.7% |
| burger | 231 | 49 | 51 | 0.21 | 15 | 0 | 5 | 0/0/0 | n/a | 98.0% | 86.2% | 91.7% |
| Donut | 1050 | 225 | 225 | 0.97 | 120 | 20 | 41 | 0/1/0 | n/a | 93.7% | 90.6% | 92.1% |
| chai | 240 | 51 | 53 | 0.22 | 11 | 3 | 1 | 0/0/0 | n/a | 94.0% | 90.4% | 92.2% |
| jalebi | 201 | 43 | 44 | 0.19 | 18 | 4 | 1 | 0/0/0 | n/a | 90.0% | 94.7% | 92.3% |
| idli | 211 | 45 | 46 | 0.19 | 16 | 1 | 0 | 0/0/0 | n/a | 97.8% | 97.8% | 97.8% |

## Most frequent confusions (leakage-free test subset)
| Actual | Predicted | Count | Share of actual class |
|---|---|---|---|
| Taquito | Taco | 28 | 14.7% |
| Baked Potato | Taco | 17 | 9.3% |
| cheesecake | apple_pie | 16 | 10.7% |
| Hot Dog | Taco | 14 | 7.7% |
| Crispy Chicken | Taco | 14 | 6.9% |
| Sandwich | Hot Dog | 10 | 4.8% |
| sushi | cheesecake | 9 | 6.0% |
| chicken_curry | dal_makhani | 9 | 6.0% |
| chicken_curry | kadai_paneer | 8 | 5.3% |
| apple_pie | omelette | 8 | 5.4% |
| Taquito | kaathi_rolls | 8 | 4.2% |
| Taquito | Hot Dog | 8 | 4.2% |
| Taco | Hot Dog | 8 | 3.7% |
| Crispy Chicken | pakode | 8 | 3.9% |
| omelette | chicken_curry | 7 | 4.7% |
| butter_naan | chapati | 7 | 15.2% |
| apple_pie | ice_cream | 7 | 4.7% |
| Taco | kaathi_rolls | 7 | 3.3% |
| Sandwich | Crispy Chicken | 7 | 3.3% |
| ice_cream | kulfi | 6 | 4.0% |

## Cross-label duplicate groups (same or near-identical image, different labels)
- `test/Baked Potato/000120.jpg` (Baked Potato); `test/Hot Dog/000202.jpg` (Hot Dog); `train/Donut/000486.jpg` (Donut); `train/Hot Dog/000031.jpg` (Hot Dog); `validation/Baked Potato/000008.jpg` (Baked Potato); `validation/Baked Potato/000150.jpg` (Baked Potato)
- `test/Baked Potato/000123.jpg` (Baked Potato); `train/Donut/000891.jpg` (Donut); `validation/burger/000024.jpg` (burger); `validation/Fries/000089.jpg` (Fries)
- `test/butter_naan/000025.jpg` (butter_naan); `test/kadai_paneer/000014.jpg` (kadai_paneer)
- `test/chai/000014.jpg` (chai); `test/Donut/000142.jpg` (Donut); `test/Donut/000209.jpg` (Donut); `test/Donut/000213.jpg` (Donut); `train/Donut/000018.jpg` (Donut); `train/Donut/000023.jpg` (Donut); `train/Donut/000259.jpg` (Donut); `train/Donut/000271.jpg` (Donut); `train/Donut/000288.jpg` (Donut); `train/Donut/000389.jpg` (Donut); `train/Donut/000402.jpg` (Donut); `train/Donut/000435.jpg` (Donut); `train/Donut/000437.jpg` (Donut); `train/Donut/000464.jpg` (Donut); `train/Donut/000699.jpg` (Donut); `train/Donut/000710.jpg` (Donut); `train/Donut/000813.jpg` (Donut); `train/Donut/000967.jpg` (Donut); `train/Donut/000974.jpg` (Donut); `validation/Donut/000129.jpg` (Donut); `validation/Donut/000176.jpg` (Donut); `validation/Donut/000194.jpg` (Donut)
- `test/chapati/000016.jpg` (chapati); `train/chapati/000151.jpg` (chapati); `train/ice_cream/000658.jpg` (ice_cream)
- `test/Crispy Chicken/000045.jpg` (Crispy Chicken); `test/Donut/000104.jpg` (Donut); `validation/jalebi/000029.jpg` (jalebi)
- `test/Crispy Chicken/000057.jpg` (Crispy Chicken); `test/Fries/000077.jpg` (Fries); `test/Hot Dog/000060.jpg` (Hot Dog); `train/cheesecake/000235.jpg` (cheesecake); `train/Crispy Chicken/000179.jpg` (Crispy Chicken); `train/Donut/000017.jpg` (Donut); `train/Donut/000073.jpg` (Donut); `train/Donut/000382.jpg` (Donut); `train/fried_rice/000161.jpg` (fried_rice); `train/Fries/000529.jpg` (Fries); `train/Fries/000551.jpg` (Fries); `train/Fries/000955.jpg` (Fries); `train/ice_cream/000697.jpg` (ice_cream); `train/momos/000104.jpg` (momos); `train/pizza/000023.jpg` (pizza); `train/Taquito/000692.jpg` (Taquito); `validation/Donut/000132.jpg` (Donut)
- `test/Crispy Chicken/000161.jpg` (Crispy Chicken); `test/Donut/000153.jpg` (Donut); `test/Taco/000045.jpg` (Taco); `train/Donut/000590.jpg` (Donut); `train/Sandwich/000186.jpg` (Sandwich)
- `test/Fries/000172.jpg` (Fries); `train/Taco/000564.jpg` (Taco)
- `test/Hot Dog/000025.jpg` (Hot Dog); `train/Baked Potato/000250.jpg` (Baked Potato)
- `test/pav_bhaji/000031.jpg` (pav_bhaji); `train/pizza/000040.jpg` (pizza)
- `test/Sandwich/000018.jpg` (Sandwich); `train/Sandwich/000402.jpg` (Sandwich); `train/Sandwich/000413.jpg` (Sandwich); `validation/Hot Dog/000217.jpg` (Hot Dog)
- `test/Taquito/000199.jpg` (Taquito); `train/Taco/000975.jpg` (Taco)
- `test/Taquito/000210.jpg` (Taquito); `train/Taco/000762.jpg` (Taco)
- `train/Baked Potato/000543.jpg` (Baked Potato); `validation/Fries/000177.jpg` (Fries)
- `train/burger/000057.jpg` (burger); `train/Donut/000873.jpg` (Donut); `train/Fries/000744.jpg` (Fries); `train/Taco/000652.jpg` (Taco); `validation/Sandwich/000034.jpg` (Sandwich)
- `train/burger/000191.jpg` (burger); `train/Donut/000035.jpg` (Donut)
- `train/burger/000196.jpg` (burger); `validation/Donut/000003.jpg` (Donut); `validation/Donut/000215.jpg` (Donut)
- `train/cheesecake/000659.jpg` (cheesecake); `train/kadai_paneer/000149.jpg` (kadai_paneer)
- `train/chicken_curry/000081.jpg` (chicken_curry); `train/Fries/000815.jpg` (Fries)
- `train/chicken_curry/000429.jpg` (chicken_curry); `validation/apple_pie/000107.jpg` (apple_pie)
- `train/Crispy Chicken/000975.jpg` (Crispy Chicken); `validation/burger/000034.jpg` (burger)
- `train/Donut/000102.jpg` (Donut); `train/Sandwich/000095.jpg` (Sandwich); `train/Sandwich/000662.jpg` (Sandwich)
- `train/Donut/000191.jpg` (Donut); `train/Donut/000648.jpg` (Donut); `train/Hot Dog/000403.jpg` (Hot Dog); `train/Hot Dog/000433.jpg` (Hot Dog); `train/Hot Dog/000647.jpg` (Hot Dog)
- `train/Donut/000496.jpg` (Donut); `train/Sandwich/000314.jpg` (Sandwich)
- `train/Donut/000498.jpg` (Donut); `validation/Fries/000080.jpg` (Fries)
- `train/Donut/000609.jpg` (Donut); `train/Hot Dog/000568.jpg` (Hot Dog)
- `train/Fries/000688.jpg` (Fries); `train/Hot Dog/000578.jpg` (Hot Dog)
- `train/pav_bhaji/000071.jpg` (pav_bhaji); `validation/Hot Dog/000146.jpg` (Hot Dog)
- `validation/Donut/000042.jpg` (Donut); `validation/Taco/000143.jpg` (Taco)
