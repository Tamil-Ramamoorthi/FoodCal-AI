# FoodCal AI real-world evaluation

Separate from the offline test-set report (`models/efficientnet_b0/evaluation_report.json`).

- Generated: 2026-10-11T05:22:11+00:00
- Status: **ok**
- Checkpoint: `models/efficientnet_b0/food_efficientnet_b0_best.pth` (epoch 4, SHA-256 `39363ab4f70d68fb...`)
- Image manifest: 263 images, SHA-256 `b4ed5ce0a4807567...`
- Dataset: `data/real_world_test`
- Device: cuda
- Preprocessing: EXIF orientation applied, converted to RGB, resized to 224x224 with LANCZOS (no cropping, aspect ratio not preserved), then Resize((224, 224)), ToTensor and ImageNet normalization. Mirrors training/prepare_dataset.py + training/evaluate_efficientnet.py.
- Valid images: 263; invalid images: 0

## Overall metrics (all valid images)
| Metric | Value |
|---|---|
| Images evaluated | 263 |
| Correct | 144 |
| Incorrect | 119 |
| Accuracy | 54.8% |
| Macro precision (over 34 classes with images) | 53.0% |
| Macro recall (over 34 classes with images) | 52.0% |
| Macro F1 (over 34 classes with images) | 48.5% |
| Weighted F1 | 51.5% |

## Metrics excluding images that overlap training data
| Metric | Value |
|---|---|
| Images evaluated | 257 |
| Correct | 141 |
| Incorrect | 116 |
| Accuracy | 54.9% |
| Macro precision (over 34 classes with images) | 53.4% |
| Macro recall (over 34 classes with images) | 52.4% |
| Macro F1 (over 34 classes with images) | 48.9% |
| Weighted F1 | 51.6% |

## Model confidence (not accuracy)
- Mean top-1 confidence: 58.6%
- Mean confidence when correct: 71.2%
- Mean confidence when incorrect: 43.4%

| Confidence range | Images | Mean confidence | Observed accuracy |
|---|---|---|---|
| 10%-20% | 13 | 16.3% | 15.4% |
| 20%-30% | 33 | 25.7% | 21.2% |
| 30%-40% | 34 | 35.3% | 29.4% |
| 40%-50% | 32 | 45.1% | 34.4% |
| 50%-60% | 22 | 54.4% | 45.5% |
| 60%-70% | 40 | 64.2% | 70.0% |
| 70%-80% | 20 | 76.0% | 75.0% |
| 80%-90% | 19 | 85.7% | 73.7% |
| 90%-100% | 50 | 96.1% | 94.0% |

## Per-class metrics
| Class | Images | Predicted | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Baked Potato | 5 | 0 | n/a | 0.0% | 0.0% |
| Crispy Chicken | 6 | 1 | 100.0% | 16.7% | 28.6% |
| Donut | 9 | 7 | 100.0% | 77.8% | 87.5% |
| Fries | 9 | 9 | 88.9% | 88.9% | 88.9% |
| Hot Dog | 10 | 10 | 90.0% | 90.0% | 90.0% |
| Sandwich | 8 | 1 | 100.0% | 12.5% | 22.2% |
| Taco | 7 | 3 | 66.7% | 28.6% | 40.0% |
| Taquito | 9 | 0 | n/a | 0.0% | 0.0% |
| apple pie | 6 | 3 | 33.3% | 16.7% | 22.2% |
| burger | 7 | 10 | 60.0% | 85.7% | 70.6% |
| butter naan | 6 | 3 | 66.7% | 33.3% | 44.4% |
| chai | 6 | 10 | 60.0% | 100.0% | 75.0% |
| chapati | 5 | 16 | 18.8% | 60.0% | 28.6% |
| cheesecake | 7 | 5 | 40.0% | 28.6% | 33.3% |
| chicken curry | 7 | 7 | 14.3% | 14.3% | 14.3% |
| chole bhature | 7 | 0 | n/a | 0.0% | 0.0% |
| dal makhani | 7 | 19 | 31.6% | 85.7% | 46.2% |
| dhokla | 7 | 12 | 33.3% | 57.1% | 42.1% |
| fried rice | 7 | 7 | 42.9% | 42.9% | 42.9% |
| ice cream | 8 | 8 | 62.5% | 62.5% | 62.5% |
| idli | 11 | 6 | 66.7% | 36.4% | 47.1% |
| jalebi | 9 | 12 | 75.0% | 100.0% | 85.7% |
| kaathi rolls | 7 | 5 | 60.0% | 42.9% | 50.0% |
| kadai paneer | 8 | 8 | 37.5% | 37.5% | 37.5% |
| kulfi | 5 | 5 | 40.0% | 40.0% | 40.0% |
| masala dosa | 8 | 8 | 62.5% | 62.5% | 62.5% |
| momos | 9 | 9 | 55.6% | 55.6% | 55.6% |
| omelette | 7 | 6 | 33.3% | 28.6% | 30.8% |
| paani puri | 9 | 8 | 62.5% | 55.6% | 58.8% |
| pakode | 9 | 9 | 66.7% | 66.7% | 66.7% |
| pav bhaji | 8 | 11 | 54.5% | 75.0% | 63.2% |
| pizza | 8 | 11 | 63.6% | 87.5% | 73.7% |
| samosa | 7 | 15 | 40.0% | 85.7% | 54.5% |
| sushi | 15 | 19 | 73.7% | 93.3% | 82.4% |

## Confusion matrix
Rows are actual classes, columns are predicted classes. Only classes that appear as an actual or predicted label are shown.

| actual \ predicted | Baked Potato | Crispy Chicken | Donut | Fries | Hot Dog | Sandwich | Taco | Taquito | apple pie | burger | butter naan | chai | chapati | cheesecake | chicken curry | chole bhature | dal makhani | dhokla | fried rice | ice cream | idli | jalebi | kaathi rolls | kadai paneer | kulfi | masala dosa | momos | omelette | paani puri | pakode | pav bhaji | pizza | samosa | sushi |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Baked Potato | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| Crispy Chicken | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 |
| Donut | 0 | 0 | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| Fries | 0 | 0 | 0 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| Hot Dog | 0 | 0 | 0 | 0 | 9 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Sandwich | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 1 |
| Taco | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Taquito | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 1 | 0 |
| apple pie | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 3 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| burger | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| butter naan | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 2 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| chai | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| chapati | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| cheesecake | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 1 |
| chicken curry | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| chole bhature | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 1 | 0 |
| dal makhani | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| dhokla | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 1 | 0 |
| fried rice | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| ice cream | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| idli | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 4 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 |
| jalebi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 9 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| kaathi rolls | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 |
| kadai paneer | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 1 |
| kulfi | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| masala dosa | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 |
| momos | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 5 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| omelette | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| paani puri | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 0 | 0 | 0 | 1 | 1 |
| pakode | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 0 | 0 | 0 |
| pav bhaji | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 6 | 0 | 0 | 0 |
| pizza | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 7 | 0 | 0 |
| samosa | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 0 |
| sushi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 14 |

## Incorrect predictions (119)
| Image | Actual | Predicted | Confidence |
|---|---|---|---|
| `data/real_world_test/Baked Potato/pexels-lorencastillo-9219092.jpg` | Baked Potato | jalebi | 24.1% |
| `data/real_world_test/Baked Potato/pexels-nicola-barts-7936637.jpg` | Baked Potato | samosa | 46.0% |
| `data/real_world_test/Baked Potato/pexels-nobleseed-34495400.jpg` | Baked Potato | apple pie | 27.4% |
| `data/real_world_test/Baked Potato/pexels-polina-tankilevitch-4110468.jpg` | Baked Potato | kulfi | 47.8% |
| `data/real_world_test/Baked Potato/pexels-polina-tankilevitch-4110470.jpg` | Baked Potato | apple pie | 30.2% |
| `data/real_world_test/Crispy Chicken/pexels-andaru-firmansyah-2152289371-37279764.jpg` | Crispy Chicken | pizza | 11.2% |
| `data/real_world_test/Crispy Chicken/pexels-chicken-mackay-3652946-5474676.jpg` | Crispy Chicken | burger | 64.6% |
| `data/real_world_test/Crispy Chicken/pexels-foturecreative-33101857.jpg` | Crispy Chicken | chapati | 28.7% |
| `data/real_world_test/Crispy Chicken/pexels-mohammad-miladi-134457715-10206125.jpg` | Crispy Chicken | omelette | 41.1% |
| `data/real_world_test/Crispy Chicken/pexels-qamar-rehman-94539242-11502374.jpg` | Crispy Chicken | chai | 22.7% |
| `data/real_world_test/Donut/pexels-perfect-lens-8358569.jpg` | Donut | pizza | 61.5% |
| `data/real_world_test/Donut/pexels-tijana-drndarski-449691-3338681.jpg` | Donut | burger | 50.2% |
| `data/real_world_test/Fries/pexels-jahratreza-39034206.jpg` | Fries | samosa | 29.5% |
| `data/real_world_test/Hot Dog/pexels-alleksana-4113501.jpg` | Hot Dog | kaathi rolls | 26.5% |
| `data/real_world_test/Sandwich/pexels-andrea-prochilo-3062027-34421469.jpg` | Sandwich | cheesecake | 26.0% |
| `data/real_world_test/Sandwich/pexels-karola-g2-6467.jpg` | Sandwich | idli | 46.7% |
| `data/real_world_test/Sandwich/pexels-nano-erdozain-120534369-19202817.jpg` | Sandwich | dal makhani | 16.1% |
| `data/real_world_test/Sandwich/pexels-nano-erdozain-120534369-19202829.jpg` | Sandwich | fried rice | 12.4% |
| `data/real_world_test/Sandwich/pexels-thongtran95-16436245.jpg` | Sandwich | sushi | 51.0% |
| `data/real_world_test/Sandwich/pexels-tochtliyeung1996-15362507.jpg` | Sandwich | omelette | 19.5% |
| `data/real_world_test/Sandwich/pexels-tolgaaslanturk-11256670.jpg` | Sandwich | samosa | 37.8% |
| `data/real_world_test/Taco/pexels-roman-odintsov-4958526.jpg` | Taco | dal makhani | 66.7% |
| `data/real_world_test/Taco/pexels-roman-odintsov-4958779.jpg` | Taco | fried rice | 66.1% |
| `data/real_world_test/Taco/pexels-roman-odintsov-5837188.jpg` | Taco | ice cream | 35.7% |
| `data/real_world_test/Taco/pexels-roman-odintsov-5837196.jpg` | Taco | burger | 38.9% |
| `data/real_world_test/Taco/pexels-styves-exantus-7162424-7041784.jpg` | Taco | dhokla | 19.5% |
| `data/real_world_test/Taquito/pexels-anthony-rahayel-125801377-32592710.jpg` | Taquito | kulfi | 51.0% |
| `data/real_world_test/Taquito/pexels-cristian-rojas-8448323.jpg` | Taquito | masala dosa | 30.5% |
| `data/real_world_test/Taquito/pexels-fausto-hernandez-196511575-12326169.jpg` | Taquito | momos | 40.2% |
| `data/real_world_test/Taquito/pexels-jonathan-reynaga-861774-17429044.jpg` | Taquito | dhokla | 33.4% |
| `data/real_world_test/Taquito/pexels-jonathan-reynaga-861774-17429144.jpg` | Taquito | dhokla | 22.7% |
| `data/real_world_test/Taquito/pexels-jonathan-reynaga-861774-17429153.jpg` | Taquito | dal makhani | 74.8% |
| `data/real_world_test/Taquito/pexels-nano-erdozain-120534369-27905921.jpg` | Taquito | samosa | 36.8% |
| `data/real_world_test/Taquito/pexels-rachel-claire-5531155.jpg` | Taquito | dhokla | 31.9% |
| `data/real_world_test/Taquito/pexels-ruben-dao-cuentas-239930863-34303345.jpg` | Taquito | omelette | 29.2% |
| `data/real_world_test/apple_pie/pexels-christina-voinova-42707474-7368070.jpg` | apple pie | cheesecake | 49.9% |
| `data/real_world_test/apple_pie/pexels-clement-bruneau-2159570516-36340982.jpg` | apple pie | momos | 65.6% |
| `data/real_world_test/apple_pie/pexels-hilalbulbul-12782728 (1).jpg` | apple pie | chapati | 41.6% |
| `data/real_world_test/apple_pie/pexels-hilalbulbul-12782728.jpg` | apple pie | chapati | 41.6% |
| `data/real_world_test/apple_pie/pexels-hilalbulbul-7552556.jpg` | apple pie | chapati | 44.0% |
| `data/real_world_test/burger/pexels-hacihuseyinerol-36007382.jpg` | burger | Hot Dog | 27.7% |
| `data/real_world_test/butter_naan/pexels-gagankaur-16851842.jpg` | butter naan | fried rice | 72.3% |
| `data/real_world_test/butter_naan/pexels-miltonphotography-35066811.jpg` | butter naan | dhokla | 24.1% |
| `data/real_world_test/butter_naan/pexels-theshuttervision-15131306.jpg` | butter naan | chapati | 61.3% |
| `data/real_world_test/butter_naan/pexels-yarensphotoworks-30556657.jpg` | butter naan | chapati | 93.0% |
| `data/real_world_test/chapati/pexels-sincegameon-20045788.jpg` | chapati | chai | 79.7% |
| `data/real_world_test/chapati/pexels-usman-yousaf-708951-5589943.jpg` | chapati | kaathi rolls | 56.9% |
| `data/real_world_test/cheesecake/pexels-marijana-blazaric-571842-13350861.jpg` | cheesecake | pav bhaji | 38.5% |
| `data/real_world_test/cheesecake/pexels-mlkbnl-25746353.jpg` | cheesecake | sushi | 18.7% |
| `data/real_world_test/cheesecake/pexels-mlkbnl-25746355.jpg` | cheesecake | kadai paneer | 31.0% |
| `data/real_world_test/cheesecake/pexels-mlkbnl-25746365.jpg` | cheesecake | pizza | 26.9% |
| `data/real_world_test/cheesecake/pexels-vivanpunk-34637999.jpg` | cheesecake | dal makhani | 39.7% |
| `data/real_world_test/chicken_curry/pexels-anna-pou-9345647.jpg` | chicken curry | chapati | 32.7% |
| `data/real_world_test/chicken_curry/pexels-drmkhawarnazir-38003495.jpg` | chicken curry | kadai paneer | 87.6% |
| `data/real_world_test/chicken_curry/pexels-iamabdullahsheik-9792458.jpg` | chicken curry | dal makhani | 17.2% |
| `data/real_world_test/chicken_curry/pexels-jkreat0r-32986463.jpg` | chicken curry | dal makhani | 70.4% |
| `data/real_world_test/chicken_curry/pexels-kunal-lakhotia-781256899-29684985.jpg` | chicken curry | kadai paneer | 69.0% |
| `data/real_world_test/chicken_curry/pexels-pinaki-panda-3019803-33643313.jpg` | chicken curry | dal makhani | 43.4% |
| `data/real_world_test/chole_bhature/pexels-adhwaith-chandran-214377112-15995519.jpg` | chole bhature | samosa | 57.5% |
| `data/real_world_test/chole_bhature/pexels-drmkhawarnazir-9287035.jpg` | chole bhature | dal makhani | 94.0% |
| `data/real_world_test/chole_bhature/pexels-inaam-ashiq-2155239180-35128164.jpg` | chole bhature | Fries | 13.0% |
| `data/real_world_test/chole_bhature/pexels-jack-baghel-2199968-20446423.jpg` | chole bhature | pav bhaji | 23.9% |
| `data/real_world_test/chole_bhature/pexels-llana-79788650-15717898.jpg` | chole bhature | chapati | 31.3% |
| `data/real_world_test/chole_bhature/pexels-sabyasachi-das-45419624-29066704.jpg` | chole bhature | chapati | 47.5% |
| `data/real_world_test/chole_bhature/pexels-yash-acharya-214379065-11818239.jpg` | chole bhature | pav bhaji | 20.9% |
| `data/real_world_test/dal_makhani/pexels-kunal-lakhotia-781256899-28674708.jpg` | dal makhani | chicken curry | 52.1% |
| `data/real_world_test/dhokla/pexels-aayushrawat-8414675.jpg` | dhokla | paani puri | 40.0% |
| `data/real_world_test/dhokla/pexels-ali-dashti-506667798-26341215.jpg` | dhokla | paani puri | 18.6% |
| `data/real_world_test/dhokla/pexels-marcelochagas-3324431.jpg` | dhokla | samosa | 51.7% |
| `data/real_world_test/fried_rice/pexels-4louette-11129150.jpg` | fried rice | kadai paneer | 28.9% |
| `data/real_world_test/fried_rice/pexels-anntarazevich-6937447.jpg` | fried rice | chapati | 23.7% |
| `data/real_world_test/fried_rice/pexels-anntarazevich-6937451.jpg` | fried rice | jalebi | 28.5% |
| `data/real_world_test/fried_rice/pexels-anntarazevich-6937457.jpg` | fried rice | chapati | 22.8% |
| `data/real_world_test/ice_cream/pexels-bank-12843687.jpg` | ice cream | fried rice | 27.0% |
| `data/real_world_test/ice_cream/pexels-ds-stories-9227980.jpg` | ice cream | momos | 35.4% |
| `data/real_world_test/ice_cream/pexels-rdne-6412948.jpg` | ice cream | pizza | 69.8% |
| `data/real_world_test/idli/pexels-jack-baghel-2199968-20422128.jpg` | idli | sushi | 37.2% |
| `data/real_world_test/idli/pexels-lunarly-83082820-17869141.jpg` | idli | dhokla | 37.3% |
| `data/real_world_test/idli/pexels-mahesh-kumar-ganesan-2148232589-35514447 (1).jpg` | idli | chicken curry | 88.4% |
| `data/real_world_test/idli/pexels-mouriful-35580913.jpg` | idli | chapati | 39.3% |
| `data/real_world_test/idli/pexels-ryshy-s-2149956588-35539315.jpg` | idli | pav bhaji | 89.9% |
| `data/real_world_test/idli/pexels-sarthak-2782336-4331488.jpg` | idli | chai | 45.0% |
| `data/real_world_test/idli/pexels-sarthak-2782336-4331491 (1).jpg` | idli | kulfi | 77.6% |
| `data/real_world_test/kaathi_rolls/pexels-jitin-arora-2155275110-33709307.jpg` | kaathi rolls | burger | 25.2% |
| `data/real_world_test/kaathi_rolls/pexels-magda-ehlers-pexels-12758949.jpg` | kaathi rolls | samosa | 21.6% |
| `data/real_world_test/kaathi_rolls/pexels-saveurssecretes-12392905.jpg` | kaathi rolls | masala dosa | 50.3% |
| `data/real_world_test/kaathi_rolls/pexels-saveurssecretes-7936011.jpg` | kaathi rolls | samosa | 39.9% |
| `data/real_world_test/kadai_paneer/pexels-grayomm-31783383.jpg` | kadai paneer | chicken curry | 85.4% |
| `data/real_world_test/kadai_paneer/pexels-hemantlatawa-2580464.jpg` | kadai paneer | pakode | 69.9% |
| `data/real_world_test/kadai_paneer/pexels-kunal-lakhotia-781256899-29631459.jpg` | kadai paneer | pakode | 22.1% |
| `data/real_world_test/kadai_paneer/pexels-kunal-lakhotia-781256899-29631468.jpg` | kadai paneer | pakode | 42.5% |
| `data/real_world_test/kadai_paneer/pexels-shootsaga-33430556.jpg` | kadai paneer | sushi | 35.4% |
| `data/real_world_test/kulfi/pexels-jahratreza-39831473.jpg` | kulfi | chai | 31.8% |
| `data/real_world_test/kulfi/pexels-leonardo-aquino-246345118-27680854.jpg` | kulfi | ice cream | 96.9% |
| `data/real_world_test/kulfi/pexels-sahaab-38678811.jpg` | kulfi | Taco | 29.0% |
| `data/real_world_test/masala_dosa/pexels-jack-baghel-2199968-20422126.jpg` | masala dosa | dal makhani | 46.4% |
| `data/real_world_test/masala_dosa/pexels-kavya-pothabattula-388296612-28107046.jpg` | masala dosa | omelette | 46.2% |
| `data/real_world_test/masala_dosa/pexels-ryshy-s-2149956588-35539315.jpg` | masala dosa | pav bhaji | 89.9% |
| `data/real_world_test/momos/pexels-anya-dunes-2153945617-33670191.jpg` | momos | chapati | 31.7% |
| `data/real_world_test/momos/pexels-momo-king-3616480-5409010.jpg` | momos | masala dosa | 40.1% |
| `data/real_world_test/momos/pexels-prateek-lall-3279406-38520712.jpg` | momos | paani puri | 56.5% |
| `data/real_world_test/momos/pexels-shyamsrivastava-33384415.jpg` | momos | idli | 47.6% |
| `data/real_world_test/omelette/pexels-ali-dashti-506667798-28241698.jpg` | omelette | dhokla | 58.8% |
| `data/real_world_test/omelette/pexels-amine-1285347-12512039.jpg` | omelette | dal makhani | 58.7% |
| `data/real_world_test/omelette/pexels-camerasimagination-11929161.jpg` | omelette | dal makhani | 26.7% |
| `data/real_world_test/omelette/pexels-garley-gibson-447183984-15481458.jpg` | omelette | jalebi | 32.7% |
| `data/real_world_test/omelette/pexels-helin-gezer-903013644-25748211.jpg` | omelette | dal makhani | 14.7% |
| `data/real_world_test/paani_puri/pexels-fahim-shizan-2152194768-34356404.jpg` | paani puri | dal makhani | 52.0% |
| `data/real_world_test/paani_puri/pexels-jahratreza-40000023.jpg` | paani puri | samosa | 38.1% |
| `data/real_world_test/paani_puri/pexels-jahratreza-40000255.jpg` | paani puri | chicken curry | 65.1% |
| `data/real_world_test/paani_puri/pexels-jahratreza-40000376.jpg` | paani puri | sushi | 48.0% |
| `data/real_world_test/pakode/pexels-chanwalrus-958544.jpg` | pakode | chicken curry | 29.4% |
| `data/real_world_test/pakode/pexels-sabbythefreeelf-36619179.jpg` | pakode | chicken curry | 25.7% |
| `data/real_world_test/pakode/pexels-tamba09-37107128.jpg` | pakode | kadai paneer | 44.5% |
| `data/real_world_test/pav_bhaji/pexels-digitalbuggu-166654.jpg` | pav bhaji | dhokla | 32.9% |
| `data/real_world_test/pav_bhaji/pexels-sateesh-polkam-706210252-18496583.jpg` | pav bhaji | momos | 45.5% |
| `data/real_world_test/pizza/pexels-matreding-9685264.jpg` | pizza | butter naan | 69.4% |
| `data/real_world_test/samosa/pexels-supibee-5031949.jpg` | samosa | cheesecake | 19.1% |
| `data/real_world_test/sushi/pexels-nadin-sh-78971847-39528139.jpg` | sushi | ice cream | 62.4% |

## Real-world dataset quality
- Folder: `data/real_world_test`
- Valid images: **263**
- Invalid images: 0
- Unsupported files ignored: 0
- Unexpected entries ignored: 0
- Classes without images: 0
- Classes below 10 images: 31
- Exact duplicate groups within the set: 3
- Possible near-duplicate pairs within the set: 0
- Exact overlaps with training data sources: 0
- Possible near-duplicate overlaps with training data sources: 6
- Provenance file: `data/real_world_test/provenance.csv` (missing; 0 documented, 263 undocumented)

**Independence status: `overlap_detected`.** Some images are identical or possibly near-identical to images the model was built from. Those images are not independent.

### Warnings
- 31 class(es) have fewer than 10 images, so their per-class metrics are unreliable.
- 3 group(s) of exact duplicate images within the real-world set (15 extra copies; 1 with conflicting labels).
- 6 real-world image(s) are possible near-duplicates of training data sources; review manually.
- No provenance.csv found; the source of each image is undocumented.

### Duplicate-detection limits
Exact duplicates are found by SHA-256 of file bytes, so any re-encoding, resizing, cropping or metadata change hides a copy. Possible near-duplicates are found with a 64-bit difference hash (Hamming distance <= 3); this can miss crops, rotations, mirrored or heavily edited copies and can occasionally match different but similar-looking photos, so matches need manual review.

### Possible near-duplicate overlaps (review manually)
- `data/real_world_test/Donut/pexels-tijana-drndarski-449691-3338681.jpg` ~ processed/train `data/processed/train/Donut/000386.jpg` (distance 0)
- `data/real_world_test/Hot Dog/pexels-kamila-bairam-80608214-12123657.jpg` ~ processed/validation `data/processed/validation/Hot Dog/000006.jpg` (distance 0), processed/validation `data/processed/validation/Hot Dog/000160.jpg` (distance 0)
- `data/real_world_test/Hot Dog/pexels-polina-tankilevitch-4518656.jpg` ~ processed/train `data/processed/train/Hot Dog/000501.jpg` (distance 2)
- `data/real_world_test/Sandwich/pexels-karola-g2-6467.jpg` ~ processed/train `data/processed/train/Sandwich/000568.jpg` (distance 0)
- `data/real_world_test/masala_dosa/pexels-jack-baghel-2199968-20422126.jpg` ~ processed/train `data/processed/train/Crispy Chicken/000432.jpg` (distance 2)
- `data/real_world_test/momos/pexels-bishop-tamrakar-2246201-3926123.jpg` ~ processed/train `data/processed/train/momos/000219.jpg` (distance 1)

### Exact duplicates within the set
- `data/real_world_test/apple_pie/pexels-hilalbulbul-12782728 (1).jpg`, `data/real_world_test/apple_pie/pexels-hilalbulbul-12782728.jpg`
- `data/real_world_test/idli/pexels-ryshy-s-2149956588-35539315.jpg`, `data/real_world_test/masala_dosa/pexels-ryshy-s-2149956588-35539315.jpg` **(conflicting labels)**
- `data/real_world_test/sushi/pexels-valeriya-28291554 (1).jpg`, `data/real_world_test/sushi/pexels-valeriya-28291554 (10).jpg`, `data/real_world_test/sushi/pexels-valeriya-28291554 (11).jpg`, `data/real_world_test/sushi/pexels-valeriya-28291554 (12).jpg`, `data/real_world_test/sushi/pexels-valeriya-28291554 (13).jpg`, `data/real_world_test/sushi/pexels-valeriya-28291554 (2).jpg`, `data/real_world_test/sushi/pexels-valeriya-28291554 (3).jpg`, `data/real_world_test/sushi/pexels-valeriya-28291554 (4).jpg`, `data/real_world_test/sushi/pexels-valeriya-28291554 (5).jpg`, `data/real_world_test/sushi/pexels-valeriya-28291554 (6).jpg`, `data/real_world_test/sushi/pexels-valeriya-28291554 (7).jpg`, `data/real_world_test/sushi/pexels-valeriya-28291554 (8).jpg`, `data/real_world_test/sushi/pexels-valeriya-28291554 (9).jpg`, `data/real_world_test/sushi/pexels-valeriya-28291554.jpg`

### Images per class
| Class | Images |
|---|---|
| Baked Potato | 5 |
| Crispy Chicken | 6 |
| Donut | 9 |
| Fries | 9 |
| Hot Dog | 10 |
| Sandwich | 8 |
| Taco | 7 |
| Taquito | 9 |
| apple pie | 6 |
| burger | 7 |
| butter naan | 6 |
| chai | 6 |
| chapati | 5 |
| cheesecake | 7 |
| chicken curry | 7 |
| chole bhature | 7 |
| dal makhani | 7 |
| dhokla | 7 |
| fried rice | 7 |
| ice cream | 8 |
| idli | 11 |
| jalebi | 9 |
| kaathi rolls | 7 |
| kadai paneer | 8 |
| kulfi | 5 |
| masala dosa | 8 |
| momos | 9 |
| omelette | 7 |
| paani puri | 9 |
| pakode | 9 |
| pav bhaji | 8 |
| pizza | 8 |
| samosa | 7 |
| sushi | 15 |

## Reproduce
```
python -m evaluation.evaluate_real_world 
```
