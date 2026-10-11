# FoodCal AI real-world evaluation

Separate from the offline test-set report (`models/efficientnet_b0/evaluation_report.json`).

- Generated: 2026-10-11T05:25:27+00:00
- Status: **ok**
- Checkpoint: `models/efficientnet_b0_v2/food_efficientnet_b0_v2_best.pth` (epoch 21, SHA-256 `e34b83ee0b1e078c...`)
- Image manifest: 263 images, SHA-256 `b4ed5ce0a4807567...`
- Dataset: `data/real_world_test`
- Device: cuda
- Preprocessing: EXIF orientation applied, converted to RGB, resized to 224x224 with LANCZOS (no cropping, aspect ratio not preserved), then Resize((224, 224)), ToTensor and ImageNet normalization. Mirrors training/prepare_dataset.py + training/evaluate_efficientnet.py.
- Valid images: 263; invalid images: 0

## Overall metrics (all valid images)
| Metric | Value |
|---|---|
| Images evaluated | 263 |
| Correct | 177 |
| Incorrect | 86 |
| Accuracy | 67.3% |
| Macro precision (over 34 classes with images) | 68.2% |
| Macro recall (over 34 classes with images) | 64.6% |
| Macro F1 (over 34 classes with images) | 63.1% |
| Weighted F1 | 65.4% |

## Metrics excluding images that overlap training data
| Metric | Value |
|---|---|
| Images evaluated | 257 |
| Correct | 171 |
| Incorrect | 86 |
| Accuracy | 66.5% |
| Macro precision (over 34 classes with images) | 67.7% |
| Macro recall (over 34 classes with images) | 64.3% |
| Macro F1 (over 34 classes with images) | 62.7% |
| Weighted F1 | 64.7% |

## Model confidence (not accuracy)
- Mean top-1 confidence: 59.9%
- Mean confidence when correct: 71.2%
- Mean confidence when incorrect: 36.5%

| Confidence range | Images | Mean confidence | Observed accuracy |
|---|---|---|---|
| 10%-20% | 19 | 16.3% | 21.1% |
| 20%-30% | 25 | 24.8% | 16.0% |
| 30%-40% | 37 | 34.7% | 35.1% |
| 40%-50% | 14 | 44.7% | 50.0% |
| 50%-60% | 28 | 56.1% | 60.7% |
| 60%-70% | 34 | 65.6% | 91.2% |
| 70%-80% | 25 | 74.8% | 92.0% |
| 80%-90% | 44 | 86.1% | 93.2% |
| 90%-100% | 37 | 93.2% | 100.0% |

## Per-class metrics
| Class | Images | Predicted | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Baked Potato | 5 | 1 | 100.0% | 20.0% | 33.3% |
| Crispy Chicken | 6 | 6 | 83.3% | 83.3% | 83.3% |
| Donut | 9 | 10 | 80.0% | 88.9% | 84.2% |
| Fries | 9 | 13 | 69.2% | 100.0% | 81.8% |
| Hot Dog | 10 | 14 | 71.4% | 100.0% | 83.3% |
| Sandwich | 8 | 8 | 75.0% | 75.0% | 75.0% |
| Taco | 7 | 5 | 80.0% | 57.1% | 66.7% |
| Taquito | 9 | 4 | 50.0% | 22.2% | 30.8% |
| apple pie | 6 | 3 | 66.7% | 33.3% | 44.4% |
| burger | 7 | 8 | 75.0% | 85.7% | 80.0% |
| butter naan | 6 | 3 | 100.0% | 50.0% | 66.7% |
| chai | 6 | 10 | 60.0% | 100.0% | 75.0% |
| chapati | 5 | 10 | 40.0% | 80.0% | 53.3% |
| cheesecake | 7 | 3 | 100.0% | 42.9% | 60.0% |
| chicken curry | 7 | 6 | 16.7% | 14.3% | 15.4% |
| chole bhature | 7 | 6 | 66.7% | 57.1% | 61.5% |
| dal makhani | 7 | 7 | 71.4% | 71.4% | 71.4% |
| dhokla | 7 | 4 | 75.0% | 42.9% | 54.5% |
| fried rice | 7 | 3 | 33.3% | 14.3% | 20.0% |
| ice cream | 8 | 9 | 55.6% | 62.5% | 58.8% |
| idli | 11 | 6 | 100.0% | 54.5% | 70.6% |
| jalebi | 9 | 10 | 90.0% | 100.0% | 94.7% |
| kaathi rolls | 7 | 5 | 40.0% | 28.6% | 33.3% |
| kadai paneer | 8 | 8 | 50.0% | 50.0% | 50.0% |
| kulfi | 5 | 6 | 33.3% | 40.0% | 36.4% |
| masala dosa | 8 | 11 | 63.6% | 87.5% | 73.7% |
| momos | 9 | 14 | 57.1% | 88.9% | 69.6% |
| omelette | 7 | 5 | 60.0% | 42.9% | 50.0% |
| paani puri | 9 | 6 | 100.0% | 66.7% | 80.0% |
| pakode | 9 | 12 | 66.7% | 88.9% | 76.2% |
| pav bhaji | 8 | 8 | 75.0% | 75.0% | 75.0% |
| pizza | 8 | 9 | 88.9% | 100.0% | 94.1% |
| samosa | 7 | 11 | 45.5% | 71.4% | 55.6% |
| sushi | 15 | 19 | 78.9% | 100.0% | 88.2% |

## Confusion matrix
Rows are actual classes, columns are predicted classes. Only classes that appear as an actual or predicted label are shown.

| actual \ predicted | Baked Potato | Crispy Chicken | Donut | Fries | Hot Dog | Sandwich | Taco | Taquito | apple pie | burger | butter naan | chai | chapati | cheesecake | chicken curry | chole bhature | dal makhani | dhokla | fried rice | ice cream | idli | jalebi | kaathi rolls | kadai paneer | kulfi | masala dosa | momos | omelette | paani puri | pakode | pav bhaji | pizza | samosa | sushi |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Baked Potato | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| Crispy Chicken | 0 | 5 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Donut | 0 | 0 | 8 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Fries | 0 | 0 | 0 | 9 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Hot Dog | 0 | 0 | 0 | 0 | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Sandwich | 0 | 0 | 1 | 0 | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| Taco | 0 | 0 | 0 | 0 | 1 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Taquito | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 3 | 1 |
| apple pie | 0 | 0 | 0 | 1 | 0 | 2 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| burger | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| butter naan | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| chai | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| chapati | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| cheesecake | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 2 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| chicken curry | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| chole bhature | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| dal makhani | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 5 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| dhokla | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| fried rice | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| ice cream | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| idli | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| jalebi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 9 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| kaathi rolls | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| kadai paneer | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| kulfi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| masala dosa | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 7 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| momos | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| omelette | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 2 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 |
| paani puri | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 6 | 0 | 0 | 0 | 0 | 1 |
| pakode | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 8 | 0 | 0 | 0 | 0 |
| pav bhaji | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 6 | 1 | 0 | 0 |
| pizza | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 8 | 0 | 0 |
| samosa | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 5 | 0 |
| sushi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 15 |

## Incorrect predictions (86)
| Image | Actual | Predicted | Confidence |
|---|---|---|---|
| `data/real_world_test/Baked Potato/pexels-lorencastillo-9219092.jpg` | Baked Potato | jalebi | 17.2% |
| `data/real_world_test/Baked Potato/pexels-nicola-barts-7936637.jpg` | Baked Potato | samosa | 23.3% |
| `data/real_world_test/Baked Potato/pexels-polina-tankilevitch-4110468.jpg` | Baked Potato | kulfi | 74.3% |
| `data/real_world_test/Baked Potato/pexels-polina-tankilevitch-4110470.jpg` | Baked Potato | Hot Dog | 16.5% |
| `data/real_world_test/Crispy Chicken/pexels-andaru-firmansyah-2152289371-37279764.jpg` | Crispy Chicken | Fries | 37.9% |
| `data/real_world_test/Donut/pexels-perfect-lens-8358569.jpg` | Donut | Hot Dog | 37.8% |
| `data/real_world_test/Sandwich/pexels-andrea-prochilo-3062027-34421469.jpg` | Sandwich | omelette | 23.6% |
| `data/real_world_test/Sandwich/pexels-thongtran95-16436245.jpg` | Sandwich | Donut | 65.8% |
| `data/real_world_test/Taco/pexels-roman-odintsov-4958779.jpg` | Taco | kaathi rolls | 25.1% |
| `data/real_world_test/Taco/pexels-roman-odintsov-5837196.jpg` | Taco | Hot Dog | 24.8% |
| `data/real_world_test/Taco/pexels-styves-exantus-7162424-7041784.jpg` | Taco | kaathi rolls | 44.4% |
| `data/real_world_test/Taquito/pexels-cristian-rojas-8448323.jpg` | Taquito | Taco | 31.4% |
| `data/real_world_test/Taquito/pexels-fausto-hernandez-196511575-12326169.jpg` | Taquito | samosa | 37.2% |
| `data/real_world_test/Taquito/pexels-jonathan-reynaga-861774-17429044.jpg` | Taquito | Fries | 20.8% |
| `data/real_world_test/Taquito/pexels-jonathan-reynaga-861774-17429153.jpg` | Taquito | sushi | 13.9% |
| `data/real_world_test/Taquito/pexels-nano-erdozain-120534369-27905921.jpg` | Taquito | samosa | 31.3% |
| `data/real_world_test/Taquito/pexels-rachel-claire-5531155.jpg` | Taquito | momos | 31.4% |
| `data/real_world_test/Taquito/pexels-ruben-dao-cuentas-239930863-34303345.jpg` | Taquito | samosa | 51.3% |
| `data/real_world_test/apple_pie/pexels-christina-voinova-42707474-7368070.jpg` | apple pie | masala dosa | 28.8% |
| `data/real_world_test/apple_pie/pexels-hilalbulbul-12782728 (1).jpg` | apple pie | Sandwich | 25.2% |
| `data/real_world_test/apple_pie/pexels-hilalbulbul-12782728.jpg` | apple pie | Sandwich | 25.2% |
| `data/real_world_test/apple_pie/pexels-hilalbulbul-7552556.jpg` | apple pie | Fries | 17.3% |
| `data/real_world_test/burger/pexels-hacihuseyinerol-36007382.jpg` | burger | Hot Dog | 35.4% |
| `data/real_world_test/butter_naan/pexels-miltonphotography-35066811.jpg` | butter naan | chapati | 39.5% |
| `data/real_world_test/butter_naan/pexels-theshuttervision-15131306.jpg` | butter naan | chapati | 33.2% |
| `data/real_world_test/butter_naan/pexels-yarensphotoworks-30556657.jpg` | butter naan | chapati | 59.8% |
| `data/real_world_test/chapati/pexels-sincegameon-20045788.jpg` | chapati | chai | 65.2% |
| `data/real_world_test/cheesecake/pexels-marijana-blazaric-571842-13350861.jpg` | cheesecake | ice cream | 59.7% |
| `data/real_world_test/cheesecake/pexels-mlkbnl-25746353.jpg` | cheesecake | burger | 33.6% |
| `data/real_world_test/cheesecake/pexels-mlkbnl-29831239.jpg` | cheesecake | burger | 18.6% |
| `data/real_world_test/cheesecake/pexels-vivanpunk-34637999.jpg` | cheesecake | apple pie | 17.4% |
| `data/real_world_test/chicken_curry/pexels-anna-pou-9345647.jpg` | chicken curry | chole bhature | 24.4% |
| `data/real_world_test/chicken_curry/pexels-drmkhawarnazir-38003495.jpg` | chicken curry | kadai paneer | 84.6% |
| `data/real_world_test/chicken_curry/pexels-iamabdullahsheik-9792458.jpg` | chicken curry | dal makhani | 39.4% |
| `data/real_world_test/chicken_curry/pexels-jkreat0r-32986463.jpg` | chicken curry | dal makhani | 13.9% |
| `data/real_world_test/chicken_curry/pexels-kunal-lakhotia-781256899-29684985.jpg` | chicken curry | kadai paneer | 59.2% |
| `data/real_world_test/chicken_curry/pexels-pinaki-panda-3019803-33643313.jpg` | chicken curry | kadai paneer | 57.5% |
| `data/real_world_test/chole_bhature/pexels-inaam-ashiq-2155239180-35128164.jpg` | chole bhature | chicken curry | 21.4% |
| `data/real_world_test/chole_bhature/pexels-jack-baghel-2199968-20446423.jpg` | chole bhature | chapati | 17.5% |
| `data/real_world_test/chole_bhature/pexels-llana-79788650-15717898.jpg` | chole bhature | masala dosa | 41.1% |
| `data/real_world_test/dal_makhani/pexels-kunal-lakhotia-781256899-28674708.jpg` | dal makhani | fried rice | 29.3% |
| `data/real_world_test/dal_makhani/pexels-kunal-lakhotia-781256899-28674710.jpg` | dal makhani | chicken curry | 22.6% |
| `data/real_world_test/dhokla/pexels-aayushrawat-8414675.jpg` | dhokla | omelette | 17.8% |
| `data/real_world_test/dhokla/pexels-ali-dashti-506667798-26341215.jpg` | dhokla | Donut | 28.4% |
| `data/real_world_test/dhokla/pexels-marcelochagas-3324431.jpg` | dhokla | kulfi | 87.4% |
| `data/real_world_test/dhokla/pexels-shootsaga-38734222.jpg` | dhokla | Fries | 43.2% |
| `data/real_world_test/fried_rice/pexels-4louette-11129150.jpg` | fried rice | kaathi rolls | 13.8% |
| `data/real_world_test/fried_rice/pexels-anntarazevich-6937425.jpg` | fried rice | samosa | 34.4% |
| `data/real_world_test/fried_rice/pexels-anntarazevich-6937439.jpg` | fried rice | dhokla | 69.4% |
| `data/real_world_test/fried_rice/pexels-anntarazevich-6937447.jpg` | fried rice | ice cream | 43.6% |
| `data/real_world_test/fried_rice/pexels-anntarazevich-6937451.jpg` | fried rice | chai | 51.3% |
| `data/real_world_test/fried_rice/pexels-anntarazevich-6937457.jpg` | fried rice | sushi | 39.2% |
| `data/real_world_test/ice_cream/pexels-bank-12843687.jpg` | ice cream | momos | 35.7% |
| `data/real_world_test/ice_cream/pexels-glebkrs-15155623.jpg` | ice cream | chai | 59.1% |
| `data/real_world_test/ice_cream/pexels-rdne-6412948.jpg` | ice cream | chapati | 21.6% |
| `data/real_world_test/idli/pexels-mahesh-kumar-ganesan-2148232589-35514447 (1).jpg` | idli | chicken curry | 32.2% |
| `data/real_world_test/idli/pexels-mouriful-35580913.jpg` | idli | chapati | 16.1% |
| `data/real_world_test/idli/pexels-ryshy-s-2149956588-35539315.jpg` | idli | pav bhaji | 56.1% |
| `data/real_world_test/idli/pexels-sarthak-2782336-4331488.jpg` | idli | kulfi | 32.3% |
| `data/real_world_test/idli/pexels-sarthak-2782336-4331491 (1).jpg` | idli | kulfi | 34.4% |
| `data/real_world_test/kaathi_rolls/pexels-albert-kayenda-2161871276-39994554.jpg` | kaathi rolls | momos | 25.3% |
| `data/real_world_test/kaathi_rolls/pexels-jitin-arora-2155275110-33709307.jpg` | kaathi rolls | sushi | 37.9% |
| `data/real_world_test/kaathi_rolls/pexels-magda-ehlers-pexels-12758949.jpg` | kaathi rolls | samosa | 89.4% |
| `data/real_world_test/kaathi_rolls/pexels-saveurssecretes-12392905.jpg` | kaathi rolls | Taquito | 30.1% |
| `data/real_world_test/kaathi_rolls/pexels-saveurssecretes-7936011.jpg` | kaathi rolls | chai | 34.2% |
| `data/real_world_test/kadai_paneer/pexels-hemantlatawa-2580464.jpg` | kadai paneer | pakode | 58.9% |
| `data/real_world_test/kadai_paneer/pexels-kunal-lakhotia-781256899-29631459.jpg` | kadai paneer | Crispy Chicken | 32.8% |
| `data/real_world_test/kadai_paneer/pexels-kunal-lakhotia-781256899-29631468.jpg` | kadai paneer | pakode | 27.6% |
| `data/real_world_test/kadai_paneer/pexels-shootsaga-33430556.jpg` | kadai paneer | pakode | 24.8% |
| `data/real_world_test/kulfi/pexels-jahratreza-39831473.jpg` | kulfi | ice cream | 74.2% |
| `data/real_world_test/kulfi/pexels-leonardo-aquino-246345118-27680854.jpg` | kulfi | ice cream | 49.8% |
| `data/real_world_test/kulfi/pexels-sahaab-38678811.jpg` | kulfi | Taquito | 16.3% |
| `data/real_world_test/masala_dosa/pexels-ryshy-s-2149956588-35539315.jpg` | masala dosa | pav bhaji | 56.1% |
| `data/real_world_test/momos/pexels-prateek-lall-3279406-38520712.jpg` | momos | chicken curry | 16.2% |
| `data/real_world_test/omelette/pexels-ali-dashti-506667798-28241698.jpg` | omelette | masala dosa | 26.2% |
| `data/real_world_test/omelette/pexels-amine-1285347-12512039.jpg` | omelette | fried rice | 30.1% |
| `data/real_world_test/omelette/pexels-camerasimagination-11929161.jpg` | omelette | kadai paneer | 26.2% |
| `data/real_world_test/omelette/pexels-garley-gibson-447183984-15481458.jpg` | omelette | masala dosa | 41.9% |
| `data/real_world_test/paani_puri/pexels-fahim-shizan-2152194768-34356404.jpg` | paani puri | momos | 58.4% |
| `data/real_world_test/paani_puri/pexels-jahratreza-40000255.jpg` | paani puri | chicken curry | 16.5% |
| `data/real_world_test/paani_puri/pexels-jahratreza-40000376.jpg` | paani puri | sushi | 34.7% |
| `data/real_world_test/pakode/pexels-miltonphotography-35066805.jpg` | pakode | momos | 20.5% |
| `data/real_world_test/pav_bhaji/pexels-digitalbuggu-166654.jpg` | pav bhaji | momos | 42.5% |
| `data/real_world_test/pav_bhaji/pexels-sateesh-polkam-706210252-18496583.jpg` | pav bhaji | pizza | 13.6% |
| `data/real_world_test/samosa/pexels-roman-saienko-1867764487-38043951.jpg` | samosa | chole bhature | 24.5% |
| `data/real_world_test/samosa/pexels-supibee-5031949.jpg` | samosa | pakode | 35.7% |

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
python -m evaluation.evaluate_real_world --checkpoint models/efficientnet_b0_v2/food_efficientnet_b0_v2_best.pth --output-dir reports/real_world_v2
```
