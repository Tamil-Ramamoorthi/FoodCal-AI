# Real-world comparison

Generated 2026-10-11T05:26:16+00:00.
- Old: `reports/real_world/real_world_evaluation.json`
- New: `reports/real_world_v2/real_world_evaluation.json`

- Old checkpoint: `models/efficientnet_b0/food_efficientnet_b0_best.pth` (SHA-256 `39363ab4f70d68fb...`)
- New checkpoint: `models/efficientnet_b0_v2/food_efficientnet_b0_v2_best.pth` (SHA-256 `e34b83ee0b1e078c...`)
- Same 263 images for both (manifest SHA-256 `b4ed5ce0a4807567...`); identical class mapping and preprocessing.

## Overall
| Metric | Old | New | Change |
|---|---|---|---|
| Accuracy | 54.8% | 67.3% | +12.5 pts |
| Macro precision | 53.0% | 68.2% | +15.2 pts |
| Macro recall | 52.0% | 64.6% | +12.6 pts |
| Macro F1 | 48.5% | 63.1% | +14.7 pts |
| Weighted F1 | 51.5% | 65.4% | +13.9 pts |

Correct: old 144 / new 177 of 263. Macro averages cover the 34 classes with images.
Paired exact McNemar test: old right & new wrong 9, old wrong & new right 42, p = 3.4e-06.
Bootstrap 95% CI (2000 resamples): accuracy difference +7.6 to +17.9 pts; macro-F1 difference +9.0 to +20.2 pts.

## Limitations of this sample
- 263 images in 34 of 34 classes; classes without images are not measured and are excluded from macro averages.
- 31 class(es) have fewer than 10 images; their per-class metrics are unreliable.
- 0 exact and 6 possible near-duplicate overlaps with training sources; those images are not independent.
- Independence status: `overlap_detected`. Some images are identical or possibly near-identical to images the model was built from. Those images are not independent.
- Confidence intervals below are wide for small samples; a difference inside them is not evidence that either model is better.

## Per-class changes (classes with images)
| Class | Images | Old P / R / F1 | New P / R / F1 | F1 change |
|---|---|---|---|---|
| fried rice | 7 | 42.9% / 42.9% / 42.9% | 33.3% / 14.3% / 20.0% | -22.9 |
| kaathi rolls | 7 | 60.0% / 42.9% / 50.0% | 40.0% / 28.6% / 33.3% | -16.7 |
| Fries | 9 | 88.9% / 88.9% / 88.9% | 69.2% / 100.0% / 81.8% | -7.1 |
| Hot Dog | 10 | 90.0% / 90.0% / 90.0% | 71.4% / 100.0% / 83.3% | -6.7 |
| ice cream | 8 | 62.5% / 62.5% / 62.5% | 55.6% / 62.5% / 58.8% | -3.7 |
| kulfi | 5 | 40.0% / 40.0% / 40.0% | 33.3% / 40.0% / 36.4% | -3.6 |
| Donut | 9 | 100.0% / 77.8% / 87.5% | 80.0% / 88.9% / 84.2% | -3.3 |
| chai | 6 | 60.0% / 100.0% / 75.0% | 60.0% / 100.0% / 75.0% | +0.0 |
| samosa | 7 | 40.0% / 85.7% / 54.5% | 45.5% / 71.4% / 55.6% | +1.0 |
| chicken curry | 7 | 14.3% / 14.3% / 14.3% | 16.7% / 14.3% / 15.4% | +1.1 |
| sushi | 15 | 73.7% / 93.3% / 82.4% | 78.9% / 100.0% / 88.2% | +5.9 |
| jalebi | 9 | 75.0% / 100.0% / 85.7% | 90.0% / 100.0% / 94.7% | +9.0 |
| burger | 7 | 60.0% / 85.7% / 70.6% | 75.0% / 85.7% / 80.0% | +9.4 |
| pakode | 9 | 66.7% / 66.7% / 66.7% | 66.7% / 88.9% / 76.2% | +9.5 |
| masala dosa | 8 | 62.5% / 62.5% / 62.5% | 63.6% / 87.5% / 73.7% | +11.2 |
| pav bhaji | 8 | 54.5% / 75.0% / 63.2% | 75.0% / 75.0% / 75.0% | +11.8 |
| dhokla | 7 | 33.3% / 57.1% / 42.1% | 75.0% / 42.9% / 54.5% | +12.4 |
| kadai paneer | 8 | 37.5% / 37.5% / 37.5% | 50.0% / 50.0% / 50.0% | +12.5 |
| momos | 9 | 55.6% / 55.6% / 55.6% | 57.1% / 88.9% / 69.6% | +14.0 |
| omelette | 7 | 33.3% / 28.6% / 30.8% | 60.0% / 42.9% / 50.0% | +19.2 |
| pizza | 8 | 63.6% / 87.5% / 73.7% | 88.9% / 100.0% / 94.1% | +20.4 |
| paani puri | 9 | 62.5% / 55.6% / 58.8% | 100.0% / 66.7% / 80.0% | +21.2 |
| apple pie | 6 | 33.3% / 16.7% / 22.2% | 66.7% / 33.3% / 44.4% | +22.2 |
| butter naan | 6 | 66.7% / 33.3% / 44.4% | 100.0% / 50.0% / 66.7% | +22.2 |
| idli | 11 | 66.7% / 36.4% / 47.1% | 100.0% / 54.5% / 70.6% | +23.5 |
| chapati | 5 | 18.8% / 60.0% / 28.6% | 40.0% / 80.0% / 53.3% | +24.8 |
| dal makhani | 7 | 31.6% / 85.7% / 46.2% | 71.4% / 71.4% / 71.4% | +25.3 |
| Taco | 7 | 66.7% / 28.6% / 40.0% | 80.0% / 57.1% / 66.7% | +26.7 |
| cheesecake | 7 | 40.0% / 28.6% / 33.3% | 100.0% / 42.9% / 60.0% | +26.7 |
| Taquito | 9 | n/a / 0.0% / 0.0% | 50.0% / 22.2% / 30.8% | +30.8 |
| Baked Potato | 5 | n/a / 0.0% / 0.0% | 100.0% / 20.0% / 33.3% | +33.3 |
| Sandwich | 8 | 100.0% / 12.5% / 22.2% | 75.0% / 75.0% / 75.0% | +52.8 |
| Crispy Chicken | 6 | 100.0% / 16.7% / 28.6% | 83.3% / 83.3% / 83.3% | +54.8 |
| chole bhature | 7 | n/a / 0.0% / 0.0% | 66.7% / 57.1% / 61.5% | +61.5 |

## Most frequent confusions (either model)
| Actual → predicted | Old | New |
|---|---|---|
| kadai paneer → pakode | 3 | 3 |
| butter naan → chapati | 2 | 3 |
| chicken curry → dal makhani | 3 | 2 |
| chicken curry → kadai paneer | 2 | 3 |
| Taquito → samosa | 1 | 3 |
| Taquito → dhokla | 3 | 0 |
| apple pie → chapati | 3 | 0 |
| chole bhature → chapati | 2 | 1 |
| idli → kulfi | 1 | 2 |
| kaathi rolls → samosa | 2 | 1 |
| kulfi → ice cream | 1 | 2 |
| omelette → dal makhani | 3 | 0 |
| Baked Potato → apple pie | 2 | 0 |
| Baked Potato → jalebi | 1 | 1 |
| Baked Potato → kulfi | 1 | 1 |

## Images whose prediction changed (105)
| Image | Actual | Old | New |
|---|---|---|---|
| `data/real_world_test/Baked Potato/pexels-nobleseed-34495400.jpg` | Baked Potato | apple pie | Baked Potato |
| `data/real_world_test/Baked Potato/pexels-polina-tankilevitch-4110470.jpg` | Baked Potato | apple pie | Hot Dog |
| `data/real_world_test/Crispy Chicken/pexels-andaru-firmansyah-2152289371-37279764.jpg` | Crispy Chicken | pizza | Fries |
| `data/real_world_test/Crispy Chicken/pexels-chicken-mackay-3652946-5474676.jpg` | Crispy Chicken | burger | Crispy Chicken |
| `data/real_world_test/Crispy Chicken/pexels-foturecreative-33101857.jpg` | Crispy Chicken | chapati | Crispy Chicken |
| `data/real_world_test/Crispy Chicken/pexels-mohammad-miladi-134457715-10206125.jpg` | Crispy Chicken | omelette | Crispy Chicken |
| `data/real_world_test/Crispy Chicken/pexels-qamar-rehman-94539242-11502374.jpg` | Crispy Chicken | chai | Crispy Chicken |
| `data/real_world_test/Donut/pexels-perfect-lens-8358569.jpg` | Donut | pizza | Hot Dog |
| `data/real_world_test/Donut/pexels-tijana-drndarski-449691-3338681.jpg` | Donut | burger | Donut |
| `data/real_world_test/Fries/pexels-jahratreza-39034206.jpg` | Fries | samosa | Fries |
| `data/real_world_test/Hot Dog/pexels-alleksana-4113501.jpg` | Hot Dog | kaathi rolls | Hot Dog |
| `data/real_world_test/Sandwich/pexels-andrea-prochilo-3062027-34421469.jpg` | Sandwich | cheesecake | omelette |
| `data/real_world_test/Sandwich/pexels-karola-g2-6467.jpg` | Sandwich | idli | Sandwich |
| `data/real_world_test/Sandwich/pexels-nano-erdozain-120534369-19202817.jpg` | Sandwich | dal makhani | Sandwich |
| `data/real_world_test/Sandwich/pexels-nano-erdozain-120534369-19202829.jpg` | Sandwich | fried rice | Sandwich |
| `data/real_world_test/Sandwich/pexels-thongtran95-16436245.jpg` | Sandwich | sushi | Donut |
| `data/real_world_test/Sandwich/pexels-tochtliyeung1996-15362507.jpg` | Sandwich | omelette | Sandwich |
| `data/real_world_test/Sandwich/pexels-tolgaaslanturk-11256670.jpg` | Sandwich | samosa | Sandwich |
| `data/real_world_test/Taco/pexels-roman-odintsov-4958526.jpg` | Taco | dal makhani | Taco |
| `data/real_world_test/Taco/pexels-roman-odintsov-4958779.jpg` | Taco | fried rice | kaathi rolls |
| `data/real_world_test/Taco/pexels-roman-odintsov-5837188.jpg` | Taco | ice cream | Taco |
| `data/real_world_test/Taco/pexels-roman-odintsov-5837196.jpg` | Taco | burger | Hot Dog |
| `data/real_world_test/Taco/pexels-styves-exantus-7162424-7041784.jpg` | Taco | dhokla | kaathi rolls |
| `data/real_world_test/Taquito/pexels-anthony-rahayel-125801377-32592710.jpg` | Taquito | kulfi | Taquito |
| `data/real_world_test/Taquito/pexels-cristian-rojas-8448323.jpg` | Taquito | masala dosa | Taco |
| `data/real_world_test/Taquito/pexels-fausto-hernandez-196511575-12326169.jpg` | Taquito | momos | samosa |
| `data/real_world_test/Taquito/pexels-jonathan-reynaga-861774-17429044.jpg` | Taquito | dhokla | Fries |
| `data/real_world_test/Taquito/pexels-jonathan-reynaga-861774-17429144.jpg` | Taquito | dhokla | Taquito |
| `data/real_world_test/Taquito/pexels-jonathan-reynaga-861774-17429153.jpg` | Taquito | dal makhani | sushi |
| `data/real_world_test/Taquito/pexels-rachel-claire-5531155.jpg` | Taquito | dhokla | momos |
| `data/real_world_test/Taquito/pexels-ruben-dao-cuentas-239930863-34303345.jpg` | Taquito | omelette | samosa |
| `data/real_world_test/apple_pie/pexels-christina-voinova-42707474-7368070.jpg` | apple pie | cheesecake | masala dosa |
| `data/real_world_test/apple_pie/pexels-clement-bruneau-2159570516-36340982.jpg` | apple pie | momos | apple pie |
| `data/real_world_test/apple_pie/pexels-hilalbulbul-12782728 (1).jpg` | apple pie | chapati | Sandwich |
| `data/real_world_test/apple_pie/pexels-hilalbulbul-12782728.jpg` | apple pie | chapati | Sandwich |
| `data/real_world_test/apple_pie/pexels-hilalbulbul-7552556.jpg` | apple pie | chapati | Fries |
| `data/real_world_test/butter_naan/pexels-gagankaur-16851842.jpg` | butter naan | fried rice | butter naan |
| `data/real_world_test/butter_naan/pexels-miltonphotography-35066811.jpg` | butter naan | dhokla | chapati |
| `data/real_world_test/chapati/pexels-usman-yousaf-708951-5589943.jpg` | chapati | kaathi rolls | chapati |
| `data/real_world_test/cheesecake/pexels-marijana-blazaric-571842-13350861.jpg` | cheesecake | pav bhaji | ice cream |
| `data/real_world_test/cheesecake/pexels-mlkbnl-25746353.jpg` | cheesecake | sushi | burger |
| `data/real_world_test/cheesecake/pexels-mlkbnl-25746355.jpg` | cheesecake | kadai paneer | cheesecake |
| `data/real_world_test/cheesecake/pexels-mlkbnl-25746365.jpg` | cheesecake | pizza | cheesecake |
| `data/real_world_test/cheesecake/pexels-mlkbnl-29831239.jpg` | cheesecake | cheesecake | burger |
| `data/real_world_test/cheesecake/pexels-vivanpunk-34637999.jpg` | cheesecake | dal makhani | apple pie |
| `data/real_world_test/chicken_curry/pexels-anna-pou-9345647.jpg` | chicken curry | chapati | chole bhature |
| `data/real_world_test/chicken_curry/pexels-pinaki-panda-3019803-33643313.jpg` | chicken curry | dal makhani | kadai paneer |
| `data/real_world_test/chole_bhature/pexels-adhwaith-chandran-214377112-15995519.jpg` | chole bhature | samosa | chole bhature |
| `data/real_world_test/chole_bhature/pexels-drmkhawarnazir-9287035.jpg` | chole bhature | dal makhani | chole bhature |
| `data/real_world_test/chole_bhature/pexels-inaam-ashiq-2155239180-35128164.jpg` | chole bhature | Fries | chicken curry |
| `data/real_world_test/chole_bhature/pexels-jack-baghel-2199968-20446423.jpg` | chole bhature | pav bhaji | chapati |
| `data/real_world_test/chole_bhature/pexels-llana-79788650-15717898.jpg` | chole bhature | chapati | masala dosa |
| `data/real_world_test/chole_bhature/pexels-sabyasachi-das-45419624-29066704.jpg` | chole bhature | chapati | chole bhature |
| `data/real_world_test/chole_bhature/pexels-yash-acharya-214379065-11818239.jpg` | chole bhature | pav bhaji | chole bhature |
| `data/real_world_test/dal_makhani/pexels-kunal-lakhotia-781256899-28674708.jpg` | dal makhani | chicken curry | fried rice |
| `data/real_world_test/dal_makhani/pexels-kunal-lakhotia-781256899-28674710.jpg` | dal makhani | dal makhani | chicken curry |
| `data/real_world_test/dhokla/pexels-aayushrawat-8414675.jpg` | dhokla | paani puri | omelette |
| `data/real_world_test/dhokla/pexels-ali-dashti-506667798-26341215.jpg` | dhokla | paani puri | Donut |
| `data/real_world_test/dhokla/pexels-marcelochagas-3324431.jpg` | dhokla | samosa | kulfi |
| `data/real_world_test/dhokla/pexels-shootsaga-38734222.jpg` | dhokla | dhokla | Fries |
| `data/real_world_test/fried_rice/pexels-4louette-11129150.jpg` | fried rice | kadai paneer | kaathi rolls |
| `data/real_world_test/fried_rice/pexels-anntarazevich-6937425.jpg` | fried rice | fried rice | samosa |
| `data/real_world_test/fried_rice/pexels-anntarazevich-6937439.jpg` | fried rice | fried rice | dhokla |
| `data/real_world_test/fried_rice/pexels-anntarazevich-6937447.jpg` | fried rice | chapati | ice cream |
| `data/real_world_test/fried_rice/pexels-anntarazevich-6937451.jpg` | fried rice | jalebi | chai |
| `data/real_world_test/fried_rice/pexels-anntarazevich-6937457.jpg` | fried rice | chapati | sushi |
| `data/real_world_test/ice_cream/pexels-bank-12843687.jpg` | ice cream | fried rice | momos |
| `data/real_world_test/ice_cream/pexels-ds-stories-9227980.jpg` | ice cream | momos | ice cream |
| `data/real_world_test/ice_cream/pexels-glebkrs-15155623.jpg` | ice cream | ice cream | chai |
| `data/real_world_test/ice_cream/pexels-rdne-6412948.jpg` | ice cream | pizza | chapati |
| `data/real_world_test/idli/pexels-jack-baghel-2199968-20422128.jpg` | idli | sushi | idli |
| `data/real_world_test/idli/pexels-lunarly-83082820-17869141.jpg` | idli | dhokla | idli |
| `data/real_world_test/idli/pexels-sarthak-2782336-4331488.jpg` | idli | chai | kulfi |
| `data/real_world_test/kaathi_rolls/pexels-albert-kayenda-2161871276-39994554.jpg` | kaathi rolls | kaathi rolls | momos |
| `data/real_world_test/kaathi_rolls/pexels-jitin-arora-2155275110-33709307.jpg` | kaathi rolls | burger | sushi |
| `data/real_world_test/kaathi_rolls/pexels-saveurssecretes-12392905.jpg` | kaathi rolls | masala dosa | Taquito |
| `data/real_world_test/kaathi_rolls/pexels-saveurssecretes-7936011.jpg` | kaathi rolls | samosa | chai |
| `data/real_world_test/kadai_paneer/pexels-grayomm-31783383.jpg` | kadai paneer | chicken curry | kadai paneer |
| `data/real_world_test/kadai_paneer/pexels-kunal-lakhotia-781256899-29631459.jpg` | kadai paneer | pakode | Crispy Chicken |
| `data/real_world_test/kadai_paneer/pexels-shootsaga-33430556.jpg` | kadai paneer | sushi | pakode |
| `data/real_world_test/kulfi/pexels-jahratreza-39831473.jpg` | kulfi | chai | ice cream |
| `data/real_world_test/kulfi/pexels-sahaab-38678811.jpg` | kulfi | Taco | Taquito |
| `data/real_world_test/masala_dosa/pexels-jack-baghel-2199968-20422126.jpg` | masala dosa | dal makhani | masala dosa |
| `data/real_world_test/masala_dosa/pexels-kavya-pothabattula-388296612-28107046.jpg` | masala dosa | omelette | masala dosa |
| `data/real_world_test/momos/pexels-anya-dunes-2153945617-33670191.jpg` | momos | chapati | momos |
| `data/real_world_test/momos/pexels-momo-king-3616480-5409010.jpg` | momos | masala dosa | momos |
| `data/real_world_test/momos/pexels-prateek-lall-3279406-38520712.jpg` | momos | paani puri | chicken curry |
| `data/real_world_test/momos/pexels-shyamsrivastava-33384415.jpg` | momos | idli | momos |
| `data/real_world_test/omelette/pexels-ali-dashti-506667798-28241698.jpg` | omelette | dhokla | masala dosa |
| `data/real_world_test/omelette/pexels-amine-1285347-12512039.jpg` | omelette | dal makhani | fried rice |
| `data/real_world_test/omelette/pexels-camerasimagination-11929161.jpg` | omelette | dal makhani | kadai paneer |
| `data/real_world_test/omelette/pexels-garley-gibson-447183984-15481458.jpg` | omelette | jalebi | masala dosa |
| `data/real_world_test/omelette/pexels-helin-gezer-903013644-25748211.jpg` | omelette | dal makhani | omelette |
| `data/real_world_test/paani_puri/pexels-fahim-shizan-2152194768-34356404.jpg` | paani puri | dal makhani | momos |
| `data/real_world_test/paani_puri/pexels-jahratreza-40000023.jpg` | paani puri | samosa | paani puri |
| `data/real_world_test/pakode/pexels-chanwalrus-958544.jpg` | pakode | chicken curry | pakode |
| `data/real_world_test/pakode/pexels-miltonphotography-35066805.jpg` | pakode | pakode | momos |
| `data/real_world_test/pakode/pexels-sabbythefreeelf-36619179.jpg` | pakode | chicken curry | pakode |
| `data/real_world_test/pakode/pexels-tamba09-37107128.jpg` | pakode | kadai paneer | pakode |
| `data/real_world_test/pav_bhaji/pexels-digitalbuggu-166654.jpg` | pav bhaji | dhokla | momos |
| `data/real_world_test/pav_bhaji/pexels-sateesh-polkam-706210252-18496583.jpg` | pav bhaji | momos | pizza |
| `data/real_world_test/pizza/pexels-matreding-9685264.jpg` | pizza | butter naan | pizza |
| `data/real_world_test/samosa/pexels-roman-saienko-1867764487-38043951.jpg` | samosa | samosa | chole bhature |
| `data/real_world_test/samosa/pexels-supibee-5031949.jpg` | samosa | cheesecake | pakode |
| `data/real_world_test/sushi/pexels-nadin-sh-78971847-39528139.jpg` | sushi | ice cream | sushi |

Full confusion matrices: `confusion_matrix_old.csv`, `confusion_matrix_new.csv` (rows actual, columns predicted).
