# Cleaned real-world evaluation

Manifest status: **reviewed**.
- Review sheet: `data/real_world_review/manual_review.csv`
- Source: 263 images (manifest SHA-256 `b4ed5ce0a4807567...`); kept 233, excluded 30.
- No image, label, checkpoint or original report was changed.

## Still needs manual review
None.

## Excluded images (30)
| Image | Label | Reason | Decided by |
|---|---|---|---|
| `data/real_world_test/Donut/pexels-tijana-drndarski-449691-3338681.jpg` | Donut | Overlaps model data: processed/train data/processed/train/Donut/000386.jpg (distance 0) | reviewer |
| `data/real_world_test/Hot Dog/pexels-kamila-bairam-80608214-12123657.jpg` | Hot Dog | Overlaps model data: processed/validation data/processed/validation/Hot Dog/000006.jpg (distance 0); processed/validation data/processed/validation/Hot Dog/000160.jpg (distance 0) | reviewer |
| `data/real_world_test/Hot Dog/pexels-polina-tankilevitch-4518656.jpg` | Hot Dog | Overlaps model data: processed/train data/processed/train/Hot Dog/000501.jpg (distance 2) | reviewer |
| `data/real_world_test/Sandwich/pexels-karola-g2-6467.jpg` | Sandwich | Overlaps model data: processed/train data/processed/train/Sandwich/000568.jpg (distance 0) | reviewer |
| `data/real_world_test/apple_pie/pexels-hilalbulbul-12782728 (1).jpg` | apple pie | Exact copy of data/real_world_test/apple_pie/pexels-hilalbulbul-12782728.jpg. | rule |
| `data/real_world_test/butter_naan/pexels-yarensphotoworks-30556657.jpg` | butter naan | Flatbread could be chapati or roti. | reviewer |
| `data/real_world_test/chicken_curry/pexels-drmkhawarnazir-38003495.jpg` | chicken curry | Yellow cubes in curry; chicken is not clearly visible. | reviewer |
| `data/real_world_test/chicken_curry/pexels-jkreat0r-32986463.jpg` | chicken curry | Curry contents are not identifiable. | reviewer |
| `data/real_world_test/dhokla/pexels-marcelochagas-3324431.jpg` | dhokla | Food is not clearly identifiable as dhokla. | reviewer |
| `data/real_world_test/idli/pexels-mahesh-kumar-ganesan-2148232589-35514447 (1).jpg` | idli | Food could be egg curry rather than idli. | reviewer |
| `data/real_world_test/idli/pexels-ryshy-s-2149956588-35539315.jpg` | idli | Same file is in several class folders (idli, masala_dosa); reviewer excluded every copy. | reviewer |
| `data/real_world_test/kaathi_rolls/pexels-magda-ehlers-pexels-12758949.jpg` | kaathi rolls | Image looks like spring rolls rather than kathi rolls. | reviewer |
| `data/real_world_test/kulfi/istockphoto-2154324654-1024x1024.jpg` | kulfi | Watermarked stock preview; licence uncertain. | reviewer |
| `data/real_world_test/kulfi/istockphoto-657073780-1024x1024.jpg` | kulfi | Same stock photograph as training image; not independent. | reviewer |
| `data/real_world_test/kulfi/pexels-sahaab-38678811.jpg` | kulfi | 121 MP street photo; food is a small part of the frame. | reviewer |
| `data/real_world_test/masala_dosa/pexels-ryshy-s-2149956588-35539315.jpg` | masala dosa | Same file is in several class folders (idli, masala_dosa); reviewer excluded every copy. | reviewer |
| `data/real_world_test/momos/pexels-bishop-tamrakar-2246201-3926123.jpg` | momos | Overlaps model data: processed/train data/processed/train/momos/000219.jpg (distance 1) | reviewer |
| `data/real_world_test/sushi/pexels-valeriya-28291554 (1).jpg` | sushi | Exact copy of data/real_world_test/sushi/pexels-valeriya-28291554.jpg. | rule |
| `data/real_world_test/sushi/pexels-valeriya-28291554 (10).jpg` | sushi | Exact copy of data/real_world_test/sushi/pexels-valeriya-28291554.jpg. | rule |
| `data/real_world_test/sushi/pexels-valeriya-28291554 (11).jpg` | sushi | Exact copy of data/real_world_test/sushi/pexels-valeriya-28291554.jpg. | rule |
| `data/real_world_test/sushi/pexels-valeriya-28291554 (12).jpg` | sushi | Exact copy of data/real_world_test/sushi/pexels-valeriya-28291554.jpg. | rule |
| `data/real_world_test/sushi/pexels-valeriya-28291554 (13).jpg` | sushi | Exact copy of data/real_world_test/sushi/pexels-valeriya-28291554.jpg. | rule |
| `data/real_world_test/sushi/pexels-valeriya-28291554 (2).jpg` | sushi | Exact copy of data/real_world_test/sushi/pexels-valeriya-28291554.jpg. | rule |
| `data/real_world_test/sushi/pexels-valeriya-28291554 (3).jpg` | sushi | Exact copy of data/real_world_test/sushi/pexels-valeriya-28291554.jpg. | rule |
| `data/real_world_test/sushi/pexels-valeriya-28291554 (4).jpg` | sushi | Exact copy of data/real_world_test/sushi/pexels-valeriya-28291554.jpg. | rule |
| `data/real_world_test/sushi/pexels-valeriya-28291554 (5).jpg` | sushi | Exact copy of data/real_world_test/sushi/pexels-valeriya-28291554.jpg. | rule |
| `data/real_world_test/sushi/pexels-valeriya-28291554 (6).jpg` | sushi | Exact copy of data/real_world_test/sushi/pexels-valeriya-28291554.jpg. | rule |
| `data/real_world_test/sushi/pexels-valeriya-28291554 (7).jpg` | sushi | Exact copy of data/real_world_test/sushi/pexels-valeriya-28291554.jpg. | rule |
| `data/real_world_test/sushi/pexels-valeriya-28291554 (8).jpg` | sushi | Exact copy of data/real_world_test/sushi/pexels-valeriya-28291554.jpg. | rule |
| `data/real_world_test/sushi/pexels-valeriya-28291554 (9).jpg` | sushi | Exact copy of data/real_world_test/sushi/pexels-valeriya-28291554.jpg. | rule |

Generated 2026-10-11T05:18:22+00:00.
- Old: `reports/real_world/real_world_evaluation.json`
- New: `reports/real_world_v2/real_world_evaluation.json`

- Old checkpoint: `models/efficientnet_b0/food_efficientnet_b0_best.pth` (SHA-256 `39363ab4f70d68fb...`)
- New checkpoint: `models/efficientnet_b0_v2/food_efficientnet_b0_v2_best.pth` (SHA-256 `e34b83ee0b1e078c...`)
- Same 233 images for both (manifest SHA-256 `e73856a83adb081d...`); identical class mapping and preprocessing.

## Overall
| Metric | Old | New | Change |
|---|---|---|---|
| Accuracy | 54.1% | 67.4% | +13.3 pts |
| Macro precision | 52.3% | 68.0% | +15.6 pts |
| Macro recall | 51.1% | 64.9% | +13.8 pts |
| Macro F1 | 47.4% | 62.8% | +15.4 pts |
| Weighted F1 | 51.1% | 66.0% | +14.9 pts |

Correct: old 126 / new 157 of 233. Macro averages cover the 34 classes with images.
Paired exact McNemar test: old right & new wrong 9, old wrong & new right 40, p = 9.3e-06.
Bootstrap 95% CI (2000 resamples): accuracy difference +7.7 to +18.9 pts; macro-F1 difference +9.1 to +20.9 pts.

## Limitations of this sample
- 233 of 263 images kept; 30 excluded (see exclusions.csv).
- 34 of 34 classes have fewer than 10 images; their per-class metrics are unreliable.
- Predictions are reused from the two original reports (same checkpoints and preprocessing); no model was re-run.
- Duplicate checks cannot find crops, re-edits or different photos of the same plate; several photos come from the same photographer and shoot, so images are not fully independent.
- Confidence intervals are wide for this sample; a difference inside them is not evidence that either model is better.
- Provenance: 263 image(s) have no documented source or licence (data/real_world_test/provenance.csv).

## Per-class changes (classes with images)
| Class | Images | Old P / R / F1 | New P / R / F1 | F1 change |
|---|---|---|---|---|
| fried rice | 7 | 42.9% / 42.9% / 42.9% | 33.3% / 14.3% / 20.0% | -22.9 |
| kaathi rolls | 6 | 60.0% / 50.0% / 54.5% | 40.0% / 33.3% / 36.4% | -18.2 |
| Donut | 8 | 100.0% / 87.5% / 93.3% | 77.8% / 87.5% / 82.4% | -11.0 |
| Hot Dog | 8 | 87.5% / 87.5% / 87.5% | 66.7% / 100.0% / 80.0% | -7.5 |
| Fries | 9 | 88.9% / 88.9% / 88.9% | 69.2% / 100.0% / 81.8% | -7.1 |
| ice cream | 8 | 62.5% / 62.5% / 62.5% | 55.6% / 62.5% / 58.8% | -3.7 |
| samosa | 7 | 46.2% / 85.7% / 60.0% | 50.0% / 71.4% / 58.8% | -1.2 |
| chai | 6 | 60.0% / 100.0% / 75.0% | 60.0% / 100.0% / 75.0% | +0.0 |
| kulfi | 2 | 0.0% / 0.0% / 0.0% | 0.0% / 0.0% / 0.0% | +0.0 |
| chicken curry | 5 | 16.7% / 20.0% / 18.2% | 20.0% / 20.0% / 20.0% | +1.8 |
| burger | 7 | 66.7% / 85.7% / 75.0% | 75.0% / 85.7% / 80.0% | +5.0 |
| jalebi | 9 | 75.0% / 100.0% / 85.7% | 90.0% / 100.0% / 94.7% | +9.0 |
| pakode | 9 | 66.7% / 66.7% / 66.7% | 66.7% / 88.9% / 76.2% | +9.5 |
| masala dosa | 7 | 62.5% / 71.4% / 66.7% | 63.6% / 100.0% / 77.8% | +11.1 |
| kadai paneer | 8 | 42.9% / 37.5% / 40.0% | 57.1% / 50.0% / 53.3% | +13.3 |
| pav bhaji | 8 | 66.7% / 75.0% / 70.6% | 100.0% / 75.0% / 85.7% | +15.1 |
| dhokla | 6 | 33.3% / 66.7% / 44.4% | 75.0% / 50.0% / 60.0% | +15.6 |
| momos | 8 | 50.0% / 50.0% / 50.0% | 53.8% / 87.5% / 66.7% | +16.7 |
| omelette | 7 | 33.3% / 28.6% / 30.8% | 60.0% / 42.9% / 50.0% | +19.2 |
| pizza | 8 | 63.6% / 87.5% / 73.7% | 88.9% / 100.0% / 94.1% | +20.4 |
| paani puri | 9 | 62.5% / 55.6% / 58.8% | 100.0% / 66.7% / 80.0% | +21.2 |
| Taco | 7 | 100.0% / 28.6% / 44.4% | 80.0% / 57.1% / 66.7% | +22.2 |
| idli | 9 | 80.0% / 44.4% / 57.1% | 100.0% / 66.7% / 80.0% | +22.9 |
| butter naan | 5 | 66.7% / 40.0% / 50.0% | 100.0% / 60.0% / 75.0% | +25.0 |
| apple pie | 5 | 33.3% / 20.0% / 25.0% | 66.7% / 40.0% / 50.0% | +25.0 |
| sushi | 2 | 16.7% / 50.0% / 25.0% | 33.3% / 100.0% / 50.0% | +25.0 |
| chapati | 5 | 21.4% / 60.0% / 31.6% | 44.4% / 80.0% / 57.1% | +25.6 |
| cheesecake | 7 | 40.0% / 28.6% / 33.3% | 100.0% / 42.9% / 60.0% | +26.7 |
| dal makhani | 7 | 33.3% / 85.7% / 48.0% | 83.3% / 71.4% / 76.9% | +28.9 |
| Taquito | 9 | n/a / 0.0% / 0.0% | 66.7% / 22.2% / 33.3% | +33.3 |
| Baked Potato | 5 | n/a / 0.0% / 0.0% | 100.0% / 20.0% / 33.3% | +33.3 |
| Sandwich | 7 | 100.0% / 14.3% / 25.0% | 83.3% / 71.4% / 76.9% | +51.9 |
| Crispy Chicken | 6 | 100.0% / 16.7% / 28.6% | 83.3% / 83.3% / 83.3% | +54.8 |
| chole bhature | 7 | n/a / 0.0% / 0.0% | 66.7% / 57.1% / 61.5% | +61.5 |

## Most frequent confusions (either model)
| Actual → predicted | Old | New |
|---|---|---|
| kadai paneer → pakode | 3 | 3 |
| Taquito → samosa | 1 | 3 |
| Taquito → dhokla | 3 | 0 |
| butter naan → chapati | 1 | 2 |
| chicken curry → dal makhani | 2 | 1 |
| chicken curry → kadai paneer | 1 | 2 |
| chole bhature → chapati | 2 | 1 |
| idli → kulfi | 1 | 2 |
| kulfi → ice cream | 1 | 2 |
| omelette → dal makhani | 3 | 0 |
| Baked Potato → apple pie | 2 | 0 |
| Baked Potato → jalebi | 1 | 1 |
| Baked Potato → kulfi | 1 | 1 |
| Baked Potato → samosa | 1 | 1 |
| Sandwich → omelette | 1 | 1 |

## Images whose prediction changed (100)
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
| `data/real_world_test/Fries/pexels-jahratreza-39034206.jpg` | Fries | samosa | Fries |
| `data/real_world_test/Hot Dog/pexels-alleksana-4113501.jpg` | Hot Dog | kaathi rolls | Hot Dog |
| `data/real_world_test/Sandwich/pexels-andrea-prochilo-3062027-34421469.jpg` | Sandwich | cheesecake | omelette |
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
