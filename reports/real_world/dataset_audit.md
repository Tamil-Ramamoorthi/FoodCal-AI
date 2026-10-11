# FoodCal AI dataset audit

Generated: 2026-10-10T14:37:41+00:00  
Class mapping from: `models/efficientnet_b0/food_efficientnet_b0_best.pth` (34 classes)

Read-only audit. No images were moved, renamed or deleted.

## Processed train / validation / test splits
| Split | Images |
|---|---|
| train | 16699 |
| validation | 3572 |
| test | 3602 |

- Exact duplicate groups (SHA-256): 449 (480 extra copies)
- Groups spanning more than one split: 203
- Groups with conflicting labels: 3
- Validation images with an exact copy in train: 96
- Test images with an exact copy in train: 96
- Validation images with a possible near-duplicate in train (dHash distance <= 3, not exact): 145
- Test images with a possible near-duplicate in train: 151

Folder issues:
None.

### Conflicting-label duplicates
- `data/processed/test/pav_bhaji/000031.jpg` (pav_bhaji), `data/processed/train/pizza/000040.jpg` (pizza)
- `data/processed/test/Sandwich/000018.jpg` (Sandwich), `data/processed/train/Sandwich/000402.jpg` (Sandwich), `data/processed/train/Sandwich/000413.jpg` (Sandwich), `data/processed/validation/Hot Dog/000217.jpg` (Hot Dog)
- `data/processed/train/Donut/000102.jpg` (Donut), `data/processed/train/Sandwich/000095.jpg` (Sandwich), `data/processed/train/Sandwich/000662.jpg` (Sandwich)

## Source archive
- Archive: `C:\Users\tamil\.cache\huggingface\hub\datasets--ewanlong--Food_Classification_Dataset\snapshots\36ef1209e436bb0147deede11ea6bfca3b3984e8\Food_Classification_Dataset.zip`
- Original images: 23873 in 34 class folders
- Folders not in the model: none
- Model classes not in the archive: none

## Real-world dataset quality
- Folder: `data/real_world_test`
- Valid images: **0**
- Invalid images: 0
- Unsupported files ignored: 0
- Unexpected entries ignored: 0
- Classes without images: 34
- Classes below 10 images: 0
- Exact duplicate groups within the set: 0
- Possible near-duplicate pairs within the set: 0
- Exact overlaps with training data sources: 0
- Possible near-duplicate overlaps with training data sources: 0
- Provenance file: `data/real_world_test/provenance.csv` (missing; 0 documented, 0 undocumented)

**Independence status: `not_assessed`.** There are no valid real-world images to assess.

### Warnings
- No valid real-world images were found, so no real-world metrics can be calculated.

### Duplicate-detection limits
Exact duplicates are found by SHA-256 of file bytes, so any re-encoding, resizing, cropping or metadata change hides a copy. Possible near-duplicates are found with a 64-bit difference hash (Hamming distance <= 3); this can miss crops, rotations, mirrored or heavily edited copies and can occasionally match different but similar-looking photos, so matches need manual review.

### Images per class
| Class | Images |
|---|---|
| Baked Potato | 0 |
| Crispy Chicken | 0 |
| Donut | 0 |
| Fries | 0 |
| Hot Dog | 0 |
| Sandwich | 0 |
| Taco | 0 |
| Taquito | 0 |
| apple pie | 0 |
| burger | 0 |
| butter naan | 0 |
| chai | 0 |
| chapati | 0 |
| cheesecake | 0 |
| chicken curry | 0 |
| chole bhature | 0 |
| dal makhani | 0 |
| dhokla | 0 |
| fried rice | 0 |
| ice cream | 0 |
| idli | 0 |
| jalebi | 0 |
| kaathi rolls | 0 |
| kadai paneer | 0 |
| kulfi | 0 |
| masala dosa | 0 |
| momos | 0 |
| omelette | 0 |
| paani puri | 0 |
| pakode | 0 |
| pav bhaji | 0 |
| pizza | 0 |
| samosa | 0 |
| sushi | 0 |
