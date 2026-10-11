# FoodCal AI Dataset Directory

## raw/
Original or exported food images.

## processed/
Reserved for final model-ready datasets.

## metadata/
Dataset reports, class mappings, manifests and verification results.

## Dataset source
Hugging Face:
ewanlong/Food_Classification_Dataset

Review the dataset card, license, source image rights and terms
before redistributing data or publishing a trained model.

## Nutrition data
Image labels do not provide reliable serving weight or exact calories.
Nutrition values will be stored separately with source and serving units.

## Data splitting
Use the source dataset's official splits when available. Otherwise,
create a deterministic stratified split. Check for duplicate images
and prevent leakage between training, validation and test data.
