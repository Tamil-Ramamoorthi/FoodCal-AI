import json
import random
import zipfile
from pathlib import Path
from collections import defaultdict

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
ZIP_DIR = Path.home() / ".cache" / "huggingface" / "hub" / "datasets--ewanlong--Food_Classification_Dataset"
RAW_DIR = ROOT / "data" / "raw" / "food_classification"
PROCESSED_DIR = ROOT / "data" / "processed"
METADATA_DIR = ROOT / "data" / "metadata"
IMAGE_SIZE = (224, 224)
SEED = 42
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main():
    archives = list(ZIP_DIR.rglob("Food_Classification_Dataset.zip"))
    if not archives:
        raise FileNotFoundError(f"Dataset ZIP not found under {ZIP_DIR}")

    archive = archives[0]
    if RAW_DIR.exists() and any(RAW_DIR.iterdir()):
        raise RuntimeError(
            f"{RAW_DIR} is not empty. Inspect it before running again."
        )

    images_by_class = defaultdict(list)

    with zipfile.ZipFile(archive) as zf:
        members = [
            name for name in zf.namelist()
            if Path(name).suffix.lower() in VALID_EXTENSIONS
            and "__MACOSX" not in name
            and not Path(name).name.startswith("._")
        ]

        print(f"Archive: {archive}")
        print(f"Image files found: {len(members)}")

        for index, name in enumerate(members, 1):
            parts = Path(name).parts
            if len(parts) < 3:
                continue

            class_name = parts[-2]
            images_by_class[class_name].append(name)

            if index % 5000 == 0:
                print(f"Scanned {index}/{len(members)} files")

        if not images_by_class:
            raise RuntimeError("No class folders found in the ZIP.")

        random.seed(SEED)
        manifest = []
        failures = []

        for class_name, filenames in sorted(images_by_class.items()):
            random.shuffle(filenames)
            total = len(filenames)
            train_end = int(total * 0.70)
            val_end = train_end + int(total * 0.15)

            splits = {
                "train": filenames[:train_end],
                "validation": filenames[train_end:val_end],
                "test": filenames[val_end:],
            }

            for split, split_files in splits.items():
                for index, member in enumerate(split_files):
                    try:
                        with zf.open(member) as file:
                            image = Image.open(file)
                            image = ImageOps.exif_transpose(image).convert("RGB")
                            image = image.resize(
                                IMAGE_SIZE, Image.Resampling.LANCZOS
                            )

                        output_dir = PROCESSED_DIR / split / class_name
                        output_dir.mkdir(parents=True, exist_ok=True)
                        output_path = output_dir / f"{index:06d}.jpg"
                        image.save(output_path, "JPEG", quality=92)

                        manifest.append({
                            "path": str(output_path.relative_to(ROOT)),
                            "label": class_name,
                            "split": split,
                            "width": IMAGE_SIZE[0],
                            "height": IMAGE_SIZE[1],
                        })
                    except Exception as error:
                        failures.append({
                            "file": member,
                            "error": str(error),
                        })

            print(f"{class_name}: {total} source images processed")

    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    (METADATA_DIR / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (METADATA_DIR / "preparation_errors.json").write_text(
        json.dumps(failures, indent=2), encoding="utf-8"
    )

    print("\nDataset preparation complete!")
    print(f"Classes: {len(images_by_class)}")
    print(f"Successfully prepared: {len(manifest)}")
    print(f"Failed images: {len(failures)}")
    print(f"Output: {PROCESSED_DIR}")
    print("Splits: train 70%, validation 15%, test 15%")


if __name__ == "__main__":
    main()
