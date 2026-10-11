"""
FoodCal AI - Dataset Inspection

Loads the Hugging Face dataset, examines class distribution,
and writes a JSON metadata report.

Images remain in the Hugging Face cache at this stage.
"""

import json
from collections import Counter
from pathlib import Path

from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[1]
DATASET_NAME = "ewanlong/Food_Classification_Dataset"
OUTPUT_FILE = ROOT / "data" / "metadata" / "dataset_report.json"


def main():
    print(f"Loading {DATASET_NAME}...")
    dataset = load_dataset(DATASET_NAME)

    report = {
        "dataset_name": DATASET_NAME,
        "splits": {},
    }

    for split_name, split_data in dataset.items():
        print(f"\nInspecting split: {split_name}")

        features = split_data.features
        label_feature = features.get("label")

        counts = Counter()
        invalid_images = 0
        image_feature = features.get("image")

        for index, row in enumerate(split_data):
            label = row.get("label")

            if label is not None:
                if label_feature is not None and hasattr(
                    label_feature, "names"
                ):
                    names = label_feature.names
                    if isinstance(label, int) and 0 <= label < len(names):
                        label = names[label]

                counts[str(label)] += 1

            image = row.get("image")

            if image_feature is not None:
                try:
                    if image is None:
                        invalid_images += 1
                    else:
                        image.verify()
                except Exception:
                    invalid_images += 1

            if (index + 1) % 1000 == 0:
                print(f"Checked {index + 1}/{len(split_data)} samples")

        report["splits"][split_name] = {
            "total_samples": len(split_data),
            "class_distribution": dict(sorted(counts.items())),
            "invalid_images": invalid_images,
            "columns": split_data.column_names,
        }

        print(f"Samples: {len(split_data)}")
        print(f"Classes found: {len(counts)}")
        print(f"Invalid images: {invalid_images}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"\nReport saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
