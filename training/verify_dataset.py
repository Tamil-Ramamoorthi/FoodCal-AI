"""
FoodCal AI - Dataset Verification

Checks exported image files, dimensions, and class distribution.
"""

import json
from collections import Counter
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "processed"
MANIFEST = ROOT / "data" / "metadata" / "dataset_manifest.json"
REPORT = ROOT / "data" / "metadata" / "verification_report.json"

EXPECTED_SIZE = (224, 224)


def main():
    if not MANIFEST.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST}\n"
            "Run prepare_dataset.py first."
        )

    rows = json.loads(MANIFEST.read_text(encoding="utf-8"))

    class_counts = Counter()
    split_counts = Counter()
    errors = []
    checked = 0

    for row in rows:
        image_path = ROOT / row["path"]

        try:
            with Image.open(image_path) as image:
                image.verify()

            with Image.open(image_path) as image:
                if image.size != EXPECTED_SIZE:
                    raise ValueError(
                        f"Unexpected dimensions: {image.size}"
                    )

            checked += 1
            class_counts[row["label"]] += 1
            split_counts[row["split"]] += 1

        except Exception as error:
            errors.append({
                "path": row["path"],
                "error": str(error),
            })

    report = {
        "total_manifest_entries": len(rows),
        "valid_images": checked,
        "invalid_images": len(errors),
        "image_dimensions": list(EXPECTED_SIZE),
        "class_distribution": dict(sorted(class_counts.items())),
        "split_distribution": dict(sorted(split_counts.items())),
        "errors": errors,
    }

    REPORT.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))
    print(f"\nReport saved to: {REPORT}")


if __name__ == "__main__":
    main()

