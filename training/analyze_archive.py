import zipfile
import collections
import os
from pathlib import PurePosixPath

ARCHIVE = (
    r"C:\Users\tamil\.cache\huggingface\hub\datasets--ewanlong--Food_Classification_Dataset"
    r"\snapshots\36ef1209e436bb0147deede11ea6bfca3b3984e8"
    r"\Food_Classification_Dataset.zip"
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def main():
    class_counts = collections.Counter()
    image_count = 0
    total_bytes = 0

    print("Analyzing FoodCal AI dataset...")
    
    with zipfile.ZipFile(ARCHIVE, "r") as archive:
        for entry in archive.infolist():
            path = PurePosixPath(entry.filename)
            parts = path.parts

            if len(parts) < 3:
                continue

            if parts[0] != "Food Classification Dataset":
                continue

            food_class = parts[1]

            if food_class.startswith(".") or food_class.startswith("__"):
                continue

            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            class_counts[food_class] += 1
            image_count += 1
            total_bytes += entry.file_size

    print("\n========== DATASET REPORT ==========")
    print(f"Food classes: {len(class_counts)}")
    print(f"Total images: {image_count}")
    print(f"Uncompressed image size: {total_bytes / 1e9:.2f} GB")

    print("\nImages per class:")
    for food_class, count in sorted(class_counts.items()):
        print(f"{food_class}: {count}")

    print("\nAnalysis completed.")

if __name__ == "__main__":
    main()