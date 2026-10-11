"""
FoodCal AI - Dataset Downloader

Downloads/accesses the selected dataset using Hugging Face Datasets.
This script does not reorganize images into data/raw automatically.
"""

from datasets import load_dataset

DATASET_NAME = "ewanlong/Food_Classification_Dataset"


def main():
    print("=" * 60)
    print("FoodCal AI - Dataset Download")
    print("=" * 60)

    print(f"\nDataset: {DATASET_NAME}")
    print("Loading dataset. The first run may take some time...")

    dataset = load_dataset(DATASET_NAME)

    print("\nDataset loaded successfully.")
    print(dataset)

    for split_name, split_data in dataset.items():
        print(f"\nSplit: {split_name}")
        print(f"Number of samples: {len(split_data)}")
        print(f"Columns: {split_data.column_names}")

        if len(split_data) > 0:
            sample = split_data[0]
            print(f"First sample keys: {list(sample.keys())}")

            if "label" in sample:
                print(f"First sample label: {sample['label']}")

    print("\nDataset inspection completed.")
    print("Use inspect_dataset.py to create a dataset report.")


if __name__ == "__main__":
    main()
