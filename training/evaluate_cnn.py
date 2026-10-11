import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# =========================================================
# 1. CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "custom_cnn" / "food_cnn_best.pth"
TEST_DIR = ROOT / "data" / "processed" / "test"
REPORT_PATH = ROOT / "models" / "custom_cnn" / "evaluation_report.json"

IMAGE_SIZE = 128
BATCH_SIZE = 32


# =========================================================
# 2. CNN ARCHITECTURE
# Must match the trained checkpoint.
# =========================================================

class FoodCNN(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# =========================================================
# 3. METRIC CALCULATION
# No scikit-learn dependency.
# =========================================================

def calculate_metrics(actual, predicted, classes):

    actual = np.asarray(actual, dtype=np.int64)
    predicted = np.asarray(predicted, dtype=np.int64)

    if actual.size == 0:
        raise ValueError("The test dataset is empty.")

    accuracy = float(np.mean(actual == predicted))

    class_report = {}
    f1_scores = []
    supports = []

    for class_id, class_name in enumerate(classes):

        true_positive = int(
            np.sum((actual == class_id) & (predicted == class_id))
        )

        false_positive = int(
            np.sum((actual != class_id) & (predicted == class_id))
        )

        false_negative = int(
            np.sum((actual == class_id) & (predicted != class_id))
        )

        support = int(np.sum(actual == class_id))

        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive > 0
            else 0.0
        )

        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative > 0
            else 0.0
        )

        f1_score = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        class_report[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1-score": f1_score,
            "support": support,
        }

        f1_scores.append(f1_score)
        supports.append(support)

    macro_f1 = float(np.mean(f1_scores))

    total_support = sum(supports)

    weighted_f1 = (
        sum(
            f1 * support
            for f1, support in zip(f1_scores, supports)
        ) / total_support
        if total_support > 0
        else 0.0
    )

    class_report["macro avg"] = {
        "f1-score": macro_f1,
        "support": total_support,
    }

    class_report["weighted avg"] = {
        "f1-score": weighted_f1,
        "support": total_support,
    }

    return accuracy, macro_f1, weighted_f1, class_report


# =========================================================
# 4. MAIN EVALUATION
# =========================================================

def main():

    print("=" * 55)
    print("FOODCAL AI - CNN MODEL EVALUATION")
    print("=" * 55)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}"
        )

    if not TEST_DIR.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {TEST_DIR}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # Load checkpoint.
    # weights_only=False is needed because this checkpoint
    # stores metadata alongside the model weights.
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False,
    )

    classes = checkpoint["class_names"]

    checkpoint_image_size = checkpoint.get(
        "image_size", IMAGE_SIZE
    )

    print("Model classes:", len(classes))
    print("Image size:", checkpoint_image_size)

    # Preprocessing must match training.
    transform = transforms.Compose([
        transforms.Resize(
            (checkpoint_image_size, checkpoint_image_size)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
        ),
    ])

    # Load test dataset.
    dataset = datasets.ImageFolder(
        TEST_DIR,
        transform=transform,
    )

    if dataset.classes != classes:
        raise ValueError(
            "Test dataset classes do not match checkpoint classes.\n"
            f"Checkpoint: {classes}\n"
            f"Dataset: {dataset.classes}"
        )

    if len(dataset) == 0:
        raise ValueError("No test images were found.")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # Build model and load trained weights.
    model = FoodCNN(len(classes)).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    actual = []
    predicted = []

    print("Test images:", len(dataset))
    print("Starting evaluation...")

    # Run inference without calculating gradients.
    with torch.inference_mode():

        for batch_index, (images, labels) in enumerate(loader):

            images = images.to(
                device,
                non_blocking=(device.type == "cuda"),
            )

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            predicted.extend(predictions.cpu().tolist())
            actual.extend(labels.tolist())

            if (batch_index + 1) % 20 == 0:
                print(
                    f"Processed "
                    f"{min((batch_index + 1) * BATCH_SIZE, len(dataset))}"
                    f"/{len(dataset)} images"
                )

    # Calculate evaluation metrics.
    accuracy, macro_f1, weighted_f1, report = (
        calculate_metrics(actual, predicted, classes)
    )

    results = {
        "model_name": "FoodCNN",
        "device": str(device),
        "test_images": len(dataset),
        "num_classes": len(classes),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "classification_report": report,
    }

    # Save report.
    REPORT_PATH.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 55)
    print("EVALUATION COMPLETED")
    print("=" * 55)

    print(f"Test images: {len(dataset)}")
    print(f"Test accuracy: {accuracy:.2%}")
    print(f"Macro F1: {macro_f1:.2%}")
    print(f"Weighted F1: {weighted_f1:.2%}")

    print("\nPer-class F1 scores:")

    for class_name in classes:
        metrics = report[class_name]

        print(
            f"{class_name:25s} "
            f"F1: {metrics['f1-score']:.2%} "
            f"Support: {metrics['support']}"
        )

    print("\nDetailed report saved to:")
    print(REPORT_PATH)


if __name__ == "__main__":
    main()