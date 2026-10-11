import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


# =====================================================
# CONFIGURATION
# =====================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT / "models" / "efficientnet_b0"
    / "food_efficientnet_b0_best.pth"
)

TEST_DIR = ROOT / "data" / "processed" / "test"

REPORT_PATH = (
    ROOT / "models" / "efficientnet_b0"
    / "evaluation_report.json"
)

BATCH_SIZE = 16


# =====================================================
# METRICS WITHOUT SCIKIT-LEARN
# =====================================================

def calculate_metrics(actual, predicted, classes):

    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    accuracy = float(np.mean(actual == predicted))

    class_report = {}
    f1_scores = []
    supports = []

    for class_id, class_name in enumerate(classes):

        tp = int(np.sum(
            (actual == class_id) &
            (predicted == class_id)
        ))

        fp = int(np.sum(
            (actual != class_id) &
            (predicted == class_id)
        ))

        fn = int(np.sum(
            (actual == class_id) &
            (predicted != class_id)
        ))

        support = int(np.sum(actual == class_id))

        precision = (
            tp / (tp + fp) if tp + fp else 0.0
        )

        recall = (
            tp / (tp + fn) if tp + fn else 0.0
        )

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        class_report[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
            "support": support,
        }

        f1_scores.append(f1)
        supports.append(support)

    macro_f1 = float(np.mean(f1_scores))

    total = sum(supports)

    weighted_f1 = (
        sum(
            f1 * support
            for f1, support in zip(f1_scores, supports)
        ) / total
        if total else 0.0
    )

    return accuracy, macro_f1, weighted_f1, class_report


# =====================================================
# EVALUATION
# =====================================================

def main():

    print("=" * 55)
    print("FOODCAL AI - EFFICIENTNET-B0 EVALUATION")
    print("=" * 55)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {MODEL_PATH}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )

    classes = checkpoint["class_names"]
    image_size = checkpoint["image_size"]

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
        ),
    ])

    dataset = datasets.ImageFolder(
        TEST_DIR,
        transform=transform,
    )

    if dataset.classes != classes:
        raise ValueError(
            "Test dataset classes do not match checkpoint classes."
        )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = models.efficientnet_b0(weights=None)

    model.classifier[1] = torch.nn.Linear(
        model.classifier[1].in_features,
        len(classes),
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    actual = []
    predicted = []

    print("Test images:", len(dataset))
    print("Starting evaluation...")

    with torch.inference_mode():

        for index, (images, labels) in enumerate(loader):

            images = images.to(device)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            actual.extend(labels.tolist())
            predicted.extend(predictions.cpu().tolist())

            if (index + 1) % 20 == 0:
                print(
                    f"Processed "
                    f"{min((index + 1) * BATCH_SIZE, len(dataset))}"
                    f"/{len(dataset)} images"
                )

    accuracy, macro_f1, weighted_f1, report = (
        calculate_metrics(actual, predicted, classes)
    )

    results = {
        "model_name": "EfficientNet-B0",
        "test_images": len(dataset),
        "num_classes": len(classes),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "classification_report": report,
    }

    REPORT_PATH.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 55)
    print("EVALUATION COMPLETED")
    print("=" * 55)

    print(f"Test accuracy: {accuracy:.2%}")
    print(f"Macro F1: {macro_f1:.2%}")
    print(f"Weighted F1: {weighted_f1:.2%}")

    print("\nPer-class F1 scores:")

    for name in classes:
        metrics = report[name]

        print(
            f"{name:25s} "
            f"F1: {metrics['f1-score']:.2%} "
            f"Support: {metrics['support']}"
        )

    print("\nReport saved to:", REPORT_PATH)


if __name__ == "__main__":
    main()