import json
import random
import shutil
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms


# =========================================================
# 1. CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models" / "custom_cnn"

CHECKPOINT = MODEL_DIR / "food_cnn_best.pth"
HISTORY_FILE = MODEL_DIR / "training_history.json"

BACKUP = MODEL_DIR / "food_cnn_before_upgrade.pth"

IMAGE_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 12
PATIENCE = 4
SEED = 42

MODEL_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 55)
print("FOODCAL AI - IMPROVED CNN TRAINING")
print("=" * 55)
print("Device:", device)


# =========================================================
# 2. DATA AUGMENTATION
# =========================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomRotation(12),

    transforms.ColorJitter(
        brightness=0.12,
        contrast=0.12,
        saturation=0.12,
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225),
    ),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225),
    ),
])


# =========================================================
# 3. LOAD DATASET
# =========================================================

train_data = datasets.ImageFolder(
    DATA_DIR / "train",
    transform=train_transform,
)

val_data = datasets.ImageFolder(
    DATA_DIR / "validation",
    transform=eval_transform,
)

if train_data.classes != val_data.classes:
    raise ValueError("Dataset classes do not match.")

class_names = train_data.classes

print("Food classes:", len(class_names))
print("Training images:", len(train_data))
print("Validation images:", len(val_data))


# =========================================================
# 4. BALANCED SAMPLING
# =========================================================

class_counts = Counter(train_data.targets)

sample_weights = [
    1.0 / class_counts[label]
    for label in train_data.targets
]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(train_data),
    replacement=True,
)

train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=0,
)

val_loader = DataLoader(
    val_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

print("Balanced sampling enabled.")


# =========================================================
# 5. EXISTING CNN ARCHITECTURE
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
# 6. LOAD AND PROTECT THE EXISTING MODEL
# =========================================================

if not CHECKPOINT.exists():
    raise FileNotFoundError(
        f"Model checkpoint not found: {CHECKPOINT}"
    )

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device,
    weights_only=False,
)

if checkpoint["class_names"] != class_names:
    raise ValueError("Checkpoint classes do not match.")

if checkpoint["image_size"] != IMAGE_SIZE:
    raise ValueError("Checkpoint image size does not match.")

model = FoodCNN(len(class_names)).to(device)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

if not BACKUP.exists():
    shutil.copy2(CHECKPOINT, BACKUP)

print("Existing model loaded.")
print("Backup:", BACKUP)


# =========================================================
# 7. LOSS, OPTIMIZER AND SCHEDULER
# =========================================================

# Label smoothing reduces overconfident predictions.
# We use balanced sampling instead of combining it with
# inverse-frequency class weights.

criterion = nn.CrossEntropyLoss(
    label_smoothing=0.05
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0002,
    weight_decay=1e-4,
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2,
)


# =========================================================
# 8. TRAINING HISTORY
# =========================================================

if HISTORY_FILE.exists():
    history = json.loads(
        HISTORY_FILE.read_text(encoding="utf-8")
    )
else:
    history = []

best_accuracy = max(
    (
        item["validation_accuracy"]
        for item in history
    ),
    default=0.0,
)

best_state = {
    key: value.detach().cpu().clone()
    for key, value in model.state_dict().items()
}

epochs_without_improvement = 0

print(f"Previous best validation accuracy: {best_accuracy:.2%}")


# =========================================================
# 9. TRAINING LOOP
# =========================================================

for epoch in range(EPOCHS):

    model.train()

    train_loss_sum = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        # Prevent excessively large gradients.
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        train_loss_sum += (
            loss.item() * labels.size(0)
        )

        train_correct += (
            outputs.argmax(1) == labels
        ).sum().item()

        train_total += labels.size(0)

    train_loss = train_loss_sum / max(train_total, 1)
    train_accuracy = train_correct / max(train_total, 1)

    # -----------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------

    model.eval()

    val_loss_sum = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_loss_sum += (
                loss.item() * labels.size(0)
            )

            val_correct += (
                outputs.argmax(1) == labels
            ).sum().item()

            val_total += labels.size(0)

    val_loss = val_loss_sum / max(val_total, 1)
    val_accuracy = val_correct / max(val_total, 1)

    scheduler.step(val_accuracy)

    result = {
        "epoch": len(history) + 1,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "validation_loss": val_loss,
        "validation_accuracy": val_accuracy,
    }

    history.append(result)

    print(
        f"\nEpoch {epoch + 1}/{EPOCHS}"
        f" | Train: {train_accuracy:.2%}"
        f" | Validation: {val_accuracy:.2%}"
        f" | Val loss: {val_loss:.4f}"
        f" | LR: {optimizer.param_groups[0]['lr']:.6f}"
    )

    # Save the best validation checkpoint.
    if val_accuracy > best_accuracy:

        best_accuracy = val_accuracy

        best_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }

        torch.save(
            {
                "model_state_dict": best_state,
                "class_names": class_names,
                "image_size": IMAGE_SIZE,
                "model_name": "FoodCNN",
            },
            CHECKPOINT,
        )

        epochs_without_improvement = 0

        print("New best model saved.")

    else:
        epochs_without_improvement += 1

    HISTORY_FILE.write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )

    if epochs_without_improvement >= PATIENCE:

        print("\nEarly stopping: validation accuracy stopped improving.")
        break


# =========================================================
# 10. SUMMARY
# =========================================================

print("\n" + "=" * 55)
print("TRAINING COMPLETED")
print("=" * 55)

print(f"Best validation accuracy: {best_accuracy:.2%}")
print(f"Best checkpoint: {CHECKPOINT}")
print(f"History: {HISTORY_FILE}")
print("Previous checkpoint backup preserved.")