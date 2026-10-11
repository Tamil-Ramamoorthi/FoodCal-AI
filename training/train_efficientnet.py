import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


# =========================================================
# 1. CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models" / "efficientnet_b0"

CHECKPOINT = MODEL_DIR / "food_efficientnet_b0_best.pth"
HISTORY_FILE = MODEL_DIR / "training_history.json"

IMAGE_SIZE = 224
BATCH_SIZE = 16
HEAD_EPOCHS = 4
FINETUNE_EPOCHS = 8
PATIENCE = 4
SEED = 42

MODEL_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

USE_AMP = device.type == "cuda"

print("=" * 60)
print("FOODCAL AI - EFFICIENTNET-B0 TRANSFER LEARNING")
print("=" * 60)
print("Device:", device)

if USE_AMP:
    print("GPU:", torch.cuda.get_device_name(0))
    print("Mixed precision: Enabled")


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
# 3. DATASETS
# =========================================================

train_dataset = datasets.ImageFolder(
    DATA_DIR / "train",
    transform=train_transform,
)

val_dataset = datasets.ImageFolder(
    DATA_DIR / "validation",
    transform=eval_transform,
)

if train_dataset.classes != val_dataset.classes:
    raise ValueError("Training and validation classes do not match.")

class_names = train_dataset.classes
num_classes = len(class_names)

print("Classes:", num_classes)
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))


# =========================================================
# 4. DATA LOADERS
# =========================================================

# Balanced sampling helps the model see minority classes more
# frequently. Validation remains unmodified.
targets = np.asarray(train_dataset.targets, dtype=np.int64)
class_counts = np.bincount(targets, minlength=num_classes)

sample_weights = 1.0 / class_counts[targets]

sampler = torch.utils.data.WeightedRandomSampler(
    weights=torch.as_tensor(sample_weights, dtype=torch.double),
    num_samples=len(sample_weights),
    replacement=True,
)

loader_options = {
    "num_workers": 0,
    "pin_memory": USE_AMP,
}

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    **loader_options,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    **loader_options,
)


# =========================================================
# 5. LOAD PRETRAINED EFFICIENTNET-B0
# =========================================================

print("\nLoading ImageNet pretrained EfficientNet-B0...")

# Downloads pretrained weights on the first run if not cached.
model = models.efficientnet_b0(
    weights=models.EfficientNet_B0_Weights.DEFAULT
)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    num_classes,
)

model = model.to(device)

# Initially freeze the feature extractor.
for parameter in model.features.parameters():
    parameter.requires_grad = False

print("Pretrained model loaded.")


# =========================================================
# 6. TRAINING UTILITIES
# =========================================================

criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=USE_AMP,
)

history = []
best_val_accuracy = 0.0
best_state = None
epochs_without_improvement = 0


def run_epoch(training, optimizer=None):

    model.train(training)

    # Keep frozen feature extractor BatchNorm statistics stable.
    if training and not any(
        parameter.requires_grad
        for parameter in model.features.parameters()
    ):
        model.features.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    loader = train_loader if training else val_loader

    context = torch.enable_grad if training else torch.no_grad

    with context():

        for images, labels in loader:

            images = images.to(
                device,
                non_blocking=USE_AMP,
            )

            labels = labels.to(
                device,
                non_blocking=USE_AMP,
            )

            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=USE_AMP,
            ):
                outputs = model(images)
                loss = criterion(outputs, labels)

            if training:
                scaler.scale(loss).backward()

                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )

                scaler.step(optimizer)
                scaler.update()

            running_loss += loss.item() * labels.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    average_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)

    return average_loss, accuracy


def save_best(epoch, val_accuracy, val_loss):

    checkpoint = {
        "model_name": "EfficientNet-B0",
        "model_state_dict": copy.deepcopy(best_state),
        "class_names": class_names,
        "image_size": IMAGE_SIZE,
        "num_classes": num_classes,
        "epoch": epoch,
        "val_accuracy": val_accuracy,
        "val_loss": val_loss,
    }

    torch.save(checkpoint, CHECKPOINT)


# =========================================================
# 7. TWO-STAGE TRAINING
# =========================================================

phases = [
    {
        "name": "HEAD TRAINING",
        "epochs": HEAD_EPOCHS,
        "unfreeze_features": False,
        "learning_rate": 1e-3,
    },
    {
        "name": "FINE-TUNING",
        "epochs": FINETUNE_EPOCHS,
        "unfreeze_features": True,
        "learning_rate": 1e-4,
    },
]

total_epochs = HEAD_EPOCHS + FINETUNE_EPOCHS
global_epoch = 0
stop_training = False

start_time = time.time()

for phase in phases:

    print("\n" + "=" * 60)
    print(phase["name"])
    print("=" * 60)

    if phase["unfreeze_features"]:
        # Fine-tune the final feature blocks first.
        for parameter in model.features.parameters():
            parameter.requires_grad = False

        for parameter in model.features[-2:].parameters():
            parameter.requires_grad = True

    optimizer = torch.optim.AdamW(
        filter(
            lambda parameter: parameter.requires_grad,
            model.parameters(),
        ),
        lr=phase["learning_rate"],
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    for phase_epoch in range(phase["epochs"]):

        global_epoch += 1

        train_loss, train_accuracy = run_epoch(
            training=True,
            optimizer=optimizer,
        )

        val_loss, val_accuracy = run_epoch(
            training=False,
        )

        scheduler.step(val_accuracy)

        learning_rate = optimizer.param_groups[0]["lr"]

        print(
            f"\nEpoch {global_epoch}/{total_epochs} "
            f"| Train: {train_accuracy:.2%} "
            f"| Validation: {val_accuracy:.2%} "
            f"| Val loss: {val_loss:.4f} "
            f"| LR: {learning_rate:.6g}"
        )

        history.append({
            "epoch": global_epoch,
            "phase": phase["name"],
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "learning_rate": learning_rate,
        })

        if (
            best_state is None
            or val_accuracy > best_val_accuracy
        ):
            best_val_accuracy = val_accuracy
            best_state = copy.deepcopy(model.state_dict())

            save_best(
                global_epoch,
                val_accuracy,
                val_loss,
            )

            print("New best checkpoint saved.")

            epochs_without_improvement = 0

        else:
            epochs_without_improvement += 1

        HISTORY_FILE.write_text(
            json.dumps(history, indent=2),
            encoding="utf-8",
        )

        if epochs_without_improvement >= PATIENCE:
            print("Early stopping triggered.")
            stop_training = True
            break

    if stop_training:
        break


# =========================================================
# 8. FINAL SUMMARY
# =========================================================

elapsed = time.time() - start_time

print("\n" + "=" * 60)
print("EFFICIENTNET-B0 TRAINING COMPLETED")
print("=" * 60)
print(f"Best validation accuracy: {best_val_accuracy:.2%}")
print(f"Training time: {elapsed / 60:.1f} minutes")
print("Checkpoint:", CHECKPOINT)
print("History:", HISTORY_FILE)

if torch.cuda.is_available():
    print(
        f"Peak GPU memory: "
        f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GB"
    )