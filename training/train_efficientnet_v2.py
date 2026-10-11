"""EfficientNet-B0 retraining with the Phase 1 fixes (Phase A).

    .venv-gpu\\Scripts\\python.exe -m training.train_efficientnet_v2

Differences from training/train_efficientnet.py (kept unchanged), each
addressing a measured problem in reports/phase1_audit/PHASE1_FINDINGS.md:

1. BatchNorm layers whose parameters are frozen stay in eval mode, so
   their ImageNet statistics are not overwritten (the old loop dropped
   validation accuracy from 83.1% to 47.5% this way).
2. Each stage has its own early-stopping state; the best checkpoint is
   selected across stages by validation macro-F1 (accuracy and loss are
   tracked too).
3. The whole backbone is fine-tuned with lower learning rates for earlier
   blocks, a one-epoch warm-up and cosine decay, instead of only the last
   two blocks.
4. Class balancing is reduced from 1/n to 1/sqrt(n) sampling weights
   (--balance-power), because 1/n over-predicted small classes.
5. Augmentation adds random resized crops (zoom in), occasional zoom out,
   rotation, lighting/colour jitter and blur.
6. Data comes from the leakage-free manifests in data/manifests/v2/.
7. Training starts from ImageNet weights, not from the old checkpoint,
   whose weights were fitted on images duplicated in validation and test.

Evaluation-time preprocessing and the checkpoint format are unchanged, so
the FastAPI backend can load the result as-is. The production checkpoint
is never written.
"""

import argparse
import copy
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

from evaluation.metrics import classification_metrics

from .build_manifests import OUTPUT_DIR as MANIFEST_DIR
from .build_manifests import read_manifest

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_CHECKPOINT = ROOT / "models" / "efficientnet_b0" / "food_efficientnet_b0_best.pth"
DEFAULT_OUTPUT_DIR = ROOT / "models" / "efficientnet_b0_v2"
CHECKPOINT_NAME = "food_efficientnet_b0_v2_best.pth"
IMAGE_SIZE = 224
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
FILL = tuple(int(255 * m) for m in MEAN)


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------

class ManifestDataset(Dataset):
    def __init__(self, rows, transform):
        self.rows = rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(ROOT / row["path"]) as image:
            image = image.convert("RGB")
        return self.transform(image), row["class_index"]


def build_train_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.6, 1.0), ratio=(3 / 4, 4 / 3)),
        transforms.RandomApply([transforms.RandomAffine(0, scale=(0.7, 1.0), fill=FILL)], p=0.25),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15, fill=FILL),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.25, hue=0.03),
        transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def build_eval_transform():
    # Identical to backend/app/models/classifier.py build_eval_transform.
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def sample_weights(class_indices, num_classes, power):
    """Per-sample weights proportional to n_class ** -power (0 = none, 1 = 1/n)."""
    counts = np.bincount(class_indices, minlength=num_classes).astype(np.float64)
    return (counts ** -power)[class_indices]


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

def build_model(num_classes, pretrained=True):
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def set_trainable(model, stage):
    """stage 'head': classifier only; 'full': everything."""
    for parameter in model.features.parameters():
        parameter.requires_grad = stage == "full"
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True


def set_train_mode(model):
    """Training mode, except BatchNorm layers with frozen parameters, which
    keep using (and keep) their stored running statistics."""
    model.train()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            if not any(p.requires_grad for p in module.parameters()):
                module.eval()


def param_groups(model, stage, lr):
    if stage == "head":
        return [{"params": model.classifier.parameters(), "lr": lr, "base_lr": lr}]
    early = [p for block in model.features[:4] for p in block.parameters()]
    late = [p for block in model.features[4:] for p in block.parameters()]
    return [
        {"params": early, "lr": lr * 0.2, "base_lr": lr * 0.2},
        {"params": late, "lr": lr, "base_lr": lr},
        {"params": model.classifier.parameters(), "lr": lr * 3, "base_lr": lr * 3},
    ]


def lr_factor(step, warmup_steps, total_steps):
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


# ---------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------

def evaluate(model, loader, device, criterion, class_names):
    model.eval()
    total_loss, y_true, y_pred = 0.0, [], []
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                outputs = model(images)
            total_loss += criterion(outputs.float(), labels).item() * labels.size(0)
            y_true += labels.tolist()
            y_pred += outputs.argmax(1).tolist()
    m = classification_metrics(y_true, y_pred, class_names)
    return {"loss": total_loss / len(y_true), "accuracy": m["accuracy"], "macro_f1": m["macro_f1"]}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Retrain EfficientNet-B0 with the Phase 1 fixes.")
    parser.add_argument("--manifest-dir", type=Path, default=MANIFEST_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--head-epochs", type=int, default=3)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--finetune-epochs", type=int, default=25)
    parser.add_argument("--finetune-lr", type=float, default=2e-4)
    parser.add_argument("--head-patience", type=int, default=2)
    parser.add_argument("--finetune-patience", type=int, default=5)
    parser.add_argument("--balance-power", type=float, default=0.5)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    output_dir = args.output_dir.resolve()
    checkpoint_path = output_dir / CHECKPOINT_NAME
    if checkpoint_path == PRODUCTION_CHECKPOINT.resolve() or output_dir == PRODUCTION_CHECKPOINT.parent.resolve():
        raise SystemExit("Refusing to write into the production checkpoint folder.")
    if checkpoint_path.exists() and not args.overwrite:
        raise SystemExit(f"{checkpoint_path} exists; pass --overwrite to replace this (non-production) checkpoint.")
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    class_names = torch.load(PRODUCTION_CHECKPOINT, map_location="cpu", weights_only=True)["class_names"]
    train_rows = read_manifest(args.manifest_dir / "train.csv")
    val_rows = read_manifest(args.manifest_dir / "validation.csv")
    for row in train_rows + val_rows:
        if class_names[row["class_index"]] != row["label"]:
            raise ValueError(f"Class index mismatch for {row['path']}")

    targets = np.array([r["class_index"] for r in train_rows])
    weights = sample_weights(targets, len(class_names), args.balance_power)
    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(weights),
                                    replacement=True, generator=generator)
    loader_options = {"num_workers": args.workers, "pin_memory": device.type == "cuda",
                      "persistent_workers": args.workers > 0}
    train_loader = DataLoader(ManifestDataset(train_rows, build_train_transform()), batch_size=args.batch_size,
                              sampler=sampler, drop_last=True, **loader_options)
    val_loader = DataLoader(ManifestDataset(val_rows, build_eval_transform()), batch_size=64,
                            shuffle=False, **loader_options)

    model = build_model(len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    config = {
        **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "train_images": len(train_rows),
        "validation_images": len(val_rows),
        "manifest_sha256": {name: sha256(args.manifest_dir / name) for name in ("train.csv", "validation.csv")},
        "initial_weights": "torchvision EfficientNet_B0_Weights.DEFAULT (ImageNet)",
        "selection_metric": "validation macro-F1",
    }
    (output_dir / "training_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(config, indent=2))

    history, best = [], {"macro_f1": -1.0}
    stages = [
        ("head", args.head_epochs, args.head_lr, args.head_patience),
        ("full", args.finetune_epochs, args.finetune_lr, args.finetune_patience),
    ]
    epoch_counter, start = 0, time.time()
    for stage, epochs, lr, patience in stages:
        set_trainable(model, stage)
        optimizer = torch.optim.AdamW(param_groups(model, stage, lr), weight_decay=args.weight_decay)
        steps_per_epoch = len(train_loader)
        total_steps = epochs * steps_per_epoch
        warmup_steps = steps_per_epoch if stage == "full" else 0
        step, stage_best, waited = 0, -1.0, 0  # early-stopping state is per stage
        print(f"\n=== Stage '{stage}': up to {epochs} epochs, patience {patience} ===")
        for _ in range(epochs):
            epoch_counter += 1
            set_train_mode(model)
            seen = correct = 0
            loss_sum = 0.0
            for images, labels in train_loader:
                for group in optimizer.param_groups:
                    group["lr"] = group["base_lr"] * lr_factor(step, warmup_steps, total_steps)
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                step += 1
                loss_sum += loss.item() * labels.size(0)
                correct += (outputs.argmax(1) == labels).sum().item()
                seen += labels.size(0)

            val = evaluate(model, val_loader, device, criterion, class_names)
            record = {
                "epoch": epoch_counter, "stage": stage,
                "train_loss": loss_sum / seen, "train_accuracy": correct / seen,
                "val_loss": val["loss"], "val_accuracy": val["accuracy"], "val_macro_f1": val["macro_f1"],
                "learning_rates": [g["lr"] for g in optimizer.param_groups],
                "elapsed_minutes": (time.time() - start) / 60,
            }
            history.append(record)
            print(f"Epoch {epoch_counter} [{stage}] train acc {record['train_accuracy']:.2%} loss {record['train_loss']:.3f} | "
                  f"val acc {val['accuracy']:.2%} macro-F1 {val['macro_f1']:.2%} loss {val['loss']:.3f} | "
                  f"{record['elapsed_minutes']:.1f} min", flush=True)

            if val["macro_f1"] > best["macro_f1"]:
                best = {**val, "epoch": epoch_counter, "stage": stage}
                torch.save({
                    "model_name": "EfficientNet-B0",
                    "model_state_dict": copy.deepcopy(model.state_dict()),
                    "class_names": class_names,
                    "image_size": IMAGE_SIZE,
                    "num_classes": len(class_names),
                    "epoch": epoch_counter,
                    "val_accuracy": val["accuracy"],
                    "val_loss": val["loss"],
                    "val_macro_f1": val["macro_f1"],
                    "training_script": "training/train_efficientnet_v2.py",
                }, checkpoint_path)
                print(f"  new best (validation macro-F1) saved to {checkpoint_path.relative_to(ROOT)}")
            (output_dir / "training_history.json").write_text(json.dumps({"best": best, "epochs": history}, indent=2),
                                                              encoding="utf-8")

            if val["macro_f1"] > stage_best:
                stage_best, waited = val["macro_f1"], 0
            else:
                waited += 1
                if waited >= patience:
                    print(f"  early stopping stage '{stage}' after {waited} epochs without improvement")
                    break

    print(f"\nBest: epoch {best['epoch']} ({best['stage']}) val macro-F1 {best['macro_f1']:.2%}, "
          f"accuracy {best['accuracy']:.2%}, loss {best['loss']:.3f}. Total {(time.time() - start) / 60:.1f} min.")


if __name__ == "__main__":
    main()
