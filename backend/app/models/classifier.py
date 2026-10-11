"""EfficientNet-B0 checkpoint loading and inference preprocessing.

Shared by the API (backend/app/main.py) and the real-world evaluation
(evaluation/evaluate_real_world.py) so both use exactly the same class
mapping and preprocessing.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import models, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

PREPROCESSING_DESCRIPTION = (
    "EXIF orientation applied, converted to RGB, resized to "
    "{size}x{size} with LANCZOS (no cropping, aspect ratio not preserved), "
    "then Resize(({size}, {size})), ToTensor and ImageNet normalization. "
    "Mirrors training/prepare_dataset.py + training/evaluate_efficientnet.py."
)


@dataclass
class Classifier:
    model: nn.Module
    class_names: list
    image_size: int
    device: torch.device
    epoch: object
    val_accuracy: object


def load_classifier(checkpoint_path, device):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    class_names = checkpoint["class_names"]

    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return Classifier(
        model=model,
        class_names=class_names,
        image_size=checkpoint["image_size"],
        device=device,
        epoch=checkpoint.get("epoch"),
        val_accuracy=checkpoint.get("val_accuracy"),
    )


# Inference must mirror how the model was trained and evaluated:
# training/prepare_dataset.py applied EXIF orientation, converted to
# RGB and resized every image to size x size with LANCZOS (no
# aspect-preserving resize, no cropping); train_efficientnet.py and
# evaluate_efficientnet.py then used Resize((size, size)) + Normalize.
def build_eval_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def preprocess_image(image, image_size, eval_transform):
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
    return eval_transform(image)
