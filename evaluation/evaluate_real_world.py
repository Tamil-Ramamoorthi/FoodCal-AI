"""Reproducible evaluation of the EfficientNet-B0 checkpoint on independent,
manually labeled real-world photos in data/real_world_test/<class>/.

Does not train or modify the model or any dataset. Run from the project
root:

    .venv-gpu\\Scripts\\python.exe -m evaluation.evaluate_real_world

Writes reports/real_world/real_world_evaluation.json and .md, separate
from the offline test-set report. That folder is what the dashboard shows
for the deployed (production) model.

Other checkpoints, e.g. the v2 candidate, need their own --output-dir:

    .venv-gpu\\Scripts\\python.exe -m evaluation.evaluate_real_world --checkpoint models/efficientnet_b0_v2/food_efficientnet_b0_v2_best.pth --output-dir reports/real_world_v2

Every checkpoint must have exactly the production checkpoint's class list
(same names, same order) and input size, so all reports are comparable.
Each report records the exact image manifest it evaluated; see
evaluation/compare_real_world.py to compare two reports.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.models.classifier import (  # noqa: E402
    PREPROCESSING_DESCRIPTION,
    build_eval_transform,
    load_classifier,
    preprocess_image,
)

from .dataset_audit import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_MIN_PER_CLASS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_NEAR_DISTANCE,
    ROOT,
    add_common_arguments,
    audit_real_world,
    rel,
    resolve_archive,
    strip_private,
    utc_now,
)
from .hashing import build_reference_index, sha256_file  # noqa: E402
from .metrics import classification_metrics, confidence_summary  # noqa: E402
from .reports import empty_dataset_guidance, evaluation_markdown, write_json, write_text  # noqa: E402

REPORT_NAME = "real_world_evaluation"


def check_checkpoint(checkpoint_path, reference_checkpoint=DEFAULT_CHECKPOINT):
    """Raises FileNotFoundError / ValueError unless the checkpoint can be evaluated
    with the same 34-class mapping and input size as the reference."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        reference = torch.load(reference_checkpoint, map_location="cpu", weights_only=True)
    except Exception as error:  # torch raises many types for corrupt files
        raise ValueError(f"{checkpoint_path} is not a readable PyTorch checkpoint: {type(error).__name__}") from error
    missing = [k for k in ("class_names", "model_state_dict", "image_size") if not isinstance(checkpoint, dict) or k not in checkpoint]
    if missing:
        raise ValueError(f"{checkpoint_path} is missing {', '.join(missing)}.")
    if list(checkpoint["class_names"]) != list(reference["class_names"]):
        raise ValueError(f"{checkpoint_path} does not use the production class list "
                         f"({len(reference['class_names'])} classes, same names and order).")
    if checkpoint["image_size"] != reference["image_size"]:
        raise ValueError(f"{checkpoint_path} expects {checkpoint['image_size']} px input, "
                         f"production uses {reference['image_size']} px.")


def image_manifest(predictions):
    """Count and digest of the exact images evaluated (path, label = class folder,
    file SHA-256). Two reports with the same digest evaluated the same images."""
    rows = sorted([p["path"], p["actual"], p["sha256"]] for p in predictions)
    return {"images": len(rows), "sha256": hashlib.sha256(json.dumps(rows).encode("utf-8")).hexdigest()}


def predict_images(classifier, records, batch_size=32):
    """Returns one prediction per record, plus records that failed to load."""
    transform = build_eval_transform(classifier.image_size)
    predictions, failed = [], []
    batch, batch_records = [], []

    def flush():
        with torch.inference_mode():
            logits = classifier.model(torch.stack(batch).to(classifier.device))
            probabilities = torch.softmax(logits, dim=1).cpu()
        top_probs, top_indices = torch.topk(probabilities, k=min(3, len(classifier.class_names)), dim=1)
        for record, probs, indices in zip(batch_records, top_probs.tolist(), top_indices.tolist()):
            predicted = classifier.class_names[indices[0]]
            predictions.append({
                "path": record["path"],
                "sha256": record["sha256"],
                "actual": record["label"],
                "predicted": predicted,
                "correct": predicted == record["label"],
                "confidence": probs[0],
                "top_predictions": [
                    {"food": classifier.class_names[i], "confidence": p} for i, p in zip(indices, probs)
                ],
            })
        batch.clear()
        batch_records.clear()

    for record in records:
        try:
            with Image.open(ROOT / record["path"]) as image:
                tensor = preprocess_image(image, classifier.image_size, transform)
        except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as error:
            failed.append({"path": record["path"], "label": record["label"],
                           "error": f"{type(error).__name__}: {error}"})
            continue
        batch.append(tensor)
        batch_records.append(record)
        if len(batch) == batch_size:
            flush()
    if batch:
        flush()
    return predictions, failed


def metrics_for(predictions, class_names):
    index = {name: i for i, name in enumerate(class_names)}
    metrics = classification_metrics(
        [index[p["actual"]] for p in predictions],
        [index[p["predicted"]] for p in predictions],
        class_names,
    )
    # Averaged over the same classes as macro F1 (those with images). A class
    # that was never predicted has undefined precision and counts as 0 here.
    present = [row for row in metrics["per_class"] if row["support"]]
    metrics["macro_precision"] = sum(row["precision"] or 0.0 for row in present) / len(present) if present else None
    metrics["macro_recall"] = sum(row["recall"] for row in present) / len(present) if present else None
    return metrics


def run_evaluation(
    real_world_dir,
    checkpoint_path,
    output_dir,
    processed_dir=None,
    archive_path=None,
    min_per_class=DEFAULT_MIN_PER_CLASS,
    near_distance=DEFAULT_NEAR_DISTANCE,
    device=None,
    cache_path=None,
    batch_size=32,
    command="python -m evaluation.evaluate_real_world",
    progress=print,
    reference_checkpoint=DEFAULT_CHECKPOINT,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    check_checkpoint(checkpoint_path, reference_checkpoint)
    progress(f"Loading checkpoint {rel(checkpoint_path)} on {device} ...")
    classifier = load_classifier(Path(checkpoint_path), device)
    class_names = classifier.class_names

    # Overlap checks need the reference hashes only when there is something to check.
    has_files = Path(real_world_dir).is_dir() and any(Path(real_world_dir).rglob("*.*"))
    reference = (
        build_reference_index(processed_dir, archive_path, cache_path, progress)
        if has_files and (processed_dir or archive_path) else None
    )

    progress(f"Auditing {rel(real_world_dir)} ...")
    quality = audit_real_world(real_world_dir, class_names, reference, min_per_class, near_distance)
    records = quality["_records"]
    for record in records:  # the ground-truth label is the class folder, nothing else
        if Path(record["path"]).parent.name != record["label"] or record["label"] not in class_names:
            raise ValueError(f"Label {record['label']!r} does not match the class folder of {record['path']}")

    progress(f"Predicting {len(records)} valid image(s) ...")
    predictions, failed = predict_images(classifier, records, batch_size)
    invalid = quality["invalid_images"] + failed

    metrics = metrics_for(predictions, class_names) if predictions else None
    confidence = confidence_summary(
        [p["confidence"] for p in predictions], [p["correct"] for p in predictions],
    ) if predictions else None

    overlapping = {o["path"] for o in quality["exact_overlaps"]} | {o["path"] for o in quality["near_duplicate_overlaps"]}
    clean = [p for p in predictions if p["path"] not in overlapping]
    excluding_overlaps = metrics_for(clean, class_names) if overlapping and clean else None

    report = {
        "report_type": "real_world_evaluation",
        "schema_version": 1,
        "status": "ok" if predictions else "no_valid_images",
        "generated_at": utc_now(),
        "command": command,
        "device": str(device),
        "checkpoint": {
            "path": rel(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "model_name": "EfficientNet-B0",
            "epoch": classifier.epoch,
            "val_accuracy": classifier.val_accuracy,
            "image_size": classifier.image_size,
        },
        "preprocessing": PREPROCESSING_DESCRIPTION.format(size=classifier.image_size),
        "class_mapping": [{"index": i, "name": name} for i, name in enumerate(class_names)],
        "image_manifest": image_manifest(predictions),
        "dataset": {
            "path": quality["path"],
            "valid_images": len(predictions),
            "invalid_images": invalid,
            "images_per_class": quality["images_per_class"],
            "classes_without_images": quality["classes_without_images"],
            "classes_below_minimum": quality["classes_below_minimum"],
            "min_images_per_class": min_per_class,
        },
        "metrics": metrics,
        "metrics_excluding_overlaps": excluding_overlaps,
        "overlapping_images_excluded": len(overlapping) if excluding_overlaps else 0,
        "confidence": confidence,
        "quality": {
            **{k: v for k, v in strip_private(quality).items() if k != "images"},
            "invalid_images": invalid,
            "valid_images": len(predictions),
        },
        "predictions": predictions,
    }

    output_dir = Path(output_dir)
    write_json(output_dir / f"{REPORT_NAME}.json", report)
    write_text(output_dir / f"{REPORT_NAME}.md", evaluation_markdown(report))
    progress(f"Wrote {rel(output_dir / (REPORT_NAME + '.json'))} and {REPORT_NAME}.md")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint on labelled real-world photos in data/real_world_test/<class>/.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples (PowerShell, from the project root):\n"
            "  .venv-gpu\\Scripts\\python.exe -m evaluation.evaluate_real_world\n"
            "  .venv-gpu\\Scripts\\python.exe -m evaluation.evaluate_real_world "
            "--checkpoint models/efficientnet_b0_v2/food_efficientnet_b0_v2_best.pth --output-dir reports/real_world_v2\n"
            "  .venv-gpu\\Scripts\\python.exe -m evaluation.compare_real_world"
        ),
    )
    add_common_arguments(parser)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)

    if (Path(args.output_dir).resolve() == DEFAULT_OUTPUT_DIR.resolve()
            and Path(args.checkpoint).resolve() != DEFAULT_CHECKPOINT.resolve()):
        parser.error(f"{rel(DEFAULT_OUTPUT_DIR)} holds the deployed model's report shown in the dashboard; "
                     "pass a separate --output-dir for any other checkpoint (e.g. reports/real_world_v2).")
    try:
        check_checkpoint(args.checkpoint)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)

    report = run_evaluation(
        real_world_dir=args.real_world_dir,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        processed_dir=args.processed_dir,
        archive_path=resolve_archive(args.source_archive),
        min_per_class=args.min_per_class,
        near_distance=args.near_distance,
        device=device,
        cache_path=None if args.no_cache else args.output_dir / "cache" / "reference_index.json",
        batch_size=args.batch_size,
        command="python -m evaluation.evaluate_real_world " + " ".join(argv if argv is not None else sys.argv[1:]),
    )

    metrics = report["metrics"]
    if metrics is None:
        print("No valid real-world images: no metrics were calculated.")
        for line in empty_dataset_guidance(report):
            print(line)
    else:
        print(f"Images: {metrics['total']}  correct: {metrics['correct']}  accuracy: {metrics['accuracy']:.2%}  "
              f"macro F1: {metrics['macro_f1']:.2%} (over {metrics['macro_f1_classes']} classes)")
    for warning in report["quality"]["warnings"]:
        print(f"WARNING: {warning}")


if __name__ == "__main__":
    main()
