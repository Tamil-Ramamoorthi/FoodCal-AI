"""Read-only reliability diagnostics for the deployed EfficientNet-B0 model.

Determines whether wrong predictions come from preprocessing, class
mapping, dataset quality or limits of the trained model, and measures how
far model confidence can be trusted. Nothing is trained or modified.

    .venv-gpu\\Scripts\\python.exe -m evaluation.diagnose_reliability

Writes reports/reliability/reliability_diagnostics.json and .md.

Every image goes through backend/app/models/classifier.py, i.e. exactly
the preprocessing and class mapping the API uses. The confidence threshold
is chosen on the validation split only; the test split is used only to
check it.
"""

import argparse
import io
import random
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torchvision import datasets

from backend.app.models.classifier import build_eval_transform, load_classifier, preprocess_image

from .dataset_audit import (
    DEFAULT_CHECKPOINT,
    DEFAULT_NEAR_DISTANCE,
    DEFAULT_PROCESSED_DIR,
    ROOT,
    rel,
    resolve_archive,
    utc_now,
)
from .hashing import build_reference_index, hamming_distances
from .metrics import classification_metrics
from .reports import pct, write_json, write_text

DEFAULT_OUTPUT_DIR = ROOT / "reports" / "reliability"
SPLITS = ("train", "validation", "test")
THRESHOLDS = [round(t, 2) for t in np.arange(0.30, 0.96, 0.05)]
# Threshold rule, fixed before looking at the test split: the largest t for
# which validation predictions with confidence < t are right less than 50%
# of the time, i.e. below t the top-1 answer is more likely wrong than right.
LOW_BAND_MAX_ACCURACY = 0.5


def image_paths(split_dir, class_names):
    return [
        (path, label)
        for label in class_names
        for path in sorted((split_dir / label).iterdir())
        if path.is_file()
    ]


def predict(classifier, images, batch_size=64):
    """images: list of PIL images or paths. Returns softmax probabilities."""
    transform = build_eval_transform(classifier.image_size)
    out = []
    for start in range(0, len(images), batch_size):
        batch = []
        for item in images[start:start + batch_size]:
            if isinstance(item, Image.Image):
                batch.append(preprocess_image(item, classifier.image_size, transform))
            else:
                with Image.open(item) as image:
                    batch.append(preprocess_image(image, classifier.image_size, transform))
        with torch.inference_mode():
            logits = classifier.model(torch.stack(batch).to(classifier.device))
        out.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(classifier.class_names)))


# ---------------------------------------------------------------------
# Class mapping
# ---------------------------------------------------------------------

def check_class_mapping(class_names, processed_dir):
    expected = {name: index for index, name in enumerate(class_names)}
    result = {"checkpoint_classes": len(class_names), "splits": {}}
    for split in SPLITS:
        folder = datasets.ImageFolder(processed_dir / split)
        result["splits"][split] = {
            "class_to_idx_matches_checkpoint": folder.class_to_idx == expected,
            "images": len(folder.samples),
        }
    return result


# ---------------------------------------------------------------------
# Confidence calibration
# ---------------------------------------------------------------------

def calibration_table(confidence, correct):
    rows = []
    for t in THRESHOLDS:
        above = confidence >= t
        rows.append({
            "threshold": t,
            "share_below": float(np.mean(~above)),
            "accuracy_at_or_above": float(np.mean(correct[above])) if above.any() else None,
            "accuracy_below": float(np.mean(correct[~above])) if (~above).any() else None,
            "share_of_errors_below": float(np.mean(~above[~correct])) if (~correct).any() else None,
        })
    return rows


def confidence_bins(confidence, correct, bins=10):
    rows, ece = [], 0.0
    for b in range(bins):
        low, high = b / bins, (b + 1) / bins
        members = (confidence >= low) & ((confidence < high) | (b == bins - 1))
        if members.any():
            mean_conf, acc = float(confidence[members].mean()), float(correct[members].mean())
            ece += members.mean() * abs(mean_conf - acc)
            rows.append({"range": [low, high], "count": int(members.sum()),
                         "mean_confidence": mean_conf, "accuracy": acc})
    return rows, float(ece)


def choose_threshold(table):
    eligible = [row["threshold"] for row in table
                if row["accuracy_below"] is not None and row["accuracy_below"] < LOW_BAND_MAX_ACCURACY]
    return max(eligible) if eligible else None


# ---------------------------------------------------------------------
# Domain-shift probes
# ---------------------------------------------------------------------

def zoom_out(image):
    # Food occupies the centre of a larger, plain table-like background.
    canvas = Image.new("RGB", (image.width * 2, image.height * 2), (120, 100, 80))
    canvas.paste(image, (image.width // 2, image.height // 2))
    return canvas


def zoom_in(image):
    w, h = image.size
    return image.crop((w * 0.2, h * 0.2, w * 0.8, h * 0.8))


def jpeg(image, quality=30):
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


PERTURBATIONS = {
    "original": lambda im: im,
    "zoom_in_60pct_centre": zoom_in,
    "zoom_out_on_background": zoom_out,
    "darker_50pct": lambda im: ImageEnhance.Brightness(im).enhance(0.5),
    "warm_colour_cast": lambda im: Image.merge("RGB", [
        c.point(lambda v, f=f: min(255, int(v * f))) for c, f in zip(im.split(), (1.15, 1.0, 0.8))]),
    "blur_radius_2": lambda im: im.filter(ImageFilter.GaussianBlur(2)),
    "rotated_90": lambda im: im.rotate(90, expand=True),
    "jpeg_quality_30": jpeg,
}


def perturbation_probe(classifier, samples, label_index):
    labels = np.array([label_index[label] for _, label in samples])
    results = {}
    for name, transform in PERTURBATIONS.items():
        images = []
        for path, _ in samples:
            with Image.open(path) as image:
                images.append(transform(image.convert("RGB")))
        probs = predict(classifier, images)
        pred, conf = probs.argmax(1), probs.max(1)
        correct = pred == labels
        results[name] = {
            "accuracy": float(correct.mean()),
            "mean_confidence": float(conf.mean()),
            "mean_confidence_when_wrong": float(conf[~correct].mean()) if (~correct).any() else None,
            "wrong_with_confidence_at_least_70pct": float(np.mean(~correct & (conf >= 0.7))),
        }
    return results


def jpeg_step_probe(classifier, archive_path, size, seed=0):
    """prepare_dataset.py saved training images as JPEG q92 after resizing;
    the API skips that re-encode. Measures whether that changes predictions."""
    with zipfile.ZipFile(archive_path) as archive:
        names = sorted(
            n for n in archive.namelist()
            if Path(n).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
            and "__MACOSX" not in n and not Path(n).name.startswith("._")
        )
        chosen = random.Random(seed).sample(names, min(size, len(names)))
        direct, reencoded = [], []
        for name in chosen:
            with Image.open(io.BytesIO(archive.read(name))) as image:
                resized = ImageOps.exif_transpose(image).convert("RGB").resize((224, 224), Image.Resampling.LANCZOS)
            direct.append(resized)
            buffer = io.BytesIO()
            resized.save(buffer, "JPEG", quality=92)
            buffer.seek(0)
            reencoded.append(Image.open(buffer).convert("RGB"))
    a, b = predict(classifier, direct), predict(classifier, reencoded)
    return {
        "images": len(chosen),
        "top1_agreement": float(np.mean(a.argmax(1) == b.argmax(1))),
        "mean_abs_confidence_difference": float(np.mean(np.abs(a.max(1) - b.max(1)))),
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def leakage_effect(samples, correct, near_distance=DEFAULT_NEAR_DISTANCE):
    """Test accuracy on images with vs without an exact or possible
    near-duplicate in the training split (hashes from the audit cache)."""
    reference = build_reference_index(
        DEFAULT_PROCESSED_DIR, None, DEFAULT_OUTPUT_DIR.parent / "real_world" / "cache" / "reference_index.json",
        progress=lambda *_: None,
    )
    by_path = {e["path"]: e for e in reference.entries}
    train = [e for e in reference.entries if e["source"] == "processed/train"]
    train_shas = {e["sha256"] for e in train}
    train_hashes = np.array([e["dhash"] for e in train if e["dhash"] is not None], dtype=np.uint64)
    leaked = np.zeros(len(samples), dtype=bool)
    for i, (path, _) in enumerate(samples):
        entry = by_path[rel(path)]
        leaked[i] = entry["sha256"] in train_shas or (
            entry["dhash"] is not None
            and bool((hamming_distances(train_hashes, entry["dhash"]) <= near_distance).any())
        )
    return {
        "near_duplicate_max_distance": near_distance,
        "images_with_copy_in_train": int(leaked.sum()),
        "accuracy_with_copy_in_train": float(correct[leaked].mean()) if leaked.any() else None,
        "images_without_copy_in_train": int((~leaked).sum()),
        "accuracy_without_copy_in_train": float(correct[~leaked].mean()) if (~leaked).any() else None,
    }


def top_confusions(matrix, class_names, limit=15):
    pairs = [
        (matrix[i][j], class_names[i], class_names[j])
        for i in range(len(class_names)) for j in range(len(class_names))
        if i != j and matrix[i][j]
    ]
    return [{"actual": a, "predicted": p, "count": int(c)} for c, a, p in sorted(pairs, reverse=True)[:limit]]


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    classifier = load_classifier(Path(args.checkpoint), device)
    names = classifier.class_names
    index = {name: i for i, name in enumerate(names)}
    processed = Path(args.processed_dir)
    print(f"Device: {device}")

    report = {
        "report_type": "reliability_diagnostics",
        "generated_at": utc_now(),
        "checkpoint": rel(args.checkpoint),
        "class_mapping": check_class_mapping(names, processed),
        "train_images_per_class": {n: len(list((processed / "train" / n).iterdir())) for n in names},
    }

    split_results = {}
    for split in ("validation", "test"):
        print(f"Predicting {split} split ...")
        samples = image_paths(processed / split, names)
        probs = predict(classifier, [p for p, _ in samples])
        labels = np.array([index[label] for _, label in samples])
        pred, conf = probs.argmax(1), probs.max(1)
        correct = pred == labels
        metrics = classification_metrics(labels.tolist(), pred.tolist(), names)
        bins, ece = confidence_bins(conf, correct)
        split_results[split] = {
            "images": len(samples),
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "per_class": {r["food"]: {k: r[k] for k in ("support", "precision", "recall", "f1")} for r in metrics["per_class"]},
            "top_confusions": top_confusions(metrics["confusion_matrix"]["matrix"], names),
            "confidence_bins": bins,
            "expected_calibration_error": ece,
            "threshold_table": calibration_table(conf, correct),
        }
        if split == "test":
            print("Comparing test images with and without a copy in train ...")
            report["test_leakage_effect"] = leakage_effect(samples, correct)
    report["splits"] = split_results

    threshold = choose_threshold(split_results["validation"]["threshold_table"])
    test_row = next((r for r in split_results["test"]["threshold_table"] if r["threshold"] == threshold), None)
    report["low_confidence_threshold"] = {
        "rule": (f"Largest threshold t (step 0.05) for which validation predictions with confidence "
                 f"< t have accuracy < {LOW_BAND_MAX_ACCURACY:.0%}, i.e. the top-1 answer is more "
                 f"likely wrong than right. Chosen on validation only."),
        "threshold": threshold,
        "validation": next((r for r in split_results["validation"]["threshold_table"] if r["threshold"] == threshold), None),
        "test_check": test_row,
    }

    print("Perturbation probe on the test split ...")
    test_samples = image_paths(processed / "test", names)
    if args.probe_size and args.probe_size < len(test_samples):
        test_samples = random.Random(0).sample(test_samples, args.probe_size)
    report["perturbation_probe"] = {"images": len(test_samples),
                                    "results": perturbation_probe(classifier, test_samples, index)}

    archive = resolve_archive(args.source_archive)
    if archive:
        print("JPEG re-encode probe on source originals ...")
        report["jpeg_step_probe"] = jpeg_step_probe(classifier, archive, args.jpeg_probe_size)

    thumbnails = sorted((ROOT / "storage" / "thumbnails").glob("*.jpg"))
    if thumbnails:
        probs = predict(classifier, thumbnails)
        report["stored_thumbnails"] = [
            {"file": t.name[:12], "top3": [{"food": names[i], "confidence": float(p[i])} for i in np.argsort(-p)[:3]]}
            for t, p in zip(thumbnails, probs)
        ]

    out = Path(args.output_dir)
    write_json(out / "reliability_diagnostics.json", report)
    write_text(out / "reliability_diagnostics.md", markdown(report))
    print(f"Wrote {rel(out / 'reliability_diagnostics.json')} and .md")
    return report


def markdown(r):
    v, t = r["splits"]["validation"], r["splits"]["test"]
    th = r["low_confidence_threshold"]
    lines = [
        "# FoodCal AI reliability diagnostics", "",
        f"Generated {r['generated_at']} from `{r['checkpoint']}`. Read-only; nothing was trained.", "",
        "## Class mapping",
        *[f"- {s}: ImageFolder class_to_idx matches checkpoint: **{d['class_to_idx_matches_checkpoint']}** ({d['images']} images)"
          for s, d in r["class_mapping"]["splits"].items()], "",
        "## Accuracy through the API preprocessing path",
        "| Split | Images | Accuracy | Macro F1 | Weighted F1 | ECE |", "|---|---|---|---|---|---|",
        *[f"| {s} | {d['images']} | {pct(d['accuracy'])} | {pct(d['macro_f1'])} | {pct(d['weighted_f1'])} | {pct(d['expected_calibration_error'])} |"
          for s, d in (("validation", v), ("test", t))], "",
        "## Most frequent test confusions",
        "| Actual | Predicted | Count |", "|---|---|---|",
        *[f"| {c['actual']} | {c['predicted']} | {c['count']} |" for c in t["top_confusions"]], "",
        "## Confidence vs observed accuracy (validation / test)",
        "| Threshold | Val share below | Val acc below | Val acc at/above | Test share below | Test acc below | Test acc at/above | Test errors below |",
        "|---|---|---|---|---|---|---|---|",
        *[f"| {a['threshold']:.2f} | {pct(a['share_below'])} | {pct(a['accuracy_below'])} | {pct(a['accuracy_at_or_above'])} | "
          f"{pct(b['share_below'])} | {pct(b['accuracy_below'])} | {pct(b['accuracy_at_or_above'])} | {pct(b['share_of_errors_below'])} |"
          for a, b in zip(v["threshold_table"], t["threshold_table"])], "",
        f"**Low-confidence threshold: {th['threshold']}.** {th['rule']}", "",
        "## Perturbation probe (test images, synthetic shifts; not real photos)",
        "| Shift | Accuracy | Mean confidence | Mean confidence when wrong | Wrong with >=70% confidence |", "|---|---|---|---|---|",
        *[f"| {k} | {pct(d['accuracy'])} | {pct(d['mean_confidence'])} | {pct(d['mean_confidence_when_wrong'])} | {pct(d['wrong_with_confidence_at_least_70pct'])} |"
          for k, d in r["perturbation_probe"]["results"].items()], "",
    ]
    if "test_leakage_effect" in r:
        k = r["test_leakage_effect"]
        lines += ["## Effect of train/test duplicates",
                  f"- {k['images_with_copy_in_train']} test images have an exact or possible near-duplicate "
                  f"(dHash <= {k['near_duplicate_max_distance']}) in train: accuracy {pct(k['accuracy_with_copy_in_train'])}",
                  f"- {k['images_without_copy_in_train']} test images without one: accuracy {pct(k['accuracy_without_copy_in_train'])}", ""]
    if r.get("stored_thumbnails"):
        lines += ["## Stored history thumbnails (256 px re-encodes of uploads, not the originals)",
                  *[f"- `{x['file']}`: " + ", ".join(f"{p['food']} {pct(p['confidence'])}" for p in x["top3"])
                    for x in r["stored_thumbnails"]], ""]
    if "jpeg_step_probe" in r:
        j = r["jpeg_step_probe"]
        lines += ["## JPEG re-encode step", f"- {j['images']} source originals: top-1 agreement {pct(j['top1_agreement'])}, "
                  f"mean |confidence difference| {pct(j['mean_abs_confidence_difference'])}", ""]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Reliability diagnostics (read-only).")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--processed-dir", default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source-archive", default="auto")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--probe-size", type=int, default=0, help="Test images for the perturbation probe (0 = all).")
    parser.add_argument("--jpeg-probe-size", type=int, default=1000)
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
