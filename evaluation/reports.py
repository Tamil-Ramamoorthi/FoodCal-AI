"""JSON and Markdown report writers for the dataset audit and the
real-world evaluation. Reports describe only what was measured."""

import json
import os
from pathlib import Path


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_json(path, data):
    write_text(path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def pct(value):
    return "n/a" if value is None else f"{value * 100:.1f}%"


def food(name):
    return name.replace("_", " ")


def empty_dataset_guidance(report):
    """Plain-text instructions shown when a real-world evaluation found no valid images."""
    q, folder = report["quality"], report["dataset"]["path"]
    classes = [c["name"] for c in report["class_mapping"]]
    return [
        f"Found 0 valid images in {folder} "
        f"({len(q['invalid_images'])} unreadable, {len(q['unsupported_files'])} unsupported file(s), "
        f"{len(q['unexpected_entries'])} ignored file(s) or folder(s)).",
        "To evaluate, add your OWN photos (never copies of training, validation or test images):",
        f"  1. Put each photo in the folder named exactly after its food class: {folder}/<class name>/<photo>",
        "     Names are case-sensitive and include spaces/underscores, e.g. apple_pie/IMG_0412.jpg, Hot Dog/hd_01.jpg.",
        f"     The {len(classes)} class folders: " + ", ".join(classes),
        "  2. Supported formats: .jpg, .jpeg, .png, .webp. Unreadable files are reported and excluded.",
        f"  3. Record where each photo came from in {folder}/provenance.csv with columns file,source,",
        "     e.g. the line: apple_pie/IMG_0412.jpg,own phone camera",
        f"  4. Aim for at least {report['dataset']['min_images_per_class']} photos per class; fewer are flagged as unreliable.",
        "  5. Re-run the evaluation for each checkpoint, then compare (see evaluation/README.md).",
    ]


def _bullets(items, empty="None."):
    return "\n".join(f"- {item}" for item in items) if items else empty


def _table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def _real_world_quality_sections(rw):
    sections = [
        "## Real-world dataset quality",
        f"- Folder: `{rw['path']}`",
        f"- Valid images: **{rw['valid_images']}**",
        f"- Invalid images: {len(rw['invalid_images'])}",
        f"- Unsupported files ignored: {len(rw['unsupported_files'])}",
        f"- Unexpected entries ignored: {len(rw['unexpected_entries'])}",
        f"- Classes without images: {len(rw['classes_without_images'])}",
        f"- Classes below {rw['min_images_per_class']} images: {len(rw['classes_below_minimum'])}",
        f"- Exact duplicate groups within the set: {len(rw['exact_duplicates_within'])}",
        f"- Possible near-duplicate pairs within the set: {len(rw['near_duplicates_within'])}",
        f"- Exact overlaps with training data sources: {len(rw['exact_overlaps'])}",
        f"- Possible near-duplicate overlaps with training data sources: {len(rw['near_duplicate_overlaps'])}",
        f"- Provenance file: `{rw['provenance']['file']}` "
        f"({'present' if rw['provenance']['present'] else 'missing'}; "
        f"{rw['provenance']['documented_images']} documented, "
        f"{rw['provenance']['undocumented_images']} undocumented)",
        "",
        f"**Independence status: `{rw['independence_status']}`.** {rw['independence_note']}",
        "",
        "### Warnings",
        _bullets(rw["warnings"]),
        "",
        "### Duplicate-detection limits",
        rw["duplicate_detection_limits"],
    ]
    if rw["invalid_images"]:
        sections += ["", "### Invalid images", _bullets([f"`{i['path']}`: {i['error']}" for i in rw["invalid_images"]])]
    if rw["exact_overlaps"]:
        sections += ["", "### Exact overlaps", _bullets([
            f"`{o['path']}` = " + ", ".join(f"{m['source']} `{m['path']}`" for m in o["matches"])
            for o in rw["exact_overlaps"]
        ])]
    if rw["near_duplicate_overlaps"]:
        sections += ["", "### Possible near-duplicate overlaps (review manually)", _bullets([
            f"`{o['path']}` ~ " + ", ".join(f"{m['source']} `{m['path']}` (distance {m['distance']})" for m in o["matches"][:3])
            for o in rw["near_duplicate_overlaps"]
        ])]
    if rw["exact_duplicates_within"]:
        sections += ["", "### Exact duplicates within the set", _bullets([
            ", ".join(f"`{p}`" for p in g["paths"]) + (" **(conflicting labels)**" if g["conflicting_labels"] else "")
            for g in rw["exact_duplicates_within"]
        ])]
    counts = [(food(k), v) for k, v in rw["images_per_class"].items()]
    sections += ["", "### Images per class", _table(["Class", "Images"], counts)]
    return sections


def audit_markdown(audit):
    p = audit["processed"]
    lines = [
        "# FoodCal AI dataset audit",
        "",
        f"Generated: {audit['generated_at']}  ",
        f"Class mapping from: `{audit['checkpoint']}` ({len(audit['class_names'])} classes)",
        "",
        "Read-only audit. No images were moved, renamed or deleted.",
        "",
        "## Processed train / validation / test splits",
        _table(["Split", "Images"], [(s, n) for s, n in p["images_per_split"].items()]),
        "",
        f"- Exact duplicate groups (SHA-256): {p['exact_duplicate_groups']} "
        f"({p['exact_duplicate_extra_copies']} extra copies)",
        f"- Groups spanning more than one split: {p['cross_split_duplicate_groups']}",
        f"- Groups with conflicting labels: {p['conflicting_label_groups']}",
        f"- Validation images with an exact copy in train: {p['validation_images_with_exact_copy_in_train']}",
        f"- Test images with an exact copy in train: {p['test_images_with_exact_copy_in_train']}",
        f"- Validation images with a possible near-duplicate in train (dHash distance <= {p['near_duplicate_max_distance']}, not exact): "
        f"{p['validation_images_with_possible_near_duplicate_in_train']}",
        f"- Test images with a possible near-duplicate in train: {p['test_images_with_possible_near_duplicate_in_train']}",
    ]
    issues = [
        f"{split}: {', '.join(f'{k}={v}' for k, v in info.items() if v)}"
        for split, info in p["folder_issues"].items() if any(info.values())
    ]
    lines += ["", "Folder issues:", _bullets(issues)]
    if p["conflicting_label_examples"]:
        lines += ["", "### Conflicting-label duplicates", _bullets([
            ", ".join(f"`{e['path']}` ({e['label']})" for e in group) for group in p["conflicting_label_examples"]
        ])]
    archive = audit.get("source_archive_classes")
    lines += ["", "## Source archive"]
    if archive:
        lines += [
            f"- Archive: `{audit['source_archive']}`",
            f"- Original images: {archive['images']} in {archive['class_folders']} class folders",
            f"- Folders not in the model: {', '.join(archive['folders_not_in_model']) or 'none'}",
            f"- Model classes not in the archive: {', '.join(archive['model_classes_not_in_archive']) or 'none'}",
        ]
    else:
        lines.append("Not checked (archive not found or disabled).")
    lines += [""] + _real_world_quality_sections(audit["real_world"])
    return "\n".join(lines) + "\n"


def _metrics_sections(metrics, title):
    lines = [
        f"## {title}",
        _table(["Metric", "Value"], [
            ("Images evaluated", metrics["total"]),
            ("Correct", metrics["correct"]),
            ("Incorrect", metrics["incorrect"]),
            ("Accuracy", pct(metrics["accuracy"])),
            *([(f"Macro precision (over {metrics['macro_f1_classes']} classes with images)", pct(metrics["macro_precision"])),
               (f"Macro recall (over {metrics['macro_f1_classes']} classes with images)", pct(metrics["macro_recall"]))]
              if "macro_precision" in metrics else []),
            (f"Macro F1 (over {metrics['macro_f1_classes']} classes with images)", pct(metrics["macro_f1"])),
            ("Weighted F1", pct(metrics["weighted_f1"])),
        ]),
    ]
    if metrics["classes_without_samples"]:
        lines += ["", f"Classes with no images (recall/F1 not measurable): "
                  f"{', '.join(food(c) for c in metrics['classes_without_samples'])}."]
    if metrics["predicted_but_absent"]:
        lines += ["", f"Predicted although no images of that class were in the set: "
                  f"{', '.join(food(c) for c in metrics['predicted_but_absent'])}."]
    return lines


def evaluation_markdown(report):
    ck = report["checkpoint"]
    lines = [
        "# FoodCal AI real-world evaluation",
        "",
        "Separate from the offline test-set report (`models/efficientnet_b0/evaluation_report.json`).",
        "",
        f"- Generated: {report['generated_at']}",
        f"- Status: **{report['status']}**",
        f"- Checkpoint: `{ck['path']}` (epoch {ck['epoch']}, SHA-256 `{ck['sha256'][:16]}...`)",
        *([f"- Image manifest: {report['image_manifest']['images']} images, SHA-256 `{report['image_manifest']['sha256'][:16]}...`"]
          if report.get("image_manifest") else []),
        f"- Dataset: `{report['dataset']['path']}`",
        f"- Device: {report['device']}",
        f"- Preprocessing: {report['preprocessing']}",
        f"- Valid images: {report['dataset']['valid_images']}; invalid images: {len(report['dataset']['invalid_images'])}",
        "",
    ]
    metrics = report["metrics"]
    if metrics is None:
        lines += ["## Metrics", "", "No valid real-world images were found, so **no metrics were calculated**.", "",
                  "## How to add real-world photos", "```", *empty_dataset_guidance(report), "```", ""]
    else:
        lines += _metrics_sections(metrics, "Overall metrics (all valid images)") + [""]
        excl = report.get("metrics_excluding_overlaps")
        if excl:
            lines += _metrics_sections(excl, "Metrics excluding images that overlap training data") + [""]
        conf = report.get("confidence")
        if conf:
            lines += [
                "## Model confidence (not accuracy)",
                f"- Mean top-1 confidence: {pct(conf['mean_confidence'])}",
                f"- Mean confidence when correct: {pct(conf['mean_confidence_correct'])}",
                f"- Mean confidence when incorrect: {pct(conf['mean_confidence_incorrect'])}",
                "",
                _table(["Confidence range", "Images", "Mean confidence", "Observed accuracy"], [
                    (f"{b['range'][0]:.0%}-{b['range'][1]:.0%}", b["count"], pct(b["mean_confidence"]), pct(b["accuracy"]))
                    for b in conf["bins"] if b["count"]
                ]),
                "",
            ]
        lines += ["## Per-class metrics", _table(
            ["Class", "Images", "Predicted", "Precision", "Recall", "F1"],
            [(food(r["food"]), r["support"], r["predicted"], pct(r["precision"]),
              "no images" if r["recall"] is None else pct(r["recall"]),
              "no images" if r["f1"] is None else pct(r["f1"]))
             for r in metrics["per_class"]],
        ), ""]
        cm = metrics["confusion_matrix"]
        active = [i for i, label in enumerate(cm["labels"])
                  if sum(cm["matrix"][i]) or sum(row[i] for row in cm["matrix"])]
        lines += [
            "## Confusion matrix",
            "Rows are actual classes, columns are predicted classes. "
            "Only classes that appear as an actual or predicted label are shown.",
            "",
            _table(["actual \\ predicted"] + [food(cm["labels"][j]) for j in active],
                   [[food(cm["labels"][i])] + [cm["matrix"][i][j] for j in active] for i in active]),
            "",
        ]
        wrong = [p for p in report.get("predictions", []) if not p["correct"]]
        lines += [f"## Incorrect predictions ({len(wrong)})", _table(
            ["Image", "Actual", "Predicted", "Confidence"],
            [(f"`{p['path']}`", food(p["actual"]), food(p["predicted"]), pct(p["confidence"])) for p in wrong],
        ) if wrong else "None.", ""]
    lines += _real_world_quality_sections(report["quality"])
    lines += ["", "## Reproduce", "```", report["command"], "```"]
    return "\n".join(lines) + "\n"
