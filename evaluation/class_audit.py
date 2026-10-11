"""Phase 1 class-wise dataset audit (read-only).

    .venv-gpu\\Scripts\\python.exe -m evaluation.class_audit

Writes reports/phase1_audit/class_audit.json, class_audit.md and contact
sheets for manual label review in reports/phase1_audit/review/. Nothing in
data/ is moved, renamed, deleted or relabelled.

Per class it reports split counts and imbalance, exact and near-duplicates
(within a split, across splits and across labels), leakage into the test
split, original image resolution, and images whose label the current model
contradicts with high confidence (candidates for manual review only).

The source image of every processed file is recovered by replaying
training/prepare_dataset.py (seed 42) and verified by dHash.
"""

import argparse
import io
import json
import random
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from backend.app.models.classifier import load_classifier

from .dataset_audit import DEFAULT_CHECKPOINT, DEFAULT_NEAR_DISTANCE, DEFAULT_PROCESSED_DIR, ROOT, rel, resolve_archive, utc_now
from .diagnose_reliability import predict
from .hashing import build_reference_index, hamming_distances
from .metrics import classification_metrics
from .reports import pct, write_json, write_text

OUTPUT_DIR = ROOT / "reports" / "phase1_audit"
SPLITS = ("train", "validation", "test")
PREPARE_SEED = 42
PREPARE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SUSPECT_CONFIDENCE = 0.90
FOCUS_CLASSES = ("idli", "kulfi", "kadai_paneer", "chicken_curry", "dhokla", "ice_cream", "Taco", "Taquito")


# ---------------------------------------------------------------------
# Provenance: replay training/prepare_dataset.py
# ---------------------------------------------------------------------

def archive_members(archive_path):
    with zipfile.ZipFile(archive_path) as archive:
        return [
            name for name in archive.namelist()
            if Path(name).suffix.lower() in PREPARE_EXTENSIONS
            and "__MACOSX" not in name and not Path(name).name.startswith("._")
        ]


def cached_archive_rows():
    """Archive member names (in archive order) and dHashes from the hash
    cache written by evaluation.dataset_audit, for when the ZIP is gone."""
    try:
        cache = json.loads((ROOT / "reports" / "real_world" / "cache" / "reference_index.json").read_text(encoding="utf-8"))
        rows = cache["archive"]["entries"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return [(name, None if hexhash is None else int(hexhash, 16)) for name, _, _, hexhash in rows]


def replay_prepare_dataset(members):
    """Returns {"<split>/<class>/<index>.jpg": archive member name}."""
    by_class = defaultdict(list)
    for name in members:
        parts = Path(name).parts
        if len(parts) >= 3:
            by_class[parts[-2]].append(name)
    random.seed(PREPARE_SEED)
    mapping = {}
    for class_name, files in sorted(by_class.items()):
        random.shuffle(files)
        train_end = int(len(files) * 0.70)
        val_end = train_end + int(len(files) * 0.15)
        for split, chunk in (("train", files[:train_end]), ("validation", files[train_end:val_end]), ("test", files[val_end:])):
            for index, member in enumerate(chunk):
                mapping[f"{split}/{class_name}/{index:06d}.jpg"] = member
    return mapping


def original_sizes(archive_path, members):
    sizes = {}
    with zipfile.ZipFile(archive_path) as archive:
        for member in members:
            try:
                with Image.open(io.BytesIO(archive.read(member))) as image:
                    sizes[member] = image.size
            except Exception:  # unreadable originals are reported, not fatal
                sizes[member] = None
    return sizes


# ---------------------------------------------------------------------
# Duplicate groups (exact SHA-256 and dHash near-duplicates)
# ---------------------------------------------------------------------

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a, b):
        self.parent[self.find(a)] = self.find(b)


def duplicate_groups(entries, near_distance):
    """Groups of processed images that are exact or near-duplicates."""
    uf = UnionFind(len(entries))
    by_sha = defaultdict(list)
    for i, e in enumerate(entries):
        by_sha[e["sha256"]].append(i)
    for members in by_sha.values():
        for other in members[1:]:
            uf.union(members[0], other)
    hashes = np.array([e["dhash"] for e in entries], dtype=np.uint64)
    near_pairs = 0
    for i in range(len(entries) - 1):
        close = np.flatnonzero(hamming_distances(hashes[i + 1:], int(hashes[i])) <= near_distance)
        for offset in close:
            j = i + 1 + int(offset)
            if entries[i]["sha256"] != entries[j]["sha256"]:
                near_pairs += 1
            uf.union(i, j)
    groups = defaultdict(list)
    for i in range(len(entries)):
        groups[uf.find(i)].append(i)
    return [g for g in groups.values() if len(g) > 1], near_pairs


# ---------------------------------------------------------------------
# Contact sheets for manual review
# ---------------------------------------------------------------------

def contact_sheet(items, path, columns=6, tile=180):
    """items: list of (image_path, caption lines)."""
    rows = max(1, (len(items) + columns - 1) // columns)
    sheet = Image.new("RGB", (columns * tile, rows * (tile + 34)), "white")
    draw = ImageDraw.Draw(sheet)
    for n, (image_path, lines) in enumerate(items):
        x, y = (n % columns) * tile, (n // columns) * (tile + 34)
        with Image.open(image_path) as image:
            sheet.paste(image.convert("RGB").resize((tile - 4, tile - 4)), (x + 2, y + 2))
        for k, line in enumerate(lines[:2]):
            draw.text((x + 3, y + tile + k * 15), line[:30], fill=(0, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=85)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def run(args):
    processed = Path(args.processed_dir)
    out = Path(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = load_classifier(Path(args.checkpoint), device)
    names = classifier.class_names
    index = {n: i for i, n in enumerate(names)}

    archive = resolve_archive(args.source_archive)
    reference = build_reference_index(processed, archive, ROOT / "reports" / "real_world" / "cache" / "reference_index.json")
    entries = [e for e in reference.entries if e["source"].startswith("processed/")]
    for e in entries:
        e["split"] = e["source"].split("/", 1)[1]
        e["key"] = e["path"].removeprefix("data/processed/")

    # --- folder check: every folder name vs checkpoint classes
    folders = {s: sorted(p.name for p in (processed / s).iterdir() if p.is_dir()) for s in SPLITS}

    # --- provenance
    provenance = {}
    cached = None if archive else cached_archive_rows()
    if archive or cached:
        print("Replaying prepare_dataset.py to recover source images ...")
        if archive:
            mapping = replay_prepare_dataset(archive_members(archive))
            archive_hash = {e["path"].split(":", 1)[1]: e["dhash"] for e in reference.entries if e["source"] == "source-archive"}
            sizes = original_sizes(archive, [mapping[e["key"]] for e in entries if e["key"] in mapping])
        else:
            # Archive deleted: replay from the cached member list; sizes unknown.
            mapping = replay_prepare_dataset([name for name, _ in cached])
            archive_hash = dict(cached)
            sizes = {}
        verified = 0
        for e in entries:
            member = mapping.get(e["key"])
            e["original"] = member
            e["original_size"] = sizes.get(member)
            if member and archive_hash.get(member) is not None and e["dhash"] is not None:
                e["provenance_distance"] = bin(archive_hash[member] ^ e["dhash"]).count("1")
                verified += e["provenance_distance"] <= 6
        provenance = {
            "processed_images": len(entries),
            "mapped": sum(1 for e in entries if e.get("original")),
            "verified_by_dhash_within_6": verified,
            "source": "archive" if archive else "hash cache (source ZIP not found; original sizes unavailable)",
        }

    # --- model predictions on every processed image
    print("Predicting all processed images with the current checkpoint ...")
    paths = [ROOT / e["path"] for e in entries]
    probs = predict(classifier, paths)
    for e, p in zip(entries, probs):
        e["pred"] = names[int(p.argmax())]
        e["pred_conf"] = float(p.max())
        e["label_prob"] = float(p[index[e["label"]]])

    # --- duplicate groups
    print("Grouping exact and near-duplicates ...")
    groups, near_pairs = duplicate_groups(entries, args.near_distance)
    group_of = {}
    for g_id, g in enumerate(groups):
        for i in g:
            group_of[i] = g_id

    per_class = {n: {s: 0 for s in SPLITS} for n in names}
    for e in entries:
        per_class[e["label"]][e["split"]] += 1

    dup_stats = {n: Counter() for n in names}
    label_conflicts, leaked_test, leaked_val = [], set(), set()
    for g in groups:
        members = [entries[i] for i in g]
        labels = {m["label"] for m in members}
        splits = {m["split"] for m in members}
        if len(labels) > 1:
            label_conflicts.append([{"path": m["path"], "label": m["label"], "pred": m["pred"]} for m in members])
        for m in members:
            dup_stats[m["label"]]["images_in_duplicate_groups"] += 1
            if len(labels) > 1:
                dup_stats[m["label"]]["in_cross_label_groups"] += 1
        for i in g:
            m = entries[i]
            others = {entries[j]["split"] for j in g if j != i}
            if m["split"] == "test" and ("train" in others or "validation" in others):
                leaked_test.add(i)
                dup_stats[m["label"]]["test_with_copy_in_train_or_val"] += 1
            if m["split"] == "validation" and "train" in others:
                leaked_val.add(i)
                dup_stats[m["label"]]["validation_with_copy_in_train"] += 1
        # extra copies within one split and class
        for (split, label), count in Counter((m["split"], m["label"]) for m in members).items():
            if count > 1:
                dup_stats[label][f"extra_copies_within_{split}"] += count - 1

    # --- label-problem candidates: model contradicts the label with high confidence
    suspects = [
        (i, e) for i, e in enumerate(entries)
        if e["pred"] != e["label"] and e["pred_conf"] >= SUSPECT_CONFIDENCE
    ]
    suspect_stats = {n: Counter() for n in names}
    for _, e in suspects:
        suspect_stats[e["label"]][e["split"]] += 1

    # --- test metrics: full test split and leakage-free subset
    test_idx = [i for i, e in enumerate(entries) if e["split"] == "test"]
    clean_idx = [i for i in test_idx if i not in leaked_test]

    def metrics_for(indices):
        return classification_metrics([index[entries[i]["label"]] for i in indices],
                                      [index[entries[i]["pred"]] for i in indices], names)

    full, clean = metrics_for(test_idx), metrics_for(clean_idx)

    def confusion_pairs(m, limit=20):
        mat = m["confusion_matrix"]["matrix"]
        pairs = [(mat[i][j], names[i], names[j], mat[i][j] / max(1, sum(mat[i])))
                 for i in range(len(names)) for j in range(len(names)) if i != j and mat[i][j]]
        return [{"actual": a, "predicted": p, "count": c, "share_of_actual": s} for c, a, p, s in sorted(pairs, reverse=True)[:limit]]

    # --- resolution of originals
    def size_stats(label):
        sides = [min(e["original_size"]) for e in entries if e["label"] == label and e.get("original_size")]
        if not sides:
            return None
        return {"median_short_side": float(np.median(sides)), "share_short_side_below_224": float(np.mean(np.array(sides) < 224))}

    train_counts = [per_class[n]["train"] for n in names]
    classes = []
    for n in names:
        m_full = next(r for r in full["per_class"] if r["food"] == n)
        m_clean = next(r for r in clean["per_class"] if r["food"] == n)
        classes.append({
            "class": n,
            **{f"{s}_images": per_class[n][s] for s in SPLITS},
            "train_share": per_class[n]["train"] / sum(train_counts),
            "train_ratio_to_largest": per_class[n]["train"] / max(train_counts),
            "duplicates": dict(dup_stats[n]),
            "high_confidence_disagreements": dict(suspect_stats[n]),
            "original_resolution": size_stats(n),
            "test_full": {k: m_full[k] for k in ("support", "precision", "recall", "f1")},
            "test_leakage_free": {k: m_clean[k] for k in ("support", "precision", "recall", "f1")},
        })

    # --- contact sheets
    print("Writing contact sheets ...")
    review = out / "review"
    rng = random.Random(0)
    for n in FOCUS_CLASSES:
        pool = [e for e in entries if e["label"] == n and e["split"] == "train"]
        sample = rng.sample(pool, min(36, len(pool)))
        contact_sheet([(ROOT / e["path"], [e["key"].split("/", 1)[1], f"pred {e['pred']} {e['pred_conf']:.0%}"]) for e in sample],
                      review / f"sample_train_{n.replace(' ', '_')}.jpg")
    for n in names:
        bad = sorted((e for _, e in suspects if e["label"] == n), key=lambda e: -e["pred_conf"])
        if bad:
            contact_sheet([(ROOT / e["path"], [e["key"], f"model: {e['pred']} {e['pred_conf']:.0%}"]) for e in bad[:48]],
                          review / f"suspects_{n.replace(' ', '_')}.jpg")
    # predictions of test images of every class that were called kulfi
    into_kulfi = [e for e in entries if e["pred"] == "kulfi" and e["label"] != "kulfi"]
    if into_kulfi:
        contact_sheet([(ROOT / e["path"], [e["key"], f"kulfi {e['pred_conf']:.0%}"]) for e in into_kulfi[:48]],
                      review / "other_classes_predicted_as_kulfi.jpg")

    report = {
        "report_type": "phase1_class_audit",
        "generated_at": utc_now(),
        "checkpoint": rel(args.checkpoint),
        "class_names": names,
        "folder_names_match_checkpoint": {s: folders[s] == sorted(names) for s in SPLITS},
        "provenance": provenance,
        "near_duplicate_max_distance": args.near_distance,
        "imbalance": {
            "largest_train_class": names[int(np.argmax(train_counts))],
            "largest_train_images": max(train_counts),
            "smallest_train_class": names[int(np.argmin(train_counts))],
            "smallest_train_images": min(train_counts),
            "max_to_min_ratio": max(train_counts) / min(train_counts),
        },
        "duplicates": {
            "groups": len(groups),
            "images_in_groups": sum(len(g) for g in groups),
            "near_duplicate_pairs_not_exact": near_pairs,
            "cross_label_groups": len(label_conflicts),
            "test_images_with_copy_in_train_or_validation": len(leaked_test),
            "validation_images_with_copy_in_train": len(leaked_val),
        },
        "label_conflict_groups": label_conflicts,
        "high_confidence_disagreement_threshold": SUSPECT_CONFIDENCE,
        "high_confidence_disagreements": [
            {"path": e["path"], "split": e["split"], "label": e["label"], "predicted": e["pred"],
             "confidence": e["pred_conf"], "original": e.get("original")}
            for _, e in sorted(suspects, key=lambda t: -t[1]["pred_conf"])
        ],
        "test_full": {k: full[k] for k in ("total", "accuracy", "macro_f1", "weighted_f1")},
        "test_leakage_free": {k: clean[k] for k in ("total", "accuracy", "macro_f1", "weighted_f1")},
        "leakage_free_test_paths": sorted(entries[i]["path"] for i in clean_idx),
        "test_confusions_full": confusion_pairs(full),
        "test_confusions_leakage_free": confusion_pairs(clean),
        "classes": classes,
    }
    write_json(out / "class_audit.json", report)
    write_text(out / "class_audit.md", markdown(report))
    print(f"Wrote {rel(out / 'class_audit.json')}, class_audit.md and review/ contact sheets")
    return report


def markdown(r):
    d, im = r["duplicates"], r["imbalance"]
    lines = [
        "# Phase 1 class-wise dataset audit", "",
        f"Generated {r['generated_at']}. Read-only: no image was moved, renamed, deleted or relabelled.", "",
        "## Summary",
        f"- Class folders match checkpoint classes: {r['folder_names_match_checkpoint']}",
        f"- Source images recovered for {r['provenance'].get('mapped', 0)} of {r['provenance'].get('processed_images', 0)} processed files; "
        f"{r['provenance'].get('verified_by_dhash_within_6', 0)} verified by dHash.",
        f"- Train imbalance: {im['largest_train_class']} {im['largest_train_images']} vs {im['smallest_train_class']} "
        f"{im['smallest_train_images']} images (ratio {im['max_to_min_ratio']:.1f}).",
        f"- Duplicate groups (exact or dHash <= {r['near_duplicate_max_distance']}): {d['groups']} groups, {d['images_in_groups']} images; "
        f"{d['cross_label_groups']} groups mix labels.",
        f"- Leakage: {d['test_images_with_copy_in_train_or_validation']} test images have a copy in train or validation; "
        f"{d['validation_images_with_copy_in_train']} validation images have a copy in train.",
        f"- Images the current model contradicts with >= {r['high_confidence_disagreement_threshold']:.0%} confidence: "
        f"{len(r['high_confidence_disagreements'])} (candidates for manual review, not confirmed label errors).",
        f"- Current model, full test split: accuracy {pct(r['test_full']['accuracy'])}, macro F1 {pct(r['test_full']['macro_f1'])} ({r['test_full']['total']} images)",
        f"- Current model, leakage-free test subset: accuracy {pct(r['test_leakage_free']['accuracy'])}, "
        f"macro F1 {pct(r['test_leakage_free']['macro_f1'])} ({r['test_leakage_free']['total']} images)", "",
        "## Per class",
        "| Class | Train | Val | Test | Train vs largest | Dup images | Test leaked | Cross-label dup | Model disagrees >=90% (train/val/test) | Median orig. short side | Clean-test recall | Clean-test precision | Clean-test F1 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in sorted(r["classes"], key=lambda c: c["test_leakage_free"]["f1"] or 0):
        dup, s, res, t = c["duplicates"], c["high_confidence_disagreements"], c["original_resolution"], c["test_leakage_free"]
        lines.append(
            f"| {c['class']} | {c['train_images']} | {c['validation_images']} | {c['test_images']} | {c['train_ratio_to_largest']:.2f} | "
            f"{dup.get('images_in_duplicate_groups', 0)} | {dup.get('test_with_copy_in_train_or_val', 0)} | {dup.get('in_cross_label_groups', 0)} | "
            f"{s.get('train', 0)}/{s.get('validation', 0)}/{s.get('test', 0)} | {res['median_short_side'] if res else 'n/a'} | "
            f"{pct(t['recall'])} | {pct(t['precision'])} | {pct(t['f1'])} |")
    lines += ["", "## Most frequent confusions (leakage-free test subset)",
              "| Actual | Predicted | Count | Share of actual class |", "|---|---|---|---|",
              *[f"| {c['actual']} | {c['predicted']} | {c['count']} | {pct(c['share_of_actual'])} |" for c in r["test_confusions_leakage_free"]],
              "", "## Cross-label duplicate groups (same or near-identical image, different labels)"]
    for g in r["label_conflict_groups"][:40]:
        lines.append("- " + "; ".join(f"`{m['path'].removeprefix('data/processed/')}` ({m['label']})" for m in g))
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Phase 1 class-wise dataset audit (read-only).")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--processed-dir", default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--source-archive", default="auto")
    parser.add_argument("--near-distance", type=int, default=DEFAULT_NEAR_DISTANCE)
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
