"""Measures how well the dHash near-duplicate threshold works on this dataset.

Read-only. Run from the project root:

    .venv-gpu\\Scripts\\python.exe -m evaluation.calibrate_dhash

(a) Sensitivity: for a random sample of source-archive originals, re-create
    the processed copy exactly as training/prepare_dataset.py did (EXIF,
    RGB, LANCZOS 224x224, JPEG q92), and separately a re-saved copy (long
    side <= 1280px, JPEG q80), then measure the dHash distance to the original.
(b) Collision rate: for every processed image, the smallest dHash distance
    to a processed image with a different label (byte-identical files
    excluded). Some different-label neighbours are genuine cross-class
    duplicates, so this is an upper bound on the false-match rate.
"""

import argparse
import io
import random
import zipfile

import numpy as np
from PIL import Image, ImageOps

from .dataset_audit import DEFAULT_OUTPUT_DIR, DEFAULT_PROCESSED_DIR, resolve_archive
from .hashing import build_reference_index, dhash, hamming_distances

THRESHOLDS = (0, 1, 2, 3, 4, 6)


def copy_distances(archive_path, entries, sample_size, seed):
    rng = random.Random(seed)
    sample = rng.sample(entries, min(sample_size, len(entries)))
    processed_style, resaved = [], []
    with zipfile.ZipFile(archive_path) as archive:
        for entry in sample:
            member = entry["path"].split(":", 1)[1]
            with Image.open(io.BytesIO(archive.read(member))) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
            small = image.resize((224, 224), Image.Resampling.LANCZOS)
            large = image.copy()
            large.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
            for copy, quality, out in ((small, 92, processed_style), (large, 80, resaved)):
                buffer = io.BytesIO()
                copy.save(buffer, "JPEG", quality=quality)
                with Image.open(buffer) as reloaded:
                    out.append(bin(entry["dhash"] ^ dhash(reloaded)).count("1"))
    return np.array(processed_style), np.array(resaved)


def nearest_other_label(entries):
    labels = np.array([e["label"] for e in entries])
    shas = np.array([e["sha256"] for e in entries])
    hashes = np.array([e["dhash"] for e in entries], dtype=np.uint64)
    nearest = np.empty(len(entries), dtype=np.int64)
    for i in range(len(entries)):
        distances = hamming_distances(hashes, int(hashes[i]))
        nearest[i] = distances[(labels != labels[i]) & (shas != shas[i])].min()
    return nearest


def describe(distances):
    return "  ".join(f"<= {t}: {np.mean(distances <= t):.2%}" for t in THRESHOLDS)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Calibrate the dHash near-duplicate threshold.")
    parser.add_argument("--processed-dir", default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--source-archive", default="auto")
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    archive = resolve_archive(args.source_archive)
    if archive is None:
        parser.error("The source archive is required for calibration.")
    reference = build_reference_index(
        args.processed_dir, archive, DEFAULT_OUTPUT_DIR / "cache" / "reference_index.json",
    )
    originals = [e for e in reference.entries if e["source"] == "source-archive" and e["dhash"] is not None]
    processed = [e for e in reference.entries if e["source"].startswith("processed/") and e["dhash"] is not None]

    processed_style, resaved = copy_distances(archive, originals, args.sample_size, args.seed)
    print(f"(a) processed-style copies (n={len(processed_style)}): {describe(processed_style)}")
    print(f"(a) re-saved copies        (n={len(resaved)}): {describe(resaved)}")
    print(f"(b) processed images with a different-label image within distance (n={len(processed)}): "
          f"{describe(nearest_other_label(processed))}")


if __name__ == "__main__":
    main()
