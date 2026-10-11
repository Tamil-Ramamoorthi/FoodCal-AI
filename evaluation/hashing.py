"""Image hashing for duplicate and overlap detection.

Two complementary checks:
- SHA-256 of file bytes finds exact copies only. Any re-encoding, resize,
  crop or metadata change produces a different hash.
- A 64-bit difference hash (dHash) of the decoded, EXIF-oriented image
  flags possible near-duplicates (re-encoded or resized copies). It is a
  heuristic: it can miss crops, rotations, heavy edits and screenshots,
  and can occasionally match visually similar but different photos, so
  matches are reported for manual review, never acted on automatically.
"""

import hashlib
import json
import os
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ARCHIVE_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CACHE_VERSION = 1


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dhash(image, size=8):
    """64-bit difference hash of an already EXIF-oriented PIL image."""
    gray = image.convert("L").resize((size + 1, size), Image.Resampling.BILINEAR)
    pixels = np.asarray(gray, dtype=np.int16)
    bits = (pixels[:, 1:] > pixels[:, :-1]).flatten()
    return int.from_bytes(np.packbits(bits).tobytes(), "big")


def dhash_from_bytes(data):
    """dHash for reference images; returns None if the image cannot be decoded."""
    try:
        with Image.open(BytesIO(data)) as image:
            image.draft("RGB", (128, 128))  # fast reduced decode for JPEGs
            return dhash(ImageOps.exif_transpose(image))
    except (OSError, ValueError, Image.DecompressionBombError):
        return None


def hamming_distances(reference_hashes, query_hash):
    return np.bitwise_count(np.bitwise_xor(reference_hashes, np.uint64(query_hash)))


# ---------------------------------------------------------------------
# Reference index: hashes of every image the model was built from
# ---------------------------------------------------------------------

class ReferenceIndex:
    """Hashes of the processed train/validation/test images and, when
    available, the original images in the source dataset archive."""

    def __init__(self, entries):
        # entries: list of dicts with source, path, label, sha256, dhash
        self.entries = entries
        self.by_sha = {}
        for position, entry in enumerate(entries):
            self.by_sha.setdefault(entry["sha256"], []).append(position)
        hashed = [(i, e["dhash"]) for i, e in enumerate(entries) if e["dhash"] is not None]
        self.dhash_positions = np.array([i for i, _ in hashed], dtype=np.int64)
        self.dhash_values = np.array([h for _, h in hashed], dtype=np.uint64)

    def sources(self):
        counts = {}
        for entry in self.entries:
            counts[entry["source"]] = counts.get(entry["source"], 0) + 1
        return counts

    def exact_matches(self, sha256):
        return [self.entries[i] for i in self.by_sha.get(sha256, [])]

    def near_matches(self, query_hash, max_distance, limit=5):
        if query_hash is None or not len(self.dhash_values):
            return []
        distances = hamming_distances(self.dhash_values, query_hash)
        hits = np.flatnonzero(distances <= max_distance)
        hits = hits[np.argsort(distances[hits], kind="stable")][:limit]
        return [
            {**self.entries[self.dhash_positions[i]], "distance": int(distances[i])}
            for i in hits
        ]


def _load_cache(cache_path):
    try:
        cache = json.loads(Path(cache_path).read_text(encoding="utf-8"))
        if cache.get("version") == CACHE_VERSION:
            return cache
    except (OSError, ValueError):
        pass
    return {"version": CACHE_VERSION, "processed": {}, "archive": None}


def _save_cache(cache_path, cache):
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(cache), encoding="utf-8")
    os.replace(temporary, cache_path)


def _hex(value):
    return None if value is None else f"{value:016x}"


def _unhex(value):
    return None if value is None else int(value, 16)


def index_processed(processed_dir, cache, progress=print):
    """Hash every image under processed/<split>/<class>/."""
    entries = []
    cached = cache["processed"]
    fresh = {}
    files = sorted(
        path for path in Path(processed_dir).glob("*/*/*")
        if path.is_file() and path.suffix.lower() in ARCHIVE_IMAGE_EXTENSIONS
    )
    for count, path in enumerate(files, 1):
        rel = path.relative_to(processed_dir).as_posix()
        stat = path.stat()
        key = [stat.st_size, stat.st_mtime_ns]
        hit = cached.get(rel)
        if hit and hit[:2] == key:
            sha, hash_hex = hit[2], hit[3]
        else:
            data = path.read_bytes()
            sha, hash_hex = sha256_bytes(data), _hex(dhash_from_bytes(data))
        fresh[rel] = [*key, sha, hash_hex]
        split, label, _ = rel.split("/", 2)
        entries.append({
            "source": f"processed/{split}", "path": f"data/processed/{rel}",
            "label": label, "sha256": sha, "dhash": _unhex(hash_hex),
        })
        if count % 5000 == 0:
            progress(f"  hashed {count}/{len(files)} processed images")
    cache["processed"] = fresh
    return entries


def index_archive(archive_path, cache, progress=print):
    """Hash the original images inside the source dataset ZIP."""
    archive_path = Path(archive_path)
    stat = archive_path.stat()
    key = {"path": str(archive_path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    cached = cache.get("archive")
    if cached and all(cached.get(k) == v for k, v in key.items()):
        rows = cached["entries"]
    else:
        rows = []
        with zipfile.ZipFile(archive_path) as archive:
            members = [
                info for info in archive.infolist()
                if not info.is_dir()
                and "__MACOSX" not in info.filename
                and not Path(info.filename).name.startswith("._")
                and Path(info.filename).suffix.lower() in ARCHIVE_IMAGE_EXTENSIONS
            ]
            for count, info in enumerate(members, 1):
                data = archive.read(info)
                parts = Path(info.filename).parts
                label = parts[-2] if len(parts) >= 2 else ""
                rows.append([info.filename, label, sha256_bytes(data), _hex(dhash_from_bytes(data))])
                if count % 5000 == 0:
                    progress(f"  hashed {count}/{len(members)} source-archive images")
        cache["archive"] = {**key, "entries": rows}
    return [
        {"source": "source-archive", "path": f"{archive_path.name}:{name}",
         "label": label, "sha256": sha, "dhash": _unhex(hash_hex)}
        for name, label, sha, hash_hex in rows
    ]


def build_reference_index(processed_dir, archive_path, cache_path, progress=print):
    cache = _load_cache(cache_path) if cache_path else {"version": CACHE_VERSION, "processed": {}, "archive": None}
    entries = []
    if processed_dir and Path(processed_dir).is_dir():
        progress(f"Indexing processed images in {processed_dir} ...")
        entries += index_processed(processed_dir, cache, progress)
    if archive_path:
        progress(f"Indexing source archive {archive_path} ...")
        entries += index_archive(archive_path, cache, progress)
    if cache_path:
        _save_cache(cache_path, cache)
    return ReferenceIndex(entries)
