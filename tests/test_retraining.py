"""Focused tests for Phase A + B: leakage-free manifests, the fixed
training helpers, the model-comparison maths and checkpoint compatibility.

Run from the project root:
    .venv-gpu\\Scripts\\python.exe -B -m unittest tests.test_retraining -v
"""
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.models.classifier import build_eval_transform as api_eval_transform  # noqa: E402
from backend.app.models.classifier import load_classifier  # noqa: E402
from evaluation.compare_models import macro_f1_fast, mcnemar_exact, summarize  # noqa: E402
from evaluation.hashing import build_reference_index, hamming_distances  # noqa: E402
from evaluation.metrics import classification_metrics  # noqa: E402
from training import train_efficientnet_v2 as v2  # noqa: E402
from training.build_manifests import read_manifest  # noqa: E402

MANIFESTS = ROOT / "data" / "manifests" / "v2"
PRODUCTION = ROOT / "models" / "efficientnet_b0" / "food_efficientnet_b0_best.pth"
NEW = ROOT / "models" / "efficientnet_b0_v2" / "food_efficientnet_b0_v2_best.pth"
CLASS_NAMES = torch.load(PRODUCTION, map_location="cpu", weights_only=True)["class_names"]


class ManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = read_manifest(MANIFESTS / "train.csv")
        cls.val = read_manifest(MANIFESTS / "validation.csv")
        cls.test = read_manifest(MANIFESTS / "test_clean.csv")

    def test_counts_and_split_membership(self):
        self.assertEqual((len(self.train), len(self.val), len(self.test)), (15949, 3572, 3304))
        for rows, split in ((self.train, "train"), (self.val, "validation"), (self.test, "test")):
            with self.subTest(split=split):
                self.assertTrue(all(r["path"].startswith(f"data/processed/{split}/") for r in rows))
                self.assertEqual(len({r["path"] for r in rows}), len(rows), "no repeated paths")

    def test_labels_folders_and_class_indices(self):
        for row in self.train + self.val + self.test:
            self.assertEqual(Path(row["path"]).parent.name, row["label"])
            self.assertEqual(CLASS_NAMES[row["class_index"]], row["label"])
        self.assertEqual({r["label"] for r in self.train}, set(CLASS_NAMES), "all 34 classes kept in training")

    def test_test_manifest_is_the_audited_clean_subset(self):
        audit = json.loads((ROOT / "reports" / "phase1_audit" / "class_audit.json").read_text(encoding="utf-8"))
        self.assertEqual(sorted(r["path"] for r in self.test), sorted(audit["leakage_free_test_paths"]))

    def test_no_exact_or_near_copy_of_held_out_images_in_train(self):
        ref = build_reference_index(ROOT / "data" / "processed", None,
                                    ROOT / "reports" / "real_world" / "cache" / "reference_index.json",
                                    progress=lambda *_: None)
        by_path = {e["path"]: e for e in ref.entries}
        for row in self.train + self.val + self.test:  # manifest hashes match the files
            self.assertEqual(by_path[row["path"]]["sha256"], row["sha256"])
        train_shas = {r["sha256"] for r in self.train}
        train_hashes = np.array([by_path[r["path"]]["dhash"] for r in self.train], dtype=np.uint64)
        for row in self.val + self.test:
            self.assertNotIn(row["sha256"], train_shas, row["path"])
            self.assertGreater(int(hamming_distances(train_hashes, by_path[row["path"]]["dhash"]).min()), 3, row["path"])
        self.assertEqual(len(train_shas), len(self.train), "no exact duplicates inside train")


class TrainingHelperTests(unittest.TestCase):
    def setUp(self):
        self.model = v2.build_model(34, pretrained=False)
        self.bns = [m for m in self.model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]

    def test_frozen_batchnorm_stays_in_eval_mode(self):
        v2.set_trainable(self.model, "head")
        v2.set_train_mode(self.model)
        self.assertTrue(self.bns and not any(bn.training for bn in self.bns))
        self.assertTrue(self.model.classifier.training)
        before = [bn.running_mean.clone() for bn in self.bns]
        self.model(torch.randn(4, 3, 224, 224))
        self.assertTrue(all(torch.equal(a, bn.running_mean) for a, bn in zip(before, self.bns)),
                        "frozen BatchNorm statistics must not change")

    def test_full_finetune_trains_every_layer(self):
        v2.set_trainable(self.model, "full")
        v2.set_train_mode(self.model)
        self.assertTrue(all(bn.training for bn in self.bns))
        groups = v2.param_groups(self.model, "full", 1e-4)
        ids = sorted(id(p) for g in groups for p in g["params"])
        self.assertEqual(ids, sorted(id(p) for p in self.model.parameters()))
        self.assertLess(groups[0]["lr"], groups[1]["lr"], "earlier blocks learn more slowly")

    def test_sampling_weights(self):
        labels = np.array([0] * 400 + [1] * 100)
        for power, ratio in ((0, 1.0), (0.5, 2.0), (1.0, 4.0)):
            w = v2.sample_weights(labels, 2, power)
            self.assertAlmostEqual(w[-1] / w[0], ratio)

    def test_eval_transform_matches_api(self):
        with Image.open(ROOT / self_test_image()) as image:
            rgb = image.convert("RGB")
        self.assertTrue(torch.equal(v2.build_eval_transform()(rgb), api_eval_transform(224)(rgb)))
        self.assertEqual(tuple(v2.build_train_transform()(rgb).shape), (3, 224, 224))

    def test_lr_schedule(self):
        self.assertAlmostEqual(v2.lr_factor(0, 10, 100), 0.1)
        self.assertAlmostEqual(v2.lr_factor(10, 10, 100), 1.0)
        self.assertAlmostEqual(v2.lr_factor(100, 10, 100), 0.0)

    def test_refuses_to_write_production_folder(self):
        with self.assertRaises(SystemExit):
            v2.main(["--output-dir", str(PRODUCTION.parent)])


def self_test_image():
    return Path("data/processed/test/pizza/000000.jpg")


class ComparisonMathTests(unittest.TestCase):
    def test_mcnemar_exact(self):
        old = np.array([True] * 10 + [False] * 10)
        new = np.array([True] * 8 + [False] * 2 + [True] * 9 + [False])
        result = mcnemar_exact(old, new)
        self.assertEqual((result["old_right_new_wrong"], result["old_wrong_new_right"]), (2, 9))
        self.assertAlmostEqual(result["p_value"], 2 * (1 + 11 + 55) / 2 ** 11)
        self.assertEqual(mcnemar_exact(old, old)["p_value"], 1.0)

    def test_macro_metrics_known_example(self):
        m = summarize([0, 0, 0, 1, 1, 2], [0, 0, 1, 1, 0, 2], ["a", "b", "c"])
        self.assertAlmostEqual(m["macro_precision"], (2 / 3 + 1 / 2 + 1) / 3)
        self.assertAlmostEqual(m["macro_recall"], (2 / 3 + 1 / 2 + 1) / 3)
        self.assertEqual(m["macro_precision_undefined_classes"], 0)

    def test_fast_macro_f1_matches_reference(self):
        rng = np.random.default_rng(1)
        t, p = rng.integers(0, 34, 800), rng.integers(0, 34, 800)
        self.assertAlmostEqual(macro_f1_fast(t, p, 34), classification_metrics(t.tolist(), p.tolist(), CLASS_NAMES)["macro_f1"])


@unittest.skipUnless(NEW.exists(), "new checkpoint not trained yet")
class NewCheckpointCompatibilityTests(unittest.TestCase):
    def test_loads_with_backend_loader_and_same_classes(self):
        new = load_classifier(NEW, torch.device("cpu"))
        old = load_classifier(PRODUCTION, torch.device("cpu"))
        self.assertEqual(new.class_names, old.class_names)
        self.assertEqual(new.image_size, 224)
        self.assertEqual(new.model.classifier[1].out_features, 34)


if __name__ == "__main__":
    unittest.main()
