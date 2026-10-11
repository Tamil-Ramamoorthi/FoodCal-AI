"""Focused tests for the inference pipeline: preprocessing parity with the
offline evaluation, class mapping, top-k ordering and the low-confidence
flag. API tests start a real uvicorn server with a temporary data
directory, so the real storage/ history is never touched.

Run from the project root:
    .venv-gpu\\Scripts\\python.exe -B -m unittest discover -s tests -v
"""
import io
import json
import os
import sys
import tempfile
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.app.models.classifier import build_eval_transform, load_classifier, preprocess_image  # noqa: E402
from test_history_api import Server  # noqa: E402

CHECKPOINT = ROOT / "models" / "efficientnet_b0" / "food_efficientnet_b0_best.pth"
PROCESSED = ROOT / "data" / "processed"
REPRESENTATIVE = ["pizza", "idli", "kulfi", "chicken_curry", "kadai_paneer", "samosa", "Hot Dog", "sushi"]
CPU = torch.device("cpu")


@contextmanager
def env(**values):
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key)
            else:
                os.environ[key] = value


def encode(image, fmt="JPEG", **options):
    buffer = io.BytesIO()
    image.save(buffer, fmt, **options)
    return buffer.getvalue()


class PreprocessingAndMappingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.classifier = load_classifier(CHECKPOINT, CPU)
        cls.transform = build_eval_transform(cls.classifier.image_size)

    def api_tensor(self, image):
        return preprocess_image(image, self.classifier.image_size, self.transform)

    def test_class_to_idx_matches_checkpoint_in_every_split(self):
        expected = {name: i for i, name in enumerate(self.classifier.class_names)}
        for split in ("train", "validation", "test"):
            with self.subTest(split=split):
                self.assertEqual(datasets.ImageFolder(PROCESSED / split).class_to_idx, expected)

    def test_api_tensor_equals_offline_evaluation_tensor(self):
        # training/evaluate_efficientnet.py: ImageFolder default loader + this transform.
        offline = datasets.ImageFolder(PROCESSED / "test", transform=transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]))
        for index in (0, 1500, len(offline) - 1):
            path, _ = offline.samples[index]
            with Image.open(path) as image, self.subTest(path=path):
                self.assertTrue(torch.equal(self.api_tensor(image), offline[index][0]))

    def test_exif_orientation_is_applied(self):
        with Image.open(PROCESSED / "test" / "pizza" / "000000.jpg") as image:
            upright = image.convert("RGB").resize((448, 300))
        exif = Image.Exif()
        exif[0x0112] = 6  # stored rotated; viewers rotate 90 degrees clockwise
        stored = Image.open(io.BytesIO(encode(upright.rotate(90, expand=True), exif=exif.tobytes())))
        self.assertEqual(stored.size, (300, 448))
        reference = Image.open(io.BytesIO(encode(upright.rotate(90, expand=True)))).rotate(-90, expand=True)
        self.assertLess((self.api_tensor(stored) - self.api_tensor(reference)).abs().max().item(), 1e-6)

    def test_non_rgb_inputs_become_3x224x224(self):
        with Image.open(PROCESSED / "test" / "samosa" / "000000.jpg") as image:
            base = image.convert("RGB")
        for mode in ("L", "RGBA", "P", "CMYK"):
            with self.subTest(mode=mode):
                self.assertEqual(tuple(self.api_tensor(base.convert(mode)).shape), (3, 224, 224))

    def test_top_k_matches_argsort_of_logits(self):
        with Image.open(PROCESSED / "test" / "kulfi" / "000000.jpg") as image:
            tensor = self.api_tensor(image).unsqueeze(0)
        with torch.inference_mode():
            logits = self.classifier.model(tensor)[0]
        probs = torch.softmax(logits, 0)
        top_p, top_i = torch.topk(probs, 3)
        self.assertEqual(top_i.tolist(), np.argsort(-logits.numpy())[:3].tolist())
        self.assertTrue(torch.all(top_p[:-1] >= top_p[1:]))
        self.assertAlmostEqual(probs.sum().item(), 1.0, places=5)


class PredictionApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.server = Server(Path(cls._tmp.name) / "storage")
        cls.server.start()
        cls.classifier = load_classifier(CHECKPOINT, CPU)

    @classmethod
    def tearDownClass(cls):
        cls.server.stop()
        cls._tmp.cleanup()

    def post(self, content, filename="image.jpg", server=None):
        boundary = uuid.uuid4().hex
        body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n"
                "Content-Type: image/jpeg\r\n\r\n").encode() + content + f"\r\n--{boundary}--\r\n".encode()
        status, _, raw = (server or self.server).request(
            "POST", "/api/predict", body, {"Content-Type": f"multipart/form-data; boundary={boundary}"})
        return status, json.loads(raw)

    def direct_top3(self, content):
        transform = build_eval_transform(224)
        with Image.open(io.BytesIO(content)) as image:
            tensor = preprocess_image(image, 224, transform).unsqueeze(0)
        with torch.inference_mode():
            probs = torch.softmax(self.classifier.model(tensor), 1)[0]
        p, i = torch.topk(probs, 3)
        return [(self.classifier.class_names[k], round(v * 100, 2)) for v, k in zip(p.tolist(), i.tolist())]

    def test_health_reports_default_threshold(self):
        status, home = self.server.json("GET", "/")
        self.assertEqual((status, home["low_confidence_threshold"]), (200, 50.0))

    def test_representative_images_match_direct_model_and_flag(self):
        for food in REPRESENTATIVE:
            content = (PROCESSED / "test" / food / "000001.jpg").read_bytes()
            with self.subTest(food=food):
                status, body = self.post(content, f"{food}.jpg")
                self.assertEqual(status, 200)
                got = [(p["food"], p["confidence"]) for p in body["top_predictions"]]
                expected = self.direct_top3(content)
                self.assertEqual([f for f, _ in got], [f for f, _ in expected])
                for (_, a), (_, b) in zip(got, expected):
                    # The server may run on CUDA; CPU vs CUDA kernels differ by
                    # ~0.15 percentage points on these images (measured).
                    self.assertAlmostEqual(a, b, delta=0.5)
                self.assertEqual(body["predicted_food"], got[0][0])
                self.assertEqual(body["low_confidence"], body["confidence"] < 50.0)
                self.assertEqual(body["low_confidence_threshold"], 50.0)
                self.assertTrue(body["saved"])  # history still works

    def test_out_of_class_image_still_gets_one_of_34_labels(self):
        noise = Image.fromarray(np.random.RandomState(0).randint(0, 255, (300, 400, 3), dtype=np.uint8))
        status, body = self.post(encode(noise), "noise.jpg")
        self.assertEqual(status, 200)
        self.assertIn(body["predicted_food"], self.classifier.class_names)
        self.assertEqual(len(body["top_predictions"]), 3)

    def test_threshold_is_configurable_and_validated(self):
        with env(FOODCAL_LOW_CONFIDENCE_THRESHOLD="99.5"):
            server = Server(Path(self._tmp.name) / "storage_threshold")
            server.start()
        try:
            status, home = server.json("GET", "/")
            self.assertEqual(home["low_confidence_threshold"], 99.5)
            content = (PROCESSED / "test" / "pizza" / "000001.jpg").read_bytes()
            status, body = self.post(content, server=server)
            self.assertEqual(body["low_confidence"], body["confidence"] < 99.5)
        finally:
            server.stop()
        for bad in ("abc", "-1", "150", "nan"):
            with self.subTest(value=bad), env(FOODCAL_LOW_CONFIDENCE_THRESHOLD=bad):
                server = Server(Path(self._tmp.name) / "storage_bad")
                with self.assertRaises(RuntimeError):
                    server.start()
                server.stop()


if __name__ == "__main__":
    unittest.main()
