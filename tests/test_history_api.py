"""End-to-end tests for prediction history. Each test class starts a real
uvicorn server with FOODCAL_DATA_DIR pointing at a temporary directory,
so the real storage/ history is never touched.

Run from the project root:
    .venv-gpu\\Scripts\\python.exe -B -m unittest discover -s tests -v
"""
import hashlib
import io
import json
import os
import socket
import stat
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TEST_IMAGES = ROOT / "data" / "processed" / "test"


def image_path(food):
    return TEST_IMAGES / food / "000000.jpg"


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class Server:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.port = free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        self.process = None

    def start(self):
        env = {**os.environ, "FOODCAL_DATA_DIR": str(self.data_dir), "PYTHONDONTWRITEBYTECODE": "1"}
        self.process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "backend.app.main:app",
             "--host", "127.0.0.1", "--port", str(self.port)],
            cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        deadline = time.time() + 180
        while time.time() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError("Backend exited during startup.")
            try:
                with urllib.request.urlopen(self.base + "/", timeout=2):
                    return
            except OSError:
                time.sleep(0.5)
        raise RuntimeError("Backend did not start in time.")

    def stop(self):
        if self.process and self.process.poll() is None:
            if os.name == "nt":
                # The venv launcher spawns a child interpreter; kill the tree.
                subprocess.run(["taskkill", "/PID", str(self.process.pid), "/T", "/F"],
                               capture_output=True, check=False)
            else:
                self.process.terminate()
            self.process.wait(timeout=30)
        self.process = None

    def request(self, method, path, body=None, headers=None):
        data = None
        headers = dict(headers or {})
        if isinstance(body, dict):
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        elif body is not None:
            data = body
        req = urllib.request.Request(self.base + path, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                return response.status, dict(response.headers), response.read()
        except urllib.error.HTTPError as error:
            return error.code, dict(error.headers), error.read()

    def json(self, method, path, body=None):
        status, _, raw = self.request(method, path, body)
        return status, json.loads(raw) if raw else None

    def predict(self, food):
        content = image_path(food).read_bytes()
        boundary = uuid.uuid4().hex
        body = (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
            f"filename=\"{food}.jpg\"\r\nContent-Type: image/jpeg\r\n\r\n"
        ).encode() + content + f"\r\n--{boundary}--\r\n".encode()
        status, _, raw = self.request(
            "POST", "/api/predict", body,
            {"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        return status, json.loads(raw), content


class HistoryApiTests(unittest.TestCase):
    """Ordered end-to-end flow against one temporary database."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.data_dir = Path(cls._tmp.name) / "storage"
        cls.server = Server(cls.data_dir)
        cls.server.start()
        cls.ids = {}

    @classmethod
    def tearDownClass(cls):
        cls.server.stop()
        cls._tmp.cleanup()

    def test_01_database_initialized_and_history_empty(self):
        self.assertTrue((self.data_dir / "foodcal.db").is_file())
        status, home = self.server.json("GET", "/")
        self.assertEqual((status, home["history_storage"]), (200, "ok"))
        status, history = self.server.json("GET", "/api/history")
        self.assertEqual(status, 200)
        self.assertEqual((history["items"], history["total"]), ([], 0))
        status, live = self.server.json("GET", "/api/evaluation/live")
        self.assertEqual(status, 200)
        self.assertEqual((live["total_predictions"], live["verified"], live["accuracy"]), (0, 0, None))

    def test_02_prediction_is_saved_with_existing_fields(self):
        status, result, content = self.server.predict("pizza")
        self.assertEqual(status, 200)
        for key in ("filename", "predicted_food", "confidence", "top_predictions", "device", "note"):
            self.assertIn(key, result)
        self.assertTrue(result["saved"])
        self.assertIsNone(result["storage_error"])
        self.assertRegex(result["prediction_id"], r"^[0-9a-f]{32}$")
        self.__class__.ids["pizza"] = result["prediction_id"]
        self.__class__.pizza_prediction = result

        status, history = self.server.json("GET", "/api/history")
        self.assertEqual(history["total"], 1)
        record = history["items"][0]
        self.assertEqual(record["id"], result["prediction_id"])
        self.assertEqual(record["original_filename"], "pizza.jpg")
        self.assertEqual(record["image_sha256"], hashlib.sha256(content).hexdigest())
        self.assertEqual(record["predicted_food"], result["predicted_food"])
        self.assertEqual(record["confidence"], result["confidence"])
        self.assertEqual(record["top_predictions"], result["top_predictions"])
        self.assertEqual(record["device"], result["device"])
        self.assertEqual(record["verification_status"], "unverified")
        self.assertIsNone(record["actual_food"])
        self.assertTrue(record["created_at"].endswith("+00:00"))
        self.assertEqual(record["thumbnail_url"], f"/api/history/{record['id']}/thumbnail")

    def test_03_thumbnail_is_a_small_reencoded_jpeg(self):
        status, headers, raw = self.server.request("GET", f"/api/history/{self.ids['pizza']}/thumbnail")
        self.assertEqual(status, 200)
        self.assertEqual(headers.get("content-type"), "image/jpeg")
        with Image.open(io.BytesIO(raw)) as thumb:
            self.assertEqual(thumb.format, "JPEG")
            self.assertLessEqual(max(thumb.size), 256)
        stored = sorted(p.name for p in (self.data_dir / "thumbnails").iterdir())
        sha = hashlib.sha256(image_path("pizza").read_bytes()).hexdigest()
        self.assertEqual(stored, [f"{sha}.jpg"], "only hash-named thumbnails, no uploaded originals")

    def test_04_feedback_correct(self):
        predicted = self.pizza_prediction["predicted_food"]
        status, record = self.server.json("PATCH", f"/api/history/{self.ids['pizza']}/feedback", {"actual_food": predicted})
        self.assertEqual(status, 200)
        self.assertEqual((record["verification_status"], record["actual_food"]), ("correct", predicted))
        self.assertIsNotNone(record["verified_at"])

    def test_05_invalid_labels_are_rejected(self):
        path = f"/api/history/{self.ids['pizza']}/feedback"
        for label in ["pasta", "Pizza", "pizza ", "kadai paneer"]:
            with self.subTest(label=label):
                status, body = self.server.json("PATCH", path, {"actual_food": label})
                self.assertEqual(status, 422)
                self.assertIn("is not one of the 34 food classes", body["detail"])
        self.assertEqual(self.server.json("PATCH", path, {"actual_food": ""})[0], 422)
        self.assertEqual(self.server.json("PATCH", path, {})[0], 422)
        self.assertEqual(self.server.json("PATCH", f"/api/history/{'0' * 32}/feedback", {"actual_food": "pizza"})[0], 404)
        self.assertEqual(self.server.json("PATCH", "/api/history/not-an-id/feedback", {"actual_food": "pizza"})[0], 422)
        # The rejected attempts left the stored feedback unchanged.
        status, history = self.server.json("GET", "/api/history")
        self.assertEqual(history["items"][0]["verification_status"], "correct")

    def test_06_feedback_update_and_clear(self):
        path = f"/api/history/{self.ids['pizza']}/feedback"
        actual = "samosa" if self.pizza_prediction["predicted_food"] != "samosa" else "burger"
        status, record = self.server.json("PATCH", path, {"actual_food": actual})
        self.assertEqual((status, record["verification_status"], record["actual_food"]), (200, "incorrect", actual))
        status, live = self.server.json("GET", "/api/evaluation/live")
        self.assertEqual((live["verified"], live["correct"], live["incorrect"], live["accuracy"]), (1, 0, 1, 0.0))

        status, record = self.server.json("DELETE", path)
        self.assertEqual((status, record["verification_status"], record["actual_food"]), (200, "unverified", None))
        status, live = self.server.json("GET", "/api/evaluation/live")
        self.assertEqual((live["verified"], live["accuracy"]), (0, None))

        # Leave a verified record for the restart test.
        self.server.json("PATCH", path, {"actual_food": actual})

    def test_07_pagination_order_and_search(self):
        for food in ("samosa", "burger", "Hot Dog"):
            status, result, _ = self.server.predict(food)
            self.assertTrue(result["saved"])
            self.__class__.ids[food] = result["prediction_id"]
        status, page1 = self.server.json("GET", "/api/history?limit=2&offset=0")
        status, page2 = self.server.json("GET", "/api/history?limit=2&offset=2")
        self.assertEqual(page1["total"], 4)
        self.assertEqual([r["id"] for r in page1["items"]], [self.ids["Hot Dog"], self.ids["burger"]])
        self.assertEqual([r["id"] for r in page2["items"]], [self.ids["samosa"], self.ids["pizza"]])
        self.assertEqual(self.server.json("GET", "/api/history?limit=2&offset=50")[1]["items"], [])
        self.assertEqual(self.server.json("GET", "/api/history?limit=0")[0], 422)
        self.assertEqual(self.server.json("GET", "/api/history?limit=101")[0], 422)
        self.assertEqual(self.server.json("GET", "/api/history?offset=-1")[0], 422)
        status, found = self.server.json("GET", "/api/history?q=hot%20dog")
        self.assertIn(self.ids["Hot Dog"], [r["id"] for r in found["items"]])

    def test_08_delete_record(self):
        record_id = self.ids["burger"]
        sha = hashlib.sha256(image_path("burger").read_bytes()).hexdigest()
        thumb = self.data_dir / "thumbnails" / f"{sha}.jpg"
        self.assertTrue(thumb.exists())
        status, body = self.server.json("DELETE", f"/api/history/{record_id}")
        self.assertEqual((status, body), (200, {"deleted": record_id}))
        self.assertFalse(thumb.exists(), "unused thumbnail removed")
        self.assertEqual(self.server.json("DELETE", f"/api/history/{record_id}")[0], 404)
        self.assertEqual(self.server.request("GET", f"/api/history/{record_id}/thumbnail")[0], 404)
        status, history = self.server.json("GET", "/api/history")
        self.assertEqual(history["total"], 3)
        self.assertNotIn(record_id, [r["id"] for r in history["items"]])

    def test_09_cors_allows_patch_and_delete(self):
        for method in ("PATCH", "DELETE"):
            status, headers, _ = self.server.request("OPTIONS", f"/api/history/{self.ids['pizza']}/feedback", headers={
                "Origin": "http://localhost:5174", "Access-Control-Request-Method": method,
                "Access-Control-Request-Headers": "content-type",
            })
            self.assertEqual(status, 200)
            self.assertIn(method, headers.get("access-control-allow-methods", ""))

    def test_10_history_persists_after_backend_restart(self):
        status, before = self.server.json("GET", "/api/history")
        status, live_before = self.server.json("GET", "/api/evaluation/live")
        self.server.stop()
        self.server.start()
        status, after = self.server.json("GET", "/api/history")
        status, live_after = self.server.json("GET", "/api/evaluation/live")
        self.assertEqual(after, before)
        self.assertEqual(live_after, live_before)
        pizza = next(r for r in after["items"] if r["id"] == self.ids["pizza"])
        self.assertEqual(pizza["verification_status"], "incorrect")
        self.assertEqual(self.server.request("GET", pizza["thumbnail_url"])[0], 200)


class StorageFailureTests(unittest.TestCase):
    """A storage failure must never be reported as a saved record."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self._tmp.name) / "storage"
        self.server = None

    def tearDown(self):
        if self.server:
            self.server.stop()
        db = self.data_dir / "foodcal.db"
        if db.is_file():
            os.chmod(db, stat.S_IREAD | stat.S_IWRITE)
        self._tmp.cleanup()

    def assert_unsaved_prediction(self, result):
        self.assertEqual(result["predicted_food"] in json.dumps(result["top_predictions"]), True)
        self.assertFalse(result["saved"])
        self.assertIsNone(result["prediction_id"])
        self.assertTrue(result["storage_error"])

    def test_storage_unavailable_at_startup(self):
        # A directory where the database file should be makes SQLite fail.
        (self.data_dir / "foodcal.db").mkdir(parents=True)
        self.server = Server(self.data_dir)
        self.server.start()
        status, home = self.server.json("GET", "/")
        self.assertEqual(home["history_storage"], "unavailable")
        status, result, _ = self.server.predict("pizza")
        self.assertEqual(status, 200, "prediction still works")
        self.assert_unsaved_prediction(result)
        status, body = self.server.json("GET", "/api/history")
        self.assertEqual(status, 503)
        self.assertIn("History storage is unavailable", body["detail"])
        self.assertEqual(self.server.json("GET", "/api/evaluation/live")[0], 503)

    def test_database_write_failure_during_prediction(self):
        self.server = Server(self.data_dir)
        self.server.start()
        os.chmod(self.data_dir / "foodcal.db", stat.S_IREAD)  # read-only
        status, result, _ = self.server.predict("pizza")
        self.assertEqual(status, 200)
        self.assert_unsaved_prediction(result)
        self.assertIn("could not be saved", result["storage_error"])
        status, history = self.server.json("GET", "/api/history")
        self.assertEqual((status, history["total"]), (200, 0), "nothing pretends to be saved")
        self.assertEqual(list((self.data_dir / "thumbnails").iterdir()), [], "no orphan thumbnail")


if __name__ == "__main__":
    unittest.main()
