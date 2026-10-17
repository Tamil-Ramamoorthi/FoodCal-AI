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
    def __init__(self, data_dir, env=None):
        self.data_dir = data_dir
        self.extra_env = env or {}
        self.port = free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        self.process = None

    def start(self):
        env = {**os.environ, "FOODCAL_DATA_DIR": str(self.data_dir), "PYTHONDONTWRITEBYTECODE": "1"}
        env.pop("FOODCAL_ADMIN_TOKEN", None)  # only what a test sets
        env.update(self.extra_env)
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

    def upload(self, filename, content, content_type="image/jpeg", headers=None):
        boundary = uuid.uuid4().hex
        body = (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
            f"filename=\"{filename}\"\r\nContent-Type: {content_type}\r\n\r\n"
        ).encode() + content + f"\r\n--{boundary}--\r\n".encode()
        status, response_headers, raw = self.request(
            "POST", "/api/predict", body,
            {**(headers or {}), "Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        return status, response_headers, json.loads(raw)

    def predict(self, food, headers=None):
        content = image_path(food).read_bytes()
        status, _, result = self.upload(f"{food}.jpg", content, headers=headers)
        return status, result, content


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

    def test_11_clear_all_requires_confirmation(self):
        status, before = self.server.json("GET", "/api/history")
        for path in ("/api/history", "/api/history?confirm=yes", "/api/history?confirm=ALL"):
            with self.subTest(path=path):
                status, body = self.server.json("DELETE", path)
                self.assertEqual(status, 400)
                self.assertIn("Nothing was deleted", body["detail"])
        self.assertEqual(self.server.json("GET", "/api/history"), (200, before))

        status, headers, _ = self.server.request("OPTIONS", "/api/history", headers={
            "Origin": "http://localhost:5173", "Access-Control-Request-Method": "DELETE",
        })
        self.assertEqual(status, 200)
        self.assertEqual(headers.get("access-control-allow-origin"), "http://localhost:5173")
        self.assertIn("DELETE", headers.get("access-control-allow-methods", ""))
        status, headers, _ = self.server.request("OPTIONS", "/api/history", headers={
            "Origin": "https://unknown.example", "Access-Control-Request-Method": "DELETE",
        })
        self.assertNotIn("access-control-allow-origin", headers, "other sites cannot call it from a browser")

    def test_12_clear_all_history(self):
        checkpoint = ROOT / "models" / "efficientnet_b0" / "food_efficientnet_b0_best.pth"
        checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        thumbnails = self.data_dir / "thumbnails"
        unrelated = [self.data_dir / "notes.txt", thumbnails / "README.txt"]
        for path in unrelated:
            path.write_text("keep")
        status, before = self.server.json("GET", "/api/history")
        self.assertEqual(before["total"], 3)
        self.assertEqual(len(list(thumbnails.glob("*.jpg"))), 3)

        status, body = self.server.json("DELETE", "/api/history?confirm=all")
        self.assertEqual((status, body), (200, {"deleted_records": 3, "deleted_thumbnails": 3}))

        status, history = self.server.json("GET", "/api/history")
        self.assertEqual((status, history["items"], history["total"]), (200, [], 0))
        status, live = self.server.json("GET", "/api/evaluation/live")
        self.assertEqual(
            (live["total_predictions"], live["verified"], live["accuracy"], live["per_class"], live["most_predicted"]),
            (0, 0, None, [], None),
            "dashboard metrics come from the same table",
        )
        old_id = self.ids["pizza"]
        self.assertEqual(self.server.request("GET", f"/api/history/{old_id}/thumbnail")[0], 404)
        self.assertEqual(self.server.json("DELETE", f"/api/history/{old_id}")[0], 404)

        self.assertEqual(sorted(p.name for p in thumbnails.iterdir()), ["README.txt"])
        for path in unrelated:
            self.assertEqual(path.read_text(), "keep")
        self.assertTrue((self.data_dir / "foodcal.db").is_file(), "database file is kept")
        self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(), checkpoint_sha)

    def test_13_after_clearing_everything_else_keeps_working(self):
        status, body = self.server.json("DELETE", "/api/history?confirm=all")
        self.assertEqual((status, body), (200, {"deleted_records": 0, "deleted_thumbnails": 0}), "already empty")

        status, home = self.server.json("GET", "/")
        self.assertEqual((status, home["classes"], home["history_storage"], home["nutrition_reference"]), (200, 34, "ok", "ok"))
        status, classes = self.server.json("GET", "/api/classes")
        self.assertEqual((status, classes["count"]), (200, 34))
        self.assertEqual(self.server.json("GET", "/api/nutrition")[0], 200)
        status, nutrition = self.server.json("GET", "/api/nutrition/pizza?grams=100")
        self.assertEqual((status, nutrition["food_class"]), (200, "pizza"))
        status, evaluation = self.server.json("GET", "/api/evaluation")
        self.assertEqual((status, evaluation["num_classes"]), (200, 34))
        self.assertEqual(self.server.json("GET", "/api/evaluation/real-world")[0], 200)

        status, result, _ = self.server.predict("pizza")
        self.assertEqual(status, 200, "the model still predicts")
        self.assertTrue(result["saved"], "and history is still written")
        status, history = self.server.json("GET", "/api/history")
        self.assertEqual([r["id"] for r in history["items"]], [result["prediction_id"]])
        self.assertEqual(self.server.request("GET", history["items"][0]["thumbnail_url"])[0], 200)
        self.assertEqual(self.server.json("GET", "/api/evaluation/live")[1]["total_predictions"], 1)

    def test_14_remote_changes_are_refused_without_an_admin_token(self):
        # This server has no FOODCAL_ADMIN_TOKEN. X-Forwarded-For marks a
        # request as proxied, as cloudflared's requests are.
        status, history = self.server.json("GET", "/api/history")
        record_id = history["items"][0]["id"]
        remote = {"X-Forwarded-For": "203.0.113.9", "X-FoodCal-Admin-Token": "guess"}
        for method, path in (("DELETE", "/api/history?confirm=all"), ("DELETE", f"/api/history/{record_id}"),
                             ("DELETE", f"/api/history/{record_id}/feedback")):
            with self.subTest(method=method, path=path):
                status, _, raw = self.server.request(method, path, headers=remote)
                self.assertEqual(status, 403)
                self.assertIn("disabled for remote access", json.loads(raw)["detail"])
        self.assertEqual(self.server.json("GET", "/api/history"), (200, history), "nothing changed")


class RemoteAccessTests(unittest.TestCase):
    """Requests through a proxy such as the Cloudflare Tunnel need the admin
    token to change saved history; reading and predicting do not."""

    TOKEN = "test-admin-token-123"
    ORIGIN = "https://food-cal-ai-sigma.vercel.app"

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.server = Server(Path(cls._tmp.name) / "storage", env={"FOODCAL_ADMIN_TOKEN": cls.TOKEN})
        cls.server.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.stop()
        cls._tmp.cleanup()

    def remote(self, method, path, body=None, token=None, ip="203.0.113.7"):
        headers = {"X-Forwarded-For": ip, "Origin": self.ORIGIN}
        if token is not None:
            headers["X-FoodCal-Admin-Token"] = token
        if isinstance(body, dict):
            body = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        status, response_headers, raw = self.server.request(method, path, body, headers)
        return status, response_headers, json.loads(raw) if raw else None

    def test_changes_need_the_admin_token(self):
        status, result, _ = self.server.predict("pizza")
        record_id, predicted = result["prediction_id"], result["predicted_food"]
        changes = [
            ("PATCH", f"/api/history/{record_id}/feedback", {"actual_food": predicted}),
            ("DELETE", f"/api/history/{record_id}/feedback", None),
            ("DELETE", f"/api/history/{record_id}", None),
            ("DELETE", "/api/history?confirm=all", None),
        ]
        for method, path, body in changes:
            for token in (None, "", "wrong", self.TOKEN + "x", self.TOKEN.upper()):
                with self.subTest(method=method, path=path, token=token):
                    status, headers, reply = self.remote(method, path, body, token)
                    self.assertEqual(status, 403)
                    self.assertIn("valid admin token", reply["detail"])
                    self.assertEqual(headers.get("access-control-allow-origin"), self.ORIGIN,
                                     "the browser can read the refusal")
        status, history = self.server.json("GET", "/api/history")
        self.assertEqual((history["total"], history["items"][0]["verification_status"]), (1, "unverified"))

        # Reading works remotely without a token.
        for path in ("/", "/api/history", "/api/classes", "/api/nutrition", "/api/evaluation", "/api/evaluation/live"):
            with self.subTest(path=path):
                self.assertEqual(self.remote("GET", path)[0], 200)

        # With the token, each change goes through.
        status, _, record = self.remote("PATCH", changes[0][1], changes[0][2], self.TOKEN)
        self.assertEqual((status, record["verification_status"]), (200, "correct"))
        status, _, record = self.remote("DELETE", changes[1][1], token=self.TOKEN)
        self.assertEqual((status, record["verification_status"]), (200, "unverified"))
        status, _, body = self.remote("DELETE", changes[2][1], token=self.TOKEN)
        self.assertEqual((status, body), (200, {"deleted": record_id}))
        self.server.predict("samosa")
        status, _, body = self.remote("DELETE", changes[3][1], token=self.TOKEN)
        self.assertEqual((status, body["deleted_records"]), (200, 1))

    def test_browser_preflight_allows_the_token_header_from_the_deployed_frontend(self):
        status, headers, _ = self.server.request("OPTIONS", "/api/history", headers={
            "Origin": self.ORIGIN, "Access-Control-Request-Method": "DELETE",
            "Access-Control-Request-Headers": "x-foodcal-admin-token",
        })
        self.assertEqual(status, 200)
        self.assertEqual(headers.get("access-control-allow-origin"), self.ORIGIN)
        self.assertIn("x-foodcal-admin-token", headers.get("access-control-allow-headers", "").lower())

    def test_remote_predictions_are_rate_limited_per_client(self):
        content = image_path("idli").read_bytes()
        limited = {"X-Forwarded-For": "198.51.100.20"}
        for _ in range(20):
            status, _, _ = self.server.upload("idli.jpg", content, headers=limited)
            self.assertEqual(status, 200)
        status, headers, body = self.server.upload("idli.jpg", content, headers=limited)
        self.assertEqual(status, 429)
        self.assertEqual(headers.get("retry-after"), "60")
        self.assertIn("Too many predictions", body["detail"])
        self.assertEqual(self.server.upload("idli.jpg", content, headers={"X-Forwarded-For": "198.51.100.21"})[0], 200,
                         "other visitors are not affected")
        self.assertEqual(self.server.upload("idli.jpg", content)[0], 200, "local requests are not limited")
        self.server.json("DELETE", "/api/history?confirm=all")

    def test_invalid_uploads_are_rejected_without_saving(self):
        status, before = self.server.json("GET", "/api/history")
        too_big = b"\xff\xd8\xff" + b"0" * (10 * 1024 * 1024)
        cases = [
            ("big.jpg", too_big, "image/jpeg", 413),
            ("empty.jpg", b"", "image/jpeg", 400),
            ("notes.txt", b"hello", "text/plain", 400),
            ("broken.jpg", b"\xff\xd8\xffnot really a jpeg", "image/jpeg", 400),
            ("../../evil.jpg", b"not an image", "image/png", 400),
        ]
        for filename, content, content_type, expected in cases:
            with self.subTest(filename=filename):
                status, _, body = self.server.upload(filename, content, content_type)
                self.assertEqual(status, expected)
                self.assertIsInstance(body["detail"], str)
        self.assertEqual(self.server.json("GET", "/api/history")[1]["total"], before["total"])


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
        self.assertNotIn(self.data_dir.parent.name, body["detail"], "no local paths in responses")
        self.assertEqual(self.server.json("GET", "/api/evaluation/live")[0], 503)
        self.assertEqual(self.server.json("DELETE", "/api/history?confirm=all")[0], 503)

    def test_database_write_failure_during_prediction(self):
        self.server = Server(self.data_dir)
        self.server.start()
        os.chmod(self.data_dir / "foodcal.db", stat.S_IREAD)  # read-only
        status, result, _ = self.server.predict("pizza")
        self.assertEqual(status, 200)
        self.assert_unsaved_prediction(result)
        self.assertIn("could not be saved", result["storage_error"])
        self.assertNotIn(self.data_dir.parent.name, result["storage_error"], "no local paths in responses")
        status, history = self.server.json("GET", "/api/history")
        self.assertEqual((status, history["total"]), (200, 0), "nothing pretends to be saved")
        self.assertEqual(list((self.data_dir / "thumbnails").iterdir()), [], "no orphan thumbnail")

    def test_clear_history_database_failure_deletes_nothing(self):
        self.server = Server(self.data_dir)
        self.server.start()
        status, result, _ = self.server.predict("pizza")
        self.assertTrue(result["saved"])
        db = self.data_dir / "foodcal.db"
        os.chmod(db, stat.S_IREAD)  # read-only
        try:
            status, body = self.server.json("DELETE", "/api/history?confirm=all")
        finally:
            os.chmod(db, stat.S_IREAD | stat.S_IWRITE)
        # A generic message: no SQL, file paths or exception text.
        self.assertEqual((status, body), (500, {"detail": "The history database could not be accessed."}))
        status, history = self.server.json("GET", "/api/history")
        self.assertEqual([r["id"] for r in history["items"]], [result["prediction_id"]])
        self.assertEqual(self.server.request("GET", history["items"][0]["thumbnail_url"])[0], 200)


if __name__ == "__main__":
    unittest.main()
