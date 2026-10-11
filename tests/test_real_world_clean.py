"""Tests for evaluation/clean_real_world.py (cleaned real-world manifest).

Synthetic reports and files live in temporary directories; no model is run.
The last test reads the real reports and review sheet (read-only) and writes
only to a temporary folder.

Run from the project root:
    .venv-gpu\\Scripts\\python.exe -B -m unittest tests.test_real_world_clean -v
"""
import contextlib
import csv
import hashlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import clean_real_world  # noqa: E402
from evaluation.evaluate_real_world import image_manifest, metrics_for  # noqa: E402

NAMES = ["idli", "masala_dosa", "sushi"]
COLUMNS = ["path", "issue", "related", "audit_finding", "default_action", "decision", "reviewer", "notes"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class CleanManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        # (path, label, content, old prediction, new prediction)
        images = [
            ("rw/sushi/a.jpg", "sushi", b"S", "sushi", "sushi"),
            ("rw/sushi/a (1).jpg", "sushi", b"S", "sushi", "sushi"),
            ("rw/sushi/a (2).jpg", "sushi", b"S", "sushi", "sushi"),
            ("rw/sushi/b.jpg", "sushi", b"B", "idli", "sushi"),
            ("rw/idli/x.jpg", "idli", b"X", "sushi", "sushi"),
            ("rw/masala_dosa/x.jpg", "masala_dosa", b"X", "sushi", "sushi"),
            ("rw/idli/c.jpg", "idli", b"C", "idli", "idli"),
            ("rw/idli/overlap.jpg", "idli", b"O", "idli", "idli"),
            ("rw/masala_dosa/d.jpg", "masala_dosa", b"D", "idli", "masala_dosa"),
            ("rw/masala_dosa/flagged.jpg", "masala_dosa", b"F", "idli", "idli"),
        ]
        for path, _, content, _, _ in images:
            (self.tmp / path).parent.mkdir(parents=True, exist_ok=True)
            (self.tmp / path).write_bytes(content)
        quality = {
            "min_images_per_class": 10,
            "provenance": {"present": False, "undocumented_images": len(images)},
            "exact_duplicates_within": [
                {"paths": ["rw/sushi/a (1).jpg", "rw/sushi/a (2).jpg", "rw/sushi/a.jpg"], "conflicting_labels": False},
                {"paths": ["rw/idli/x.jpg", "rw/masala_dosa/x.jpg"], "conflicting_labels": True},
            ],
            "exact_overlaps": [],
            "near_duplicate_overlaps": [{"path": "rw/idli/overlap.jpg", "label": "idli", "matches": [
                {"source": "processed/train", "path": "data/processed/train/idli/1.jpg", "distance": 0}]}],
        }
        self.reports = {}
        for tag, column in (("old", 3), ("new", 4)):
            predictions = [{"path": i[0], "actual": i[1], "sha256": hashlib.sha256(i[2]).hexdigest(),
                            "predicted": i[column], "correct": i[column] == i[1], "confidence": 0.5}
                           for i in images]
            report = {"report_type": "real_world_evaluation", "status": "ok",
                      "checkpoint": {"path": f"{tag}.pth", "sha256": tag * 8},
                      "class_mapping": [{"index": i, "name": n} for i, n in enumerate(NAMES)],
                      "image_manifest": image_manifest(predictions), "quality": quality,
                      "predictions": predictions}
            self.reports[tag] = self.tmp / f"{tag}.json"
            self.reports[tag].write_text(json.dumps(report), encoding="utf-8")
        self.review = self.tmp / "review.csv"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write_review(self, rows):
        with open(self.review, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            writer.writeheader()
            for row in rows:
                writer.writerow({c: row.get(c, "") for c in COLUMNS})

    def clean(self):
        return clean_real_world.clean(self.reports["old"], self.reports["new"], self.review, root=self.tmp)

    def test_default_rules_are_conservative_and_provisional(self):
        self.write_review([{"path": "rw/masala_dosa/flagged.jpg", "issue": "label_uncertain", "default_action": "exclude"}])
        result = self.clean()
        excluded = {e["path"]: e for e in result["exclusions"]}
        kept = {k["path"] for k in result["kept"]}
        self.assertIn("rw/sushi/a.jpg", kept)  # the copy without a " (n)" suffix
        self.assertEqual(excluded["rw/sushi/a (1).jpg"]["decided_by"], "rule")
        self.assertIn("rw/sushi/a (2).jpg", excluded)
        for path in ("rw/idli/x.jpg", "rw/masala_dosa/x.jpg"):  # label never chosen automatically
            self.assertEqual(excluded[path]["decided_by"], "rule: pending review")
        self.assertEqual(excluded["rw/idli/overlap.jpg"]["decided_by"], "rule: pending review")
        self.assertEqual(excluded["rw/masala_dosa/flagged.jpg"]["decided_by"], "audit default: pending review")
        self.assertEqual(kept, {"rw/sushi/a.jpg", "rw/sushi/b.jpg", "rw/idli/c.jpg", "rw/masala_dosa/d.jpg"})
        self.assertEqual(result["manifest_status"], "provisional")
        self.assertEqual(len(result["pending_review"]), 3)

    def test_metrics_cover_only_kept_images_and_match_for_both_models(self):
        self.write_review([])
        result = self.clean()
        kept = {k["path"] for k in result["kept"]}
        for tag in ("old", "new"):
            report = json.loads(self.reports[tag].read_text(encoding="utf-8"))
            expected = metrics_for([p for p in report["predictions"] if p["path"] in kept], NAMES)
            self.assertEqual(result["overall"][tag]["correct"], expected["correct"])
            self.assertAlmostEqual(result["overall"][tag]["macro_f1"], expected["macro_f1"])
            self.assertEqual(result["overall"][tag]["total"], len(kept))
        self.assertEqual(result["image_manifest"]["images"], len(kept))
        self.assertEqual(result["paired_test"]["old_wrong_new_right"], 2)  # b.jpg and d.jpg

    def test_reviewer_decisions_resolve_conflicts_and_overlaps(self):
        self.write_review([
            {"path": "rw/idli/x.jpg", "default_action": "exclude", "decision": "keep", "reviewer": "me"},
            {"path": "rw/masala_dosa/x.jpg", "default_action": "exclude", "decision": "exclude", "reviewer": "me"},
            {"path": "rw/idli/overlap.jpg", "default_action": "exclude", "decision": "keep", "reviewer": "me"},
        ])
        result = self.clean()
        kept = {k["path"] for k in result["kept"]}
        self.assertIn("rw/idli/x.jpg", kept)
        self.assertIn("rw/idli/overlap.jpg", kept)
        self.assertNotIn("rw/masala_dosa/x.jpg", kept)
        self.assertEqual(result["manifest_status"], "reviewed")
        self.assertEqual(result["pending_review"], [])

    def conflict_review(self, idli_decision, dosa_decision):
        """Review sheet where only the conflicting pair varies; the overlap is decided."""
        rows = [{"path": "rw/idli/overlap.jpg", "default_action": "exclude", "decision": "exclude", "reviewer": "me"}]
        for path, decision in (("rw/idli/x.jpg", idli_decision), ("rw/masala_dosa/x.jpg", dosa_decision)):
            rows.append({"path": path, "default_action": "exclude", "decision": decision,
                         "reviewer": "me" if decision else ""})
        self.write_review(rows)
        result = self.clean()
        excluded = {e["path"]: e for e in result["exclusions"]}
        conflict_notes = [n for n in result["pending_review"] if n.startswith("Conflicting labels")]
        return result, {k["path"] for k in result["kept"]}, excluded, conflict_notes

    def test_conflict_with_both_copies_explicitly_excluded_is_resolved(self):
        result, kept, excluded, notes = self.conflict_review("exclude", "exclude")
        for path in ("rw/idli/x.jpg", "rw/masala_dosa/x.jpg"):
            self.assertNotIn(path, kept)
            self.assertEqual(excluded[path]["decided_by"], "reviewer")
            self.assertIn("reviewer excluded every copy", excluded[path]["reason"])
        self.assertEqual(notes, [])
        self.assertEqual(result["manifest_status"], "reviewed")

    def test_conflict_with_exactly_one_copy_kept_is_resolved(self):
        result, kept, excluded, notes = self.conflict_review("", "keep")
        self.assertIn("rw/masala_dosa/x.jpg", kept)
        self.assertEqual(excluded["rw/idli/x.jpg"]["decided_by"], "reviewer")
        self.assertEqual(notes, [])
        self.assertEqual(result["manifest_status"], "reviewed")

    def test_partially_decided_conflict_stays_provisional(self):
        result, kept, excluded, notes = self.conflict_review("exclude", "")
        self.assertFalse({"rw/idli/x.jpg", "rw/masala_dosa/x.jpg"} & kept)
        self.assertEqual(excluded["rw/masala_dosa/x.jpg"]["decided_by"], "rule: pending review")
        self.assertEqual(len(notes), 1)
        self.assertEqual(result["manifest_status"], "provisional")

    def test_conflict_without_reviewer_decisions_stays_provisional(self):
        result, kept, excluded, notes = self.conflict_review("", "")
        self.assertFalse({"rw/idli/x.jpg", "rw/masala_dosa/x.jpg"} & kept)
        for path in ("rw/idli/x.jpg", "rw/masala_dosa/x.jpg"):
            self.assertEqual(excluded[path]["decided_by"], "rule: pending review")
        self.assertEqual(len(notes), 1)
        self.assertEqual(result["manifest_status"], "provisional")

    def test_two_kept_copies_with_different_labels_stay_excluded(self):
        self.write_review([
            {"path": "rw/idli/x.jpg", "default_action": "exclude", "decision": "keep", "reviewer": "me"},
            {"path": "rw/masala_dosa/x.jpg", "default_action": "exclude", "decision": "keep", "reviewer": "me"},
        ])
        result = self.clean()
        kept = {k["path"] for k in result["kept"]}
        self.assertFalse({"rw/idli/x.jpg", "rw/masala_dosa/x.jpg"} & kept)
        self.assertEqual(result["manifest_status"], "provisional")

    def test_invalid_review_sheets_are_rejected(self):
        cases = {
            "needs a reviewer": [{"path": "rw/idli/c.jpg", "default_action": "keep", "decision": "exclude"}],
            "not an evaluated image": [{"path": "rw/idli/missing.jpg", "default_action": "keep"}],
            "decision must be": [{"path": "rw/idli/c.jpg", "default_action": "keep", "decision": "relabel:sushi",
                                  "reviewer": "me"}],
            "listed twice": [{"path": "rw/idli/c.jpg", "default_action": "keep"},
                             {"path": "rw/idli/c.jpg", "default_action": "keep"}],
        }
        for message, rows in cases.items():
            with self.subTest(message=message):
                self.write_review(rows)
                with self.assertRaisesRegex(ValueError, message):
                    self.clean()

    def test_changed_or_missing_image_stops_the_build(self):
        self.write_review([])
        (self.tmp / "rw/idli/c.jpg").write_bytes(b"edited")
        with self.assertRaisesRegex(ValueError, "changed since"):
            self.clean()
        (self.tmp / "rw/idli/c.jpg").unlink()
        with self.assertRaisesRegex(ValueError, "no longer exists"):
            self.clean()


class RealReportsTests(unittest.TestCase):
    OLD = ROOT / "reports" / "real_world" / "real_world_evaluation.json"
    NEW = ROOT / "reports" / "real_world_v2" / "real_world_evaluation.json"
    REVIEW = ROOT / "data" / "real_world_review" / "manual_review.csv"

    def test_cli_writes_only_to_its_output_dir(self):
        if not (self.OLD.is_file() and self.NEW.is_file() and self.REVIEW.is_file()):
            self.skipTest("real-world reports or review sheet not present")
        before = {p: digest(p) for p in (self.OLD, self.NEW, self.REVIEW)}
        out = Path(tempfile.mkdtemp())
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(clean_real_world.main(["--output-dir", str(out)]), 0)
            self.assertTrue((out / "clean_manifest.csv").is_file())
            self.assertTrue((out / "exclusions.csv").is_file())
            result = json.loads((out / "clean_evaluation.json").read_text(encoding="utf-8"))
            self.assertEqual(len(result["kept"]) + len(result["exclusions"]), result["source_manifest"]["images"])
        finally:
            shutil.rmtree(out, ignore_errors=True)
        self.assertEqual(before, {p: digest(p) for p in before})

    def test_cli_refuses_to_write_into_original_report_or_data_folders(self):
        for target in (self.OLD.parent, ROOT / "data" / "real_world_review"):
            with self.subTest(target=target), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    clean_real_world.main(["--output-dir", str(target)])


if __name__ == "__main__":
    unittest.main()
