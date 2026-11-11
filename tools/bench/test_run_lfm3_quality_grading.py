"""Verify admission and final grading-window identity without starting a guest."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_lfm3_quality_grading as subject
from lfm3_quality_inputs import INPUTS_SHA256, digest


class GradingWindowTests(unittest.TestCase):
    def setup_args(self, root, *, count=100):
        for profile, role in (("baseline", "stock"), ("candidate", "modified_parent")):
            folder = root / profile
            folder.mkdir()
            (folder / "result.json").write_text(json.dumps(dict(
                passed=True, role=role, first=0, end=count, tasks=count,
                input_reconciliation_sha256=INPUTS_SHA256, collection_sha256="1" * 64,
                decoded_sha256="2" * 64,
            )))
        (root / "reconciliation.json").write_text("{}")
        return argparse.Namespace(inputs=root, baseline=root / "baseline", candidate=root / "candidate",
                                  baseline_sha256=digest(root / "baseline/result.json"),
                                  candidate_sha256=digest(root / "candidate/result.json"),
                                  output=root / "artifacts/grade", roles=["stock", "modified_parent"],
                                  qualified_cpu_witness=False)

    def test_partial_decoded_role_cannot_start_window(self):
        with tempfile.TemporaryDirectory() as folder:
            args = self.setup_args(Path(folder).resolve(), count=1)
            with patch.object(subject.support, "supervise_window") as supervisor:
                with self.assertRaisesRegex(ValueError, "Complete independently decoded role"):
                    subject.run(args)
                supervisor.assert_not_called()
            self.assertFalse(args.output.exists())

    def test_only_matching_inner_decision_and_stopped_guest_qualify(self):
        for defect in (None, "identity", "restoration"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as folder:
                root = Path(folder).resolve()
                args = self.setup_args(root)
                (root / "artifacts").mkdir()
                client = root / "podman"
                client.write_bytes(b"fixture")
                inner = dict(completed=True, restored_stopped=True, pipe_lifetime_finished=True,
                             pair=args.roles, decoded_inputs=subject.admit_pair(args), decision={"complete": True})
                window = dict(exit_code=0, primary_error=None, cleanup_errors=[],
                              independent_stopped_verification=True, pid=33)
                if defect == "identity":
                    inner["decoded_inputs"]["candidate"]["decoded_tasks_sha256"] = "3" * 64
                if defect == "restoration":
                    window["independent_stopped_verification"] = False
                with (patch.object(subject, "ROOT", root),
                      patch.object(subject.runner, "PODMAN", str(client)),
                      patch.object(subject.runner, "checked_json", return_value={"podman_sha256": digest(client)}),
                      patch.object(subject.support, "supervise_window", return_value=window) as supervisor,
                      patch.object(subject, "group_alive", return_value=False),
                      patch.object(subject.os, "kill", side_effect=ProcessLookupError),
                      patch.object(subject, "read_inner_result", return_value=(inner, {"sha256": "4" * 64}))):
                    actual = subject.run(args)
                self.assertEqual(supervisor.call_args.args[3]["TZ"], "Europe/Moscow")
                self.assertEqual(actual["completed"], defect is None)
                self.assertEqual((args.output / "result.json").is_file(), True)


if __name__ == "__main__":
    unittest.main()
