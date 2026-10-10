"""Offline CLI receipt admission; no models, GPU or native binaries."""
from contextlib import ExitStack
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import application
import cpu_cli
import prepare


class Qualification(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.stack.enter_context(patch.object(prepare, "ROOT", self.root))
        for name in ("Cargo.toml", "Cargo.lock", "tests/cli.rs", "src/lib.rs", "src/bin/toklet.rs"):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name)
        fixture = {"cases": {name: {"tokens": [index + 1, index + 2]} for index, name in enumerate(cpu_cli.CASE_NAMES)},
                   "argmax": {name: [11 + index] for index, name in enumerate(cpu_cli.CASE_NAMES)}}
        fixture_path = self.root / "tests/fixtures/qwen-v2/manifest.json"
        fixture_path.parent.mkdir(parents=True)
        fixture_path.write_text(json.dumps(fixture))
        fixture_sha = prepare.digest(fixture_path)
        self.stack.enter_context(patch.object(cpu_cli, "FIXTURE_SHA", fixture_sha))
        numerical_path = self.root / "numerical.json"
        numerical_path.write_text("{}")
        self.worker = self.stack.enter_context(patch.object(cpu_cli.cpu, "qualified_cpu_worker", return_value=(
            Path("unused-worker"), "fixture compiler", {"rustflags": None})))
        self.binary = self.root / "artifacts/cli/release/toklet"
        self.binary.parent.mkdir(parents=True)
        self.binary.write_bytes(b"fixture binary")
        cases = []
        for name in cpu_cli.CASE_NAMES:
            count = len(fixture["cases"][name]["tokens"])
            token_ids = [fixture["argmax"][name][-1], 23]
            events = []
            for index, token in enumerate(token_ids):
                events += [{"event": "token", "index": 0, "token_id": token, "text": "x"},
                           {"event": "done", "generation": {"reason": {"kind": "output_limit"},
                            "reused_tokens": 0 if index == 0 else count, "prefill_tokens": count if index == 0 else 1,
                            "sampled_tokens": 1, "emitted_tokens": 1, "history_tokens": count + index + 1,
                            "evaluated_tokens": count + index, "pending_tokens": 1}}]
            events.append({"event": "reset"})
            cases.append({"name": name, "tokens": fixture["cases"][name]["tokens"],
                          "expected_first_token": token_ids[0], "token_ids": token_ids, "semantic_events": events})
        self.receipt = {"format": cpu_cli.FORMAT, "completed": True,
                        "source_sha256": {name: prepare.digest(self.root / name) for name in cpu_cli.compiled_sources()},
                        "numerical_qualification": {"path": "numerical.json", "sha256": prepare.digest(numerical_path)},
                        "compiler": "fixture compiler", "rustflags": None,
                        "build_commands": [["copy cache"], ["build cli"]],
                        "fixture_manifest_sha256": fixture_sha, "binary_path": "artifacts/cli/release/toklet",
                        "binary_sha256": prepare.digest(self.binary), "runs": {}}
        self.receipt["checks"] = [{"command": command, "exit_code": 0, "timed_out": False} for command in (
            ["cargo", "fmt", "--check"],
            ["cargo", "clippy", "--locked", "--offline", "--all-targets", "--", "-D", "warnings"],
            ["cargo", "test", "--locked", "--offline"], ["build cli"])]
        for role, projection in cpu_cli.MODES.items():
            ready = {"event": "ready", "backend": "cpu", "context_capacity": 2048, "memory_admitted_bytes": 64 * 1024**2}
            if role != "frozen_scalar":
                ready["cpu_projection"] = projection
            self.receipt["runs"][role] = {"passed": True, "exit_code": 0, "cpu_projection": projection,
                                           "ready": ready, "physical_process_charge_peak_bytes": 32 * 1024**2,
                                           "cases": copy.deepcopy(cases)}
        self.path = self.root / "cli.json"

    def write(self, receipt):
        self.path.write_text(json.dumps(receipt))

    def test_complete_source_build_reference_and_session_binding(self):
        self.write(self.receipt)
        binary, receipt = cpu_cli.qualified_cli(self.path)
        self.assertEqual(binary, self.binary)
        self.assertEqual(receipt, self.receipt)
        self.worker.assert_called_once_with(self.root / "numerical.json")

    def test_incomplete_identity_inventory_and_builds_fail(self):
        mutations = [
            lambda r: r.update(completed=False),
            lambda r: r["source_sha256"].pop("src/lib.rs"),
            lambda r: r["source_sha256"].update({"src/lib.rs": "wrong"}),
            lambda r: r["numerical_qualification"].update(sha256="wrong"),
            lambda r: r.update(compiler="different compiler"),
            lambda r: r.update(rustflags="different flags"),
            lambda r: r["checks"].pop(),
            lambda r: r["checks"][0].update(exit_code=1),
            lambda r: r["checks"][0].update(timed_out=True),
            lambda r: r.update(fixture_manifest_sha256="wrong"),
            lambda r: r.update(binary_sha256="wrong"),
            lambda r: r.update(binary_path="outside-artifact"),
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                receipt = copy.deepcopy(self.receipt)
                mutate(receipt)
                self.write(receipt)
                with self.assertRaises(ValueError):
                    cpu_cli.qualified_cli(self.path)

    def test_incomplete_modes_tokens_state_and_ownership_fail(self):
        mutations = [
            lambda r: r["runs"].pop("tiled"),
            lambda r: r["runs"]["tiled"].update(passed=False),
            lambda r: r["runs"]["tiled"]["ready"].update(cpu_projection="scalar"),
            lambda r: r["runs"]["tiled"].update(physical_process_charge_peak_bytes=7 * 1024**3),
            lambda r: r["runs"]["tiled"].update(cases=[]),
            lambda r: r["runs"]["tiled"]["cases"][0]["token_ids"].__setitem__(0, 99),
            lambda r: r["runs"]["tiled"]["cases"][0].update(semantic_events=[]),
            lambda r: r["runs"]["tiled"]["cases"][0]["semantic_events"][1]["generation"].update(pending_tokens=0),
            lambda r: r["runs"]["tiled"]["ready"].update(memory_admitted_bytes=65 * 1024**2),
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                receipt = copy.deepcopy(self.receipt)
                mutate(receipt)
                self.write(receipt)
                with self.assertRaises(ValueError):
                    cpu_cli.qualified_cli(self.path)

    def test_legacy_scalar_fallback_and_qualified_mode_acknowledgement(self):
        legacy = {"event": "ready", "backend": "cpu", "context_capacity": 2048}
        cpu_cli.check_ready(legacy, "scalar", False)
        cpu_cli.check_statistics({}, "scalar", False)
        with self.assertRaises(ValueError):
            cpu_cli.check_ready(legacy, "tiled", True)
        with self.assertRaises(ValueError):
            cpu_cli.check_statistics({}, "scalar", True)
        with self.assertRaises(ValueError):
            cpu_cli.check_statistics({"cpu_projection": "scalar"}, "tiled", True)
        with self.assertRaises(ValueError):
            cpu_cli.check_ready({**legacy, "cpu_projection": "tiled", "context_capacity": 1024}, "tiled", True)

    def test_application_rejects_unqualified_tiled_before_model_reads(self):
        with patch("sys.argv", ["application", "--output", "unused", "--cpu-projection", "tiled"]), \
                patch.object(application.prepare, "verify") as verify:
            with self.assertRaisesRegex(ValueError, "require completed CLI qualification"):
                application.main()
            verify.assert_not_called()

    def test_application_rejects_supplied_incomplete_receipt_before_contract_or_model_reads(self):
        self.write({"format": cpu_cli.FORMAT, "completed": False})
        with patch("sys.argv", ["application", "--output", str(self.root / "attempt.json"),
                                "--cpu-projection", "tiled", "--cpu-qualification", str(self.path)]), \
                patch.object(application.prepare, "verify") as verify, \
                patch.object(application.prepare, "validate_source") as source:
            with self.assertRaisesRegex(ValueError, "CLI qualification is incomplete"):
                application.main()
            verify.assert_not_called()
            source.assert_not_called()


if __name__ == "__main__":
    unittest.main()
