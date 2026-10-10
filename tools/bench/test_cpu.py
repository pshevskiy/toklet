"""Offline checks for benchmark cache preparation and fixed workload shapes."""
import unittest
import tempfile
import hashlib
import json
import io
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch
import cpu
import prepare


class Requests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = prepare.load_json(prepare.CONTRACT / "manifest.json")
        cls.arrays = {name: prepare.read_tokens(prepare.CONTRACT / item["file"])
                      for name, item in cls.contract["arrays"].items()}

    def test_empty_cache_and_exact_capacity(self):
        for name, length in list(cpu.WORKLOADS.items())[:4]:
            prep, request, state = cpu.requests(name, self.arrays, self.contract)
            self.assertEqual(prep, [])
            self.assertEqual(state, "empty")
            self.assertEqual(request["action"], "reset")
            self.assertEqual(len(request["tokens"]), length)
            self.assertEqual(request["continuation"], self.arrays["continuation_128"])
            self.assertLessEqual(length + len(request["continuation"]), 2048)
        self.assertEqual(len(self.arrays["prompt_1921"]) + 128, 2049)

    def test_cache_workloads_prepare_the_declared_valid_prefix(self):
        for name, action, state in [("append", "append", "exact_prefix_append"),
                                    ("cached", "prompt", "fully_cached"),
                                    ("changed", "prompt", "changed_prefix_replay")]:
            prep, request, cache = cpu.requests(name, self.arrays, self.contract)
            self.assertEqual(prep, [{"action":"prepare", "tokens": self.arrays["prompt_512"]}])
            self.assertEqual((request["action"], cache), (action, state))
        self.assertEqual(cpu.requests("append", self.arrays, self.contract)[1]["tokens"], self.arrays["append_64"])
        self.assertEqual(cpu.requests("changed", self.arrays, self.contract)[1]["tokens"], self.arrays["changed_prompt_512"])

    def test_pending_token_is_staged_without_evaluation(self):
        prep, request, cache = cpu.requests("pending", self.arrays, self.contract)
        self.assertEqual(cache, "pending_sample")
        self.assertEqual(prep[-1], {"action":"stage", "tokens":[2077]})
        self.assertEqual(request["action"], "continue")
        self.assertEqual(request["tokens"], [])

    def test_interrupted_record_retry_never_launches_a_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"run.json"
            output.with_suffix(".json.samples.jsonl").write_text("partial evidence\n")
            with patch("sys.argv", ["cpu.py", "--workload", "short", "--output", str(output)]), patch.object(cpu.subprocess, "Popen") as launch:
                with self.assertRaisesRegex(ValueError, "overwrite"):
                    cpu.main()
                launch.assert_not_called()


class CandidateIdentity(unittest.TestCase):
    def test_ready_rejects_silent_default_and_shape_drift(self):
        ready = {"chunk_size":128, "context_capacity":2048}
        cpu.check_cpu_ready(ready, "scalar", False)  # frozen scalar worker
        for projection, qualified in (("tiled", True), ("scalar", True)):
            with self.assertRaisesRegex(ValueError, "differs"):
                cpu.check_cpu_ready(ready, projection, qualified)
        ready["cpu_projection"] = "tiled"
        cpu.check_cpu_ready(ready, "tiled", True)
        for key, wrong in (("chunk_size", 64), ("context_capacity", 4096)):
            with self.assertRaisesRegex(ValueError, "differs"):
                cpu.check_cpu_ready(dict(ready, **{key:wrong}), "tiled", True)

    def test_unqualified_candidate_never_launches(self):
        for options in (["--cpu-projection", "tiled"],
                        ["--backend", "llama", "--cpu-projection", "tiled"],
                        ["--backend", "llama", "--cpu-qualification", "unused.json"]):
            with patch("sys.argv", ["cpu.py", "--workload", "short", "--output", "unused.json"]+options), patch.object(cpu.subprocess, "Popen") as launch:
                with self.assertRaises(ValueError):
                    cpu.main()
                launch.assert_not_called()

    def test_qualification_binds_sources_worker_and_passed_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def write(name, content):
                path = root/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
                return hashlib.sha256(content).hexdigest()
            source_sha = write("src/model.rs", b"qualified source")
            binary = root/"worker"
            binary_sha = write("worker", b"qualified binary")
            q = {"format":"toklet-cpu-tiled-engine-functional-v1", "completed":True,
                 "source_sha256":{"src/model.rs":source_sha},
                 "worker_smoke":{"tiled":{"command":[str(binary)]}},
                 "binary_sha256":{"worker":binary_sha},
                 "rustc":{"exit_code":0,"stdout":"compiler version\n"}}
            for section in ("forward", "sessions"):
                payload = json.dumps({"cpu_projection":"tiled", "passed":True,
                                      "results":{"case":{}}}).encode()
                sha = write(f"benchmarks/results/{section}.json", payload)
                q[section] = {"report":f"benchmarks/raw/{section}.json", "report_sha256":sha}
            path = root/"qualification.json"
            def save(): path.write_text(json.dumps(q))
            save()
            with patch.object(cpu.prepare, "ROOT", root):
                actual, compiler, _ = cpu.qualified_cpu_worker(path)
                self.assertEqual(actual, binary)
                self.assertEqual(compiler, "compiler version")
                for changed in (root/"src/model.rs", binary, root/"benchmarks/results/forward.json"):
                    original = changed.read_bytes()
                    changed.write_bytes(original+b"changed")
                    with self.assertRaisesRegex(ValueError, "identity differs"):
                        cpu.qualified_cpu_worker(path)
                    changed.write_bytes(original)
                q["completed"] = False
                save()
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    cpu.qualified_cpu_worker(path)
                q["completed"] = True
                failed = json.dumps({"cpu_projection":"tiled", "passed":False, "results":{"case":{}}}).encode()
                q["forward"]["report_sha256"] = write("benchmarks/results/forward.json", failed)
                save()
                with self.assertRaisesRegex(ValueError, "failed"):
                    cpu.qualified_cpu_worker(path)

    def test_full_driver_preserves_contract_and_records_isolated_selection(self):
        # Exercise thirteen protocol responses, checkpoint rows and the final
        # schema without model downloads, device tools or numerical dispatch.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = root/"qualification.json"
            receipt.write_text("qualification receipt")
            output = root/"run.json"
            ready = {"event":"ready", "cpu_projection":"tiled", "chunk_size":128,
                     "context_capacity":2048, "memory_admitted_bytes":100,
                     "load_seconds":1, "conversion_seconds":0.5,
                     "logical_weights_bytes":20, "allocated_kv_bytes":30, "scratch_bytes":10}
            sample = {"prefill_seconds":0.25, "ttft_seconds":0.25,
                      "decode_seconds":0.75, "total_seconds":1,
                      "token_event_seconds":[0.25+i*0.75/127 for i in range(128)],
                      "emitted_tokens":128, "reused_tokens":0,
                      "prefill_tokens":128, "evaluated_tokens":255, "pending_tokens":1}
            protocol = [ready, {"event":"error", "message":"capacity"}]
            protocol += [{"event":"sample", "sample":sample}]*13
            stdin = io.StringIO()
            worker = Mock()
            worker.pid = 42
            worker.stdin = stdin
            worker.stdout = io.StringIO("\n".join(json.dumps(row) for row in protocol)+"\n")
            worker.wait.return_value = 0
            worker.poll.return_value = 0
            qualification = {"build_commands":["isolated release command"],
                             "engine_revision":{"stdout":"a"*40}, "rustflags":None,
                             "forward":{"report_sha256":"b"*64},
                             "sessions":{"report_sha256":"c"*64}}
            # Preserve input after main closes the pipe to inspect the actual
            # capacity rejection and every unchanged frozen request.
            stdin.close = lambda: None
            def command(args):
                if args[-1] == "hw.memsize": return "17179869184"
                if args[:2] == ["git", "rev-parse"]: return "d"*40
                if args[:2] == ["git", "status"]: return ""
                return "test metadata"
            argv = ["cpu.py", "--workload", "short", "--output", str(output),
                    "--cpu-projection", "tiled", "--cpu-qualification", str(receipt)]
            with ExitStack() as stack:
                for context in (
                    patch("sys.argv", argv), patch.object(cpu.prepare, "verify"),
                    patch.object(cpu.prepare, "validate_source"),
                    patch.object(cpu.prepare, "digest", return_value="e"*64),
                    patch.object(cpu.common, "command", side_effect=command),
                    patch.object(cpu.common, "footprint_peak", return_value=100),
                    patch("builtins.print"),
                ):
                    stack.enter_context(context)
                identity = stack.enter_context(patch.object(
                    cpu, "qualified_cpu_worker",
                    return_value=(root/"worker", "test compiler", qualification)))
                launch = stack.enter_context(patch.object(cpu.subprocess, "Popen", return_value=worker))
                cpu.main()
            self.assertEqual(identity.call_count, 2)
            self.assertEqual(launch.call_args.args[0][-2:], ["128", "tiled"])
            sent = [json.loads(line) for line in stdin.getvalue().splitlines()]
            self.assertEqual(len(sent), 14)
            self.assertEqual(len(sent[0]["tokens"]), 1921)
            self.assertTrue(all(len(row["tokens"])==128 and len(row["continuation"])==128 for row in sent[1:]))
            record = json.loads(output.read_text())
            self.assertEqual(record["runtime"], "Toklet tiled CPU")
            self.assertEqual(record["setup"]["warmup_repetitions"], 3)
            self.assertEqual(record["identity"]["build_flags"], ["isolated release command"])
            self.assertEqual(len(record["samples"]), 10)
            raw = output.with_suffix(".json.samples.jsonl")
            self.assertEqual(len(raw.read_text().splitlines()), 10)
            self.assertTrue(any("qualification SHA256" in note for note in record["notes"]))
            worker.kill.assert_not_called()


if __name__ == "__main__":
    unittest.main()
