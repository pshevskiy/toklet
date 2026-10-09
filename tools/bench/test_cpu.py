"""Offline checks for benchmark cache preparation and fixed workload shapes."""
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
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


if __name__ == "__main__":
    unittest.main()
