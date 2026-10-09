"""Offline checks for malformed projection evidence and interrupted retries."""
import copy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import projection


def complete_row():
    return {
        "tokens": 1, "in_features": 896, "out_features": 896,
        "warmup_pairs": 3, "measured_pairs": 10,
        "samples": [{"candidate_first": i % 2 == 0,
                     "scalar_seconds": 0.002, "tiled_seconds": 0.001}
                    for i in range(10)],
        "bitwise_equal_every_pair": True,
        "input_bytes": 896 * 4, "weights_bytes": 896 * 896 * 4,
        "output_buffers_bytes": 896 * 8, "candidate_accumulator_stack_bytes": 128,
        "input_sha256": "a" * 64, "weights_sha256": "b" * 64,
        "output_sha256": "c" * 64,
    }


class Evidence(unittest.TestCase):
    def test_complete_pairs_and_corrupted_evidence(self):
        row = complete_row()
        projection.validate(row, projection.SHAPES[0])
        self.assertEqual(row["summary"]["median_pair_speedup"], 2.0)
        corruptions = [
            ("measured_pairs", 9), ("bitwise_equal_every_pair", False),
            ("weights_bytes", 0), ("output_sha256", "g" * 64),
            ("tokens", 128),
        ]
        for key, value in corruptions:
            bad = copy.deepcopy(row)
            bad[key] = value
            with self.assertRaises(ValueError):
                projection.validate(bad, projection.SHAPES[0])
        for value in [0, -1, float("nan"), float("inf"), True]:
            bad = copy.deepcopy(row)
            bad["samples"][0]["tiled_seconds"] = value
            with self.assertRaises(ValueError):
                projection.validate(bad, projection.SHAPES[0])
        bad = copy.deepcopy(row)
        bad["samples"][0]["candidate_first"] = False
        with self.assertRaises(ValueError):
            projection.validate(bad, projection.SHAPES[0])

    def test_partial_evidence_blocks_retry_before_worker_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "pairs.json"
            output.with_suffix(".json.rows.jsonl").write_text("partial evidence\n")
            with patch("sys.argv", ["projection.py", "--output", str(output)]), \
                    patch.object(projection.subprocess, "Popen") as launch:
                with self.assertRaisesRegex(ValueError, "overwrite"):
                    projection.main()
                launch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
