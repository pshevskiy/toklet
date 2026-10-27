"""Exact interval and coherent streaming controls, independent of checkpoints."""
from contextlib import ExitStack
from fractions import Fraction
import hashlib
import math
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

import qwen3_context_precision as reader


def records(values, dtype="<f4", shape=None):
    body = b"".join(struct.pack("<f" if dtype == "<f4" else "<e", value) for value in values)
    record = dict(bytes=len(body), sha256=hashlib.sha256(body).hexdigest())
    item = dict(bytes=len(body), sha256=record["sha256"], offset=0,
                dtype=dtype, shape=shape or [len(values)])
    return body, record, item


class PrecisionReader(unittest.TestCase):
    def test_exact_inclusive_touch_and_clear_gap(self):
        gate = dict(atol=1.0, rtol=0.0)
        self.assertFalse(reader.intervals(0, 2, gate)["disjoint"])
        self.assertTrue(reader.intervals(0, 3, gate)["disjoint"])
        self.assertGreater(Fraction(reader.intervals(0, 3, gate)["exact_guarded_gap"]), 0)
        # One evaluated endpoint ULP of separation is conservatively touching.
        self.assertFalse(reader.intervals(0, math.nextafter(2, math.inf), gate)["disjoint"])

    def test_relative_radii_are_oracle_specific_and_sign_symmetric(self):
        gate = dict(atol=0.125, rtol=0.01)
        value = reader.intervals(-2.0, -4.0, gate)
        self.assertEqual(value["fp32_evaluated_tolerance"], 0.145)
        self.assertEqual(value["fp64_evaluated_tolerance"], 0.165)
        self.assertNotEqual(value["fp32_radius"], value["fp64_radius"])
        self.assertEqual(value["exact_guarded_gap"], reader.intervals(2.0, 4.0, gate)["exact_guarded_gap"])
        zero = reader.intervals(0.0, -0.0, gate)
        self.assertFalse(zero["disjoint"])

    def test_invalid_interval_and_radius_overflow_reject(self):
        for a, b, gate in [(math.nan, 1, dict(atol=1, rtol=0)),
                           (1, math.inf, dict(atol=1, rtol=0)),
                           (True, 1, dict(atol=1, rtol=0)),
                           (1, 1, dict(atol=0, rtol=1)),
                           (1, 1, dict(atol=1, rtol=-1)),
                           (1e308, 1, dict(atol=1, rtol=1e308))]:
            with self.assertRaises(ValueError):
                reader.intervals(a, b, gate)

    def test_layout_coordinates_and_semantic_axes(self):
        self.assertEqual(reader.coordinates(23, [2, 3, 4]), [1, 2, 3])
        self.assertEqual(reader.semantic("layers.27.attention_scores", [3, 19, 7]),
                         dict(query_head=3, query_position=19, key_position=7))
        self.assertEqual(reader.semantic("logits", [19, 1234]), dict(token_position=19, vocabulary_id=1234))
        self.assertEqual(reader.semantic("layers.27.k_rotary", [19, 2, 5]), dict(token_position=19, kv_head=2, head_component=5))
        for index, shape in [(-1, [2]), (2, [2]), (0, []), (0, [True]), (0, [-1])]:
            with self.assertRaises(ValueError):
                reader.coordinates(index, shape)

    def test_authenticated_streamed_pair_replays_asymmetric_failure(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as contexts:
            owners, items = [], []
            for label, values in [("a", [1, -2, 0, 4]), ("b", [1, -2, 0, 4.5])]:
                body, record, item = records(values, shape=[2, 2])
                path = Path(temporary) / label; path.write_bytes(body)
                owners.append(contexts.enter_context(reader.Owner(path, record))); items.append(item)
            selection = {name: dict(ids=[1, 0], margins=[0.25, 0.5]) for name in ("fp32", "fp64")}
            report = []
            state = reader.inspect_pair(*owners, "logits", items, dict(atol=0.1, rtol=0.01),
                dict(failed_values=1, maximum_absolute=0.5, maximum_gate_fraction=0.5 / (0.1 + 0.01 * 4.5)), selection, report)
            self.assertTrue(state["authenticated"])
            self.assertEqual(state["affected_positions"], [1])
            self.assertEqual(state["disjoint_values"], 1)
            self.assertEqual(state["findings"][0]["semantic"], dict(token_position=1, vocabulary_id=1))
            self.assertEqual(state["findings"][0]["raw_greedy_at_position"]["fp64"], dict(id=0, margin=0.5))

    def test_changed_consumed_array_hash_keeps_partial_findings(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as contexts:
            owners, items = [], []
            for label, values in [("a", [0, 0]), ("b", [1, 0])]:
                body, record, item = records(values, shape=[1, 2])
                path = Path(temporary) / label; path.write_bytes(body)
                owners.append(contexts.enter_context(reader.Owner(path, record))); items.append(item)
            items[0]["sha256"] = "0" * 64
            report = []
            selection = {name: dict(ids=[0], margins=[0.5]) for name in ("fp32", "fp64")}
            with self.assertRaisesRegex(ValueError, "Consumed array bytes differ"):
                reader.inspect_pair(*owners, "logits", items, dict(atol=0.1, rtol=0),
                    dict(failed_values=1, maximum_absolute=1, maximum_gate_fraction=10), selection, report)
            self.assertEqual(len(report[0]["findings"]), 1)
            self.assertFalse(report[0]["authenticated"])

    def test_nonfinite_pair_preserves_primary_error(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as contexts:
            owners, items = [], []
            for label, values in [("a", [0]), ("b", [math.nan])]:
                body, record, item = records(values, shape=[1, 1]); path = Path(temporary) / label; path.write_bytes(body)
                owners.append(contexts.enter_context(reader.Owner(path, record))); items.append(item)
            report = []
            with self.assertRaisesRegex(ValueError, "Nonfinite captured pair"):
                reader.inspect_pair(*owners, "logits", items, dict(atol=0.1, rtol=0),
                    dict(failed_values=1), {}, report)
            self.assertFalse(report[0]["authenticated"])

    def test_invalid_extents_and_symlinks_reject(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); body, record, item = records([1, 2]); path = root / "data"; path.write_bytes(body)
            link = root / "link"; link.symlink_to(path)
            with self.assertRaises(OSError):
                with reader.Owner(link, record):
                    pass
            with reader.Owner(path, record) as owner:
                for change in [dict(offset=-1), dict(bytes=1), dict(shape=[-2]), dict(dtype="<f8")]:
                    with self.assertRaises(ValueError):
                        list(owner.blocks({**item, **change}))
            self.assertTrue(owner.audit["endpoint_checked"])

    def test_truncated_stream_and_endpoint_error_do_not_replace_primary(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "data"; body, record, item = records([1, 2]); path.write_bytes(body)
            with self.assertRaisesRegex(ValueError, "Captured array is truncated"):
                with reader.Owner(path, record) as owner, mock.patch.object(reader, "ELEMENTS", 1):
                    blocks = owner.blocks(item)
                    next(blocks)
                    path.write_bytes(body[:4])
                    list(blocks)
            self.assertFalse(owner.audit["endpoint_checked"])
            self.assertIsNotNone(owner.audit["endpoint_error"])

    def test_growing_file_is_bounded_and_preserves_primary_error(self):
        for primary in [False, True]:
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "data"; body, record, _ = records([1, 2]); path.write_bytes(body)
                expected = RuntimeError if primary else ValueError
                message = "original failure" if primary else "grew beyond admitted extent"
                with self.assertRaisesRegex(expected, message):
                    with reader.Owner(path, record) as owner:
                        with path.open("ab") as file:
                            file.write(b"extra bytes")
                        if primary:
                            raise RuntimeError("original failure")
                self.assertFalse(owner.audit["endpoint_checked"])
                self.assertIn("grew beyond admitted extent", owner.audit["endpoint_error"])


if __name__ == "__main__":
    unittest.main()
