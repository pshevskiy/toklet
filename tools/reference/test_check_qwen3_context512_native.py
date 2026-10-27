"""Miniature coherent journal/raw/oracle controls; no model or GPU required."""
from contextlib import ExitStack
import copy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

import check_qwen3_context512_native as checker


def miniature(root):
    names = ["embedding", "layers.0.output", "logits"]
    manifest = dict(plan=dict(gates={name: dict(atol=0.01, rtol=0.001) for name in names}), records={})
    for contract in ("fp32", "fp64"):
        body, arrays = b"", {}
        for name in names:
            dtype = "<f4" if name == "logits" else "<f2"
            data = b"".join(struct.pack("<f" if dtype == "<f4" else "<e", value) for value in [0, 1] * 4)
            arrays[name] = dict(offset=len(body), bytes=len(data), shape=[4, 2], dtype=dtype, sha256=hashlib.sha256(data).hexdigest())
            body += data
        filename = "original." + contract + ".bin"; (root / filename).write_bytes(body)
        manifest["records"]["original." + contract] = dict(file=filename, bytes=len(body), sha256=hashlib.sha256(body).hexdigest(), arrays=arrays)
    raw, frames = b"", []
    for start, rows in [(0, 3), (3, 1)]:
        for name in reversed(names):  # Legal within-chunk permutation.
            body = b"".join(struct.pack("<f", value) for value in [0, 1] * rows)
            frames.append(dict(name=name, start=start, rows=rows, columns=2, dtype="<f4", offset=len(raw), bytes=len(body), sha256=hashlib.sha256(body).hexdigest()))
            raw += body
    (root / "small-original.bin").write_bytes(raw)
    journal = root / "small-original.jsonl"; journal.write_text("".join(json.dumps(v) + "\n" for v in frames))
    stats = {name: {**checker.statistics(), "values": 8} for name in names}
    native = dict(canonical=stats, fp64_diagnostic=copy.deepcopy(stats), argmax_ids=[1] * 4,
        canonical_argmax_mismatches=[], canonical_passed=True, fp64_diagnostic_passed=True,
        native_bytes=96, native_sha256=hashlib.sha256(raw).hexdigest(), snapshots=6,
        case="original", recipe="small", chunk=3, segments=[3, 1], rows_per_state=4,
        admitted_bytes=80, state_checks_passed=True)
    (root / "small-original.json").write_text(json.dumps(native))
    frozen = dict(selections={"original.fp32": dict(ids=[1] * 4)})
    return manifest, native, frozen, frames


class NativeReplay(unittest.TestCase):
    def patch_geometry(self):
        stack = ExitStack()
        for key, value in [("POSITIONS", 4), ("STATES", 3), ("RAW_BYTES", 96)]:
            stack.enter_context(mock.patch.object(checker, key, value))
        return stack

    def test_complete_frames_and_nonuniform_chunk_boundaries(self):
        with tempfile.TemporaryDirectory() as temporary, self.patch_geometry():
            root = Path(temporary); manifest, _, _, frames = miniature(root)
            recipe = dict(chunk=3, segments=[3, 1])
            self.assertEqual(list(checker.chunks(recipe)), [(0, 3), (3, 1)])
            self.assertEqual(list(checker.frames(root / "small-original.jsonl", recipe, manifest)), frames)

    def test_journal_keyset_duplicate_missing_extent_and_boolean_controls(self):
        with tempfile.TemporaryDirectory() as temporary, self.patch_geometry():
            root = Path(temporary); manifest, _, _, original = miniature(root)
            recipe = dict(chunk=3, segments=[3, 1]); path = root / "small-original.jsonl"
            mutations = [lambda rows: rows.pop(), lambda rows: rows.append(copy.deepcopy(rows[-1])),
                lambda rows: rows[0].update(extra=1), lambda rows: rows[0].update(start=1),
                lambda rows: rows[0].update(offset=4), lambda rows: rows[0].update(columns=3),
                lambda rows: rows[0].update(dtype="<f2"), lambda rows: rows[0].update(bytes=1),
                lambda rows: rows[0].update(start=False), lambda rows: rows[3].update(rows=True),
                lambda rows: rows[1].update(name=rows[0]["name"])]
            for mutation in mutations:
                rows = copy.deepcopy(original); mutation(rows)
                path.write_text("".join(json.dumps(v) + "\n" for v in rows))
                with self.assertRaises(ValueError):
                    list(checker.frames(path, recipe, manifest))

    def test_raw_replay_authenticates_every_row_and_both_oracles(self):
        with tempfile.TemporaryDirectory() as temporary, self.patch_geometry(), ExitStack() as owners:
            root = Path(temporary); manifest, native, frozen, _ = miniature(root)
            source = {label: owners.enter_context(checker.precision.Owner(root / record["file"], record)) for label, record in manifest["records"].items()}
            owner = owners.enter_context(checker.precision.Owner(root / "small-original.bin", dict(bytes=96, sha256=native["native_sha256"])))
            report = []
            checker.replay_case(root, "small-original", dict(chunk=3, segments=[3, 1]), "original", manifest, frozen, source, owner, report)
            self.assertTrue(report[0]["complete"] and report[0]["canonical_passed"])
            self.assertEqual(report[0]["bytes"], 96)
            self.assertEqual(report[0]["snapshots"], 6)
            self.assertEqual(report[0]["argmax_mismatches"], [])

    def test_changed_frame_hash_retains_partial_comparison(self):
        with tempfile.TemporaryDirectory() as temporary, self.patch_geometry(), ExitStack() as owners:
            root = Path(temporary); manifest, native, frozen, frames = miniature(root)
            frames[0]["sha256"] = "0" * 64
            (root / "small-original.jsonl").write_text("".join(json.dumps(v) + "\n" for v in frames))
            source = {label: owners.enter_context(checker.precision.Owner(root / record["file"], record)) for label, record in manifest["records"].items()}
            owner = owners.enter_context(checker.precision.Owner(root / "small-original.bin", dict(bytes=96, sha256=native["native_sha256"])))
            report = []
            with self.assertRaisesRegex(ValueError, "Consumed native frame bytes differ"):
                checker.replay_case(root, "small-original", dict(chunk=3, segments=[3, 1]), "original", manifest, frozen, source, owner, report)
            self.assertFalse(report[0]["complete"])
            self.assertEqual(report[0]["canonical"]["logits"]["values"], 6)

    def test_metrics_keep_nonfinite_and_canonical_rejections_explicit(self):
        import numpy as np
        result = checker.statistics()
        checker.observe(result, np.array([1, np.nan, 2]), np.array([1, 0, 1]), 7, dict(atol=0.1, rtol=0.01))
        self.assertEqual((result["values"], result["failed_values"], result["nonfinite_values"]), (3, 1, 1))
        self.assertEqual(result["first_failures"][0]["position"], 7)
        self.assertEqual(result["first_failures"][0]["feature"], 2)
        self.assertFalse(checker.passed(dict(logits=result)))

    def test_document_duplicate_nonfinite_and_size_controls(self):
        for body in ['{"a":1,"a":2}', '{"a":NaN}']:
            with self.assertRaises(ValueError):
                checker.decode(body)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "document"; path.write_bytes(b" " * 17)
            with self.assertRaisesRegex(ValueError, "exceeds bound"):
                checker.document(path, 16)

    def test_coherently_relabeled_case_and_typed_metadata_reject(self):
        with tempfile.TemporaryDirectory() as temporary, self.patch_geometry():
            root = Path(temporary); _, native, _, _ = miniature(root)
            recipe = dict(name="small", chunk=3, segments=[3, 1])
            checker.case_metadata(native, recipe, "original", 100)
            mutations = [lambda v: v.update(case="future_tail"), lambda v: v.update(recipe="other"),
                lambda v: v.update(chunk=4), lambda v: v.update(segments=[1, 3]),
                lambda v: v.update(segments=[3, True]), lambda v: v.update(rows_per_state=3),
                lambda v: v.update(rows_per_state=True), lambda v: v.update(snapshots=7),
                lambda v: v.update(admitted_bytes=101), lambda v: v.update(admitted_bytes=True),
                lambda v: v.update(state_checks_passed=1), lambda v: v.update(canonical_passed=1),
                lambda v: v["canonical"]["logits"].update(failed_values=False)]
            for mutation in mutations:
                value = copy.deepcopy(native); mutation(value)
                aggregate = dict(recipes=[copy.deepcopy(value)])  # Coherent forgery remains a forgery.
                self.assertEqual(value, aggregate["recipes"][0])
                with self.assertRaises(ValueError):
                    checker.case_metadata(value, recipe, "original", 100)

    def test_aggregate_reference_and_generation_counts_bind_to_plan(self):
        plan = dict(context=512, total_stage_snapshots=5016, recipes=[1] * 4, cases=["original", "future_tail"],
            pending_generation=dict(prompt_tokens=510, requested_output=2, cancel_after_emitted=1,
                                    first_sample=1879, continue_one_sample=0))
        summary = dict(capacity=512, stage_snapshots=5016, manifest_sha256=checker.precision.MANIFEST_SHA,
            comparison_sha256=checker.precision.COMPARISON_SHA, recipes=[{}] * 8,
            canonical_fp32_passed=True, fp64_is_diagnostic_only=True, future_prefix_exact=True,
            source_quality_qualified=False, held_out_quality_qualified=False, full_model_window_qualified=False,
            phone_qualified=False, precision_or_capacity_changed=False)
        checker.aggregate_metadata(summary, plan)
        for field, value in [("capacity", 511), ("capacity", True), ("stage_snapshots", 1),
                             ("manifest_sha256", "0" * 64), ("comparison_sha256", "0" * 64),
                             ("phone_qualified", 0), ("source_quality_qualified", True)]:
            changed = copy.deepcopy(summary); changed[field] = value
            with self.assertRaises(ValueError):
                checker.aggregate_metadata(changed, plan)
        value = dict(passed=True, sequence="original", capacity=512, callback_cancel_prompt=510,
            requested_output=2, emitted_before_cancel=1, resumed_ids=[1879, 0], pending_then_flushed=True,
            rejected_513_state_unchanged=True, output_failure_retry=True, sampled_history_fresh_replay=True,
            partial_discard_reappend=True, mid_gpu_cancellation_qualified=False, canonical_fp32_passed=True,
            canonical=checker.statistics(), fp64_diagnostic=checker.statistics())
        checker.contract_metadata(value, plan)
        for field, replacement in [("sequence", "future_tail"), ("capacity", 511),
                ("callback_cancel_prompt", 509), ("requested_output", 1), ("emitted_before_cancel", True),
                ("resumed_ids", [1879, True]), ("mid_gpu_cancellation_qualified", True)]:
            changed = copy.deepcopy(value); changed[field] = replacement
            with self.assertRaises(ValueError):
                checker.contract_metadata(changed, plan)

    def test_shortest_f32_examples_require_exact_typed_bits(self):
        original = checker.statistics()
        original["values"] = original["failed_values"] = 1
        original["first_failures"] = [dict(position=25, feature=625, actual=3.744140625,
            expected=3.6875, delta=0.056640625, bound=0.05446875)]
        encoded = copy.deepcopy(original)
        encoded["first_failures"][0]["actual"] = 3.7441406
        self.assertTrue(checker.same_statistic(original, encoded))
        # Both a changed f32 bit pattern and a changed f64 metric must reject.
        encoded["first_failures"][0]["actual"] = 3.7441409
        self.assertFalse(checker.same_statistic(original, encoded))
        encoded = copy.deepcopy(original); encoded["first_failures"][0]["delta"] += 1e-12
        self.assertFalse(checker.same_statistic(original, encoded))
        encoded = copy.deepcopy(original); encoded["first_failures"][0]["actual"] = True
        self.assertFalse(checker.same_statistic(original, encoded))
        zero = copy.deepcopy(original); zero["first_failures"][0]["actual"] = 0.0
        negative = copy.deepcopy(zero); negative["first_failures"][0]["actual"] = -0.0
        self.assertFalse(checker.same_statistic(zero, negative))


if __name__ == "__main__":
    unittest.main()
