"""Portable controls independent of downloaded inputs, model execution and MLX."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import qwen3_context8192_profile as profile
import qwen3_context8192_source as subject


def manifest(raw):
    # Synthetic receipt hash is only a corruption-test fixture, never publication.
    return dict(profile.REQUIRED, complete=True,
        files={name: dict(bytes=len(raw), sha256=profile.sha(raw))
               for name in ('A.u32le', 'B.u32le', 'Q.u32le')},
        a4096_prefix_sha256=profile.F2_A, admission_sha256='a' * 64,
        parent_receipts=dict(f2_result=dict(bytes=1, sha256=profile.F2_RESULT),
                             retained_parallel_result=dict(bytes=1, sha256='b' * 64)))


class Context8192Source(unittest.TestCase):
    def test_old_helpers_are_directly_reused_and_loader_keyword_is_supported(self):
        for name in ('adapted_forward', 'original_model_classes', 'stock_capacity_after_update',
                     'release_pass', 'retain_output', 'cache_shapes', 'scored'):
            self.assertIs(getattr(subject, name), getattr(subject.original, name))
        model, arguments = object(), object()
        with patch.dict(sys.modules, {'mlx_lm.models.qwen3': SimpleNamespace(Model=model, ModelArgs=arguments)}):
            self.assertEqual(subject.original_model_classes(config={}), (model, arguments))
        self.assertEqual(subject.original.CAPACITY, 4096)
        self.assertEqual(subject.original.QUALITY_RANGES, ((3071, 4095),))

    def test_exact_ledger_and_fixed_schedule_masks(self):
        memory = subject.memory_plan()
        self.assertEqual(sum(memory['load'].values()), 5927671248)
        self.assertEqual(sum(memory['inference'].values()), 6239337192)
        self.assertEqual(memory['inference']['cache_and_update_overlap'], 1879048192)
        self.assertEqual(memory['inference']['graph_workspace'], 2147483648)
        self.assertEqual(memory['limit'] - memory['admitted'], 203113752)
        eq = list(subject.schedule('equivalence'))
        self.assertEqual([i for start, end in eq for i in range(start, end)], list(range(1025)))
        selected = [i for i in range(1025) if subject.scored(i, subject.PROBE_RANGES)]
        self.assertEqual(selected, [253, 254, 255, 256, 257, 1023])
        self.assertIn(0, [sum(start <= i < end for i in selected) for start, end in eq])
        self.assertTrue(any(0 < sum(start <= i < end for i in selected) < end - start for start, end in eq))
        quality = list(subject.schedule('quality'))
        self.assertEqual(len(quality), 1024)
        projected = [(start, end) for start, end in quality
                     if any(subject.scored(i, subject.QUALITY_RANGES) for i in range(start, end))]
        self.assertEqual((projected[0], projected[-1]), ((7160, 7168), (8184, 8192)))
        self.assertEqual([i + 1 for i in range(8192) if subject.scored(i, subject.QUALITY_RANGES)],
                         list(range(7168, 8192)))
        with self.assertRaises(ValueError):
            list(subject.schedule('loss'))

    def test_stock_recurrence_and_exact8192_cache_metadata_before_rejection(self):
        capacity, growth = 0, []
        for start, end in subject.schedule('equivalence'):
            new = subject.stock_capacity_after_update(capacity, start, end - start)
            if new != capacity:
                growth.append((start, end, new))
            capacity = new
        self.assertEqual(growth, [(0, 8, 256), (256, 257, 512), (505, 513, 761),
                                  (761, 769, 1017), (1017, 1024, 1273)])
        value = SimpleNamespace(shape=(1, 8, 8192, 128), dtype='bf16')
        cache = [SimpleNamespace(offset=1025, step=8192, keys=value, values=value) for _ in range(28)]
        subject.validate_cache(cache, 1025, True, 'bf16', 8192)
        events = []
        cache[13].step = 4096
        with self.assertRaises(ValueError):
            subject.validate_cache(cache, 1025, True, 'bf16', 8192, before=['before'], emit=events.append)
        self.assertEqual(events[0]['before'], ['before'])
        self.assertEqual(events[0]['after'][13]['step'], 4096)
        self.assertEqual(events[0]['expected_capacity'], 8192)

    def test_adapter_step_instance_only_and_evaluates_before_fresh_view(self):
        calls = []
        value = SimpleNamespace(shape=(1, 8, 8, 128), dtype='bf16')
        class Stock:
            step = 256
            def __init__(self):
                self.offset = 0
            def update_and_fetch(self, keys, values):
                calls.append('update')
                self.offset += keys.shape[2]
                self.keys, self.values = keys, values
            def keys_and_values(self):
                calls.append('fresh_view')
                return self.keys, self.values
        mx = SimpleNamespace(bfloat16='bf16', eval=lambda *args: calls.append('eval'),
                             synchronize=lambda: calls.append('sync'))
        adapted = subject.adapted_cache_class(mx, Stock)()
        self.assertEqual((adapted.step, Stock.step), (8192, 256))
        self.assertEqual(adapted.update_and_fetch(value, value), (value, value))
        self.assertEqual(calls, ['update', 'eval', 'sync', 'fresh_view'])
        adapted.offset = 8185
        with self.assertRaises(ValueError):
            adapted.update_and_fetch(value, value)
        self.assertEqual(calls, ['update', 'eval', 'sync', 'fresh_view'])

    def test_target_rows_and_exact_widening_failure_retained(self):
        import numpy as np
        bits = np.zeros((8, 151936), dtype='<u2')
        wide = np.zeros((8, 151936), dtype='<f4')
        bits[7, 13], wide[7, 13] = 0x3f80, 1
        events, raw = [], []
        rows, losses = subject.consume_rows(bits, wide, 7160, 7168, 'quality', 'quality', True,
            [13] * 8192, None, events.append, lambda *args: raw.append(args))
        self.assertEqual((list(rows), len(losses)), ([7167], 1))
        self.assertEqual((events[0]['target_bf16_bits'], events[0]['target_logit_f32_bits']), (0x3f80, 0x3f800000))
        rows, losses = subject.consume_rows(bits, wide, 8184, 8192, 'quality', 'quality', True,
            [13] * 8192, None, events.append, lambda *args: raw.append(args))
        self.assertEqual((list(rows), len(losses)), (list(range(8184, 8191)), 7))
        wide[7, 13] = 0
        with self.assertRaises(ValueError):
            subject.consume_rows(bits, wide, 7160, 7168, 'quality', 'quality', True,
                [13] * 8192, None, events.append, lambda *args: raw.append(args))
        self.assertEqual(raw[-1][1:], (bits.tobytes(), wide.tobytes()))
        self.assertEqual(events[-1]['event'], 'invalid_output')

    def test_partial_failure_primary_survives_cleanup(self):
        errors, primary = [], ValueError('primary numerical failure')
        def fail():
            raise RuntimeError('cleanup failure')
        mx = SimpleNamespace(synchronize=fail, clear_cache=fail)
        with patch.object(subject.original.gc, 'collect', side_effect=fail):
            try:
                raise primary
            except ValueError as active:
                subject.release_pass(mx, lambda _: fail(), 'probe', errors, active)
                self.assertIs(active, primary)
        self.assertEqual(len(errors), 4)
        with patch.object(subject.original.gc, 'collect', return_value=0), self.assertRaisesRegex(ValueError, 'cleanup'):
            subject.release_pass(mx, lambda _: None, 'probe', [], None)

    def test_input_policy_and_old_receipt_cannot_cross_profiles(self):
        raw = struct.pack('<8192I', *([9707, 11, 1879, 0] * 2048))
        good = manifest(raw)
        profile.validate(good)
        for key, value in [('capacity', 4096), ('score_rows', [7168, 8192]),
                           ('quality_global_token_start', 10001), ('targets', 1023),
                           ('complete', 1), ('quality_mean_nll_increase_limit', 0.051),
                           ('preserved_quality_spans', [[False, 10001], [10001, 14097]])]:
            bad = copy.deepcopy(good)
            bad[key] = value
            with self.assertRaises(ValueError):
                profile.validate(bad)
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / 'A.u32le').write_bytes(raw)
            (folder / 'Q.u32le').write_bytes(raw)
            body = json.dumps(good).encode()
            (folder / 'manifest.json').write_bytes(body)
            self.assertEqual(len(subject.load_inputs(folder, 'equivalence', profile.sha(body))[1]), 1025)
            self.assertEqual(len(subject.load_inputs(folder, 'quality', profile.sha(body))[1]), 8192)
            bad = bytearray(raw)
            bad[-1] ^= 1
            (folder / 'Q.u32le').write_bytes(bad)
            with self.assertRaises(ValueError):
                subject.load_inputs(folder, 'quality', profile.sha(body))
        receipt = dict(complete=True, mode='equivalence', profile=subject.PROFILE,
            model_pins=subject.baseline.PINS, input_manifest_sha256='a' * 64,
            producer_sha256='b' * 64, equivalence_passed=True, cleanup_errors=[], dependencies={'test': 'pin'})
        with patch.object(subject.source.calibration.reference, 'load', return_value=receipt), \
                patch.object(subject, 'dependencies', return_value={'test': 'pin'}):
            subject.require_equivalence('unused', 'c' * 64, 'a' * 64, 'b' * 64)
            for field, value in [('profile', subject.original.PROFILE), ('dependencies', {}),
                                 ('input_manifest_sha256', 'd' * 64), ('cleanup_errors', ['failed']),
                                 ('complete', False)]:
                bad = dict(receipt, **{field: value})
                with patch.object(subject.source.calibration.reference, 'load', return_value=bad), self.assertRaises(ValueError):
                    subject.require_equivalence('unused', 'c' * 64, 'a' * 64, 'b' * 64)


if __name__ == '__main__':
    unittest.main()
