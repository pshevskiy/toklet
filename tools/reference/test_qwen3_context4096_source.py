"""Portable fixed-profile controls; no MLX import, GPU, model or corpus needed."""
import copy
import hashlib
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import qwen3_context4096_source as subject


class Context4096Source(unittest.TestCase):
    def test_pass_cleanup_preserves_primary_and_fails_when_cleanup_alone_fails(self):
        calls, errors = [], []

        def fail(name):
            calls.append(name)
            raise RuntimeError(name + ' failed')

        mx = SimpleNamespace(synchronize=lambda: fail('synchronize'), clear_cache=lambda: fail('clear_cache'))
        primary = ValueError('original identity mismatch')
        with patch.object(subject.gc, 'collect', side_effect=lambda: fail('gc')):
            with self.assertRaises(ValueError) as caught:
                pass_error = None
                try:
                    raise primary
                except BaseException as error:
                    pass_error = error
                    raise
                finally:
                    subject.release_pass(mx, lambda _: fail('usage'), 'test', errors, pass_error)
        self.assertIs(caught.exception, primary)
        self.assertEqual(calls, ['gc', 'synchronize', 'clear_cache', 'usage'])
        self.assertEqual(len(errors), 4)
        self.assertTrue(all(message.startswith('test/') for message in errors))
        calls.clear()
        errors.clear()
        with patch.object(subject.gc, 'collect', return_value=0):
            with self.assertRaisesRegex(ValueError, 'Source pass cleanup failed'):
                subject.release_pass(mx, lambda _: fail('usage'), 'test', errors, None)
        self.assertEqual(calls, ['synchronize', 'clear_cache', 'usage'])
        self.assertEqual(len(errors), 3)

    def test_original_loader_hook_accepts_config_keyword_without_model_work(self):
        model, arguments = object(), object()
        module = SimpleNamespace(Model=model, ModelArgs=arguments)
        with patch.dict(sys.modules, {'mlx_lm.models.qwen3': module}):
            self.assertEqual(subject.original_model_classes(config={'model_type': 'qwen3'}), (model, arguments))

    def test_fixed_quality_target_selection_uses_full_chunk_before_mask(self):
        import numpy as np
        bits = np.zeros((8, subject.VOCAB), dtype='<u2')
        wide = np.zeros((8, subject.VOCAB), dtype='<f4')
        bits[7, 13] = 0x3f80
        wide[7, 13] = 1
        ids = [13] * 4096
        events, retained = [], []
        rows, losses = subject.consume_rows(bits, wide, 3064, 3072, 'quality', 'quality', True, ids, None,
                                             events.append, lambda *args: retained.append(args))
        self.assertEqual(list(rows), [3071])
        self.assertEqual(len(losses), 1)
        self.assertEqual((events[0]['position'], events[0]['target_token'], events[0]['target_bf16_bits'],
                          events[0]['target_logit_f32_bits']), (3071, 13, 0x3f80, 0x3f800000))
        self.assertEqual(retained, [])
        events.clear()
        rows, losses = subject.consume_rows(bits, wide, 4088, 4096, 'quality', 'quality', True, ids, None,
                                             events.append, lambda *args: retained.append(args))
        self.assertEqual(list(rows), list(range(4088, 4095)))
        self.assertEqual(len(losses), 7)
        self.assertEqual([event['target_token'] for event in events], [13] * 7)

    def test_nonfinite_wrong_widening_and_mismatch_retain_before_rejection(self):
        import numpy as np
        for kind in ('nonfinite', 'wrong_widening', 'mismatch'):
            events = []
            bits = np.zeros((1, subject.VOCAB), dtype='<u2')
            wide = np.zeros((1, subject.VOCAB), dtype='<f4')
            if kind == 'nonfinite':
                bits[0, 17] = 0x7f80
                wide[0, 17] = np.inf
            elif kind == 'wrong_widening':
                bits[0, 17] = 0x3f80
            expected = {0: 'incorrect'} if kind == 'mismatch' else None
            with self.assertRaises(ValueError):
                subject.consume_rows(bits, wide, 0, 1, 'equivalence', 'test', False, [0], expected,
                                     lambda event: events.append(('event', event)),
                                     lambda name, raw, widened: events.append(('raw', name, raw, widened)))
            raw_records = [event for event in events if event[0] == 'raw']
            self.assertEqual(len(raw_records), 1)
            self.assertEqual(raw_records[0][2:], (bits.tobytes(), wide.tobytes()))
            self.assertEqual(events[-1][1]['event'], 'row_mismatch' if kind == 'mismatch' else 'invalid_output')
        events = []
        with self.assertRaises(ValueError):
            subject.consume_rows(bits, wide, 0, 2, 'equivalence', 'test', False, [0, 0], None,
                                 events.append, lambda *args: self.fail('Malformed geometry is not a valid raw row'))
        self.assertEqual(events[0]['event'], 'invalid_readback')

    def test_partial_write_and_pre_runtime_failure_have_receipts(self):
        class Short(io.BytesIO):
            def write(self, body):
                return super().write(body[:3])

            def close(self):
                pass

        records = {}
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(Path, 'open', return_value=Short()):
                with self.assertRaises(OSError):
                    subject.retain_output(temporary, records, 'failed', b'abcdefgh', b'wide')
            record = records[str(Path(temporary) / 'failed.bf16le.bin')]
            self.assertEqual((record['bytes'], record['intended_bytes']), (3, 8))
            self.assertEqual(record['sha256'], hashlib.sha256(b'abc').hexdigest())
        report = {}
        with patch.object(subject, '_execute', side_effect=ValueError('early admission failure')):
            with self.assertRaises(ValueError):
                subject.execute(None, 'equivalence', None, report, None, '0' * 64)
        self.assertEqual(report, dict(complete=False, primary_error='ValueError: early admission failure'))

    def test_staged_schedule_and_fixed_masks(self):
        chunks = list(subject.schedule('equivalence'))
        self.assertEqual([position for start, end in chunks for position in range(start, end)], list(range(1025)))
        self.assertEqual({end - start for start, end in chunks}, {1, 7, 8})
        self.assertTrue(set(subject.ENDPOINTS) <= {end for _, end in chunks})
        selected = [i for i in range(1025) if subject.scored(i, subject.PROBE_RANGES)]
        self.assertEqual(selected, [253, 254, 255, 256, 257, 1023])
        counts = [sum(subject.scored(i, subject.PROBE_RANGES) for i in range(start, end)) for start, end in chunks]
        self.assertIn(0, counts)
        self.assertTrue(any(0 < count < end - start for (start, end), count in zip(chunks, counts)))
        quality = list(subject.schedule('quality'))
        self.assertEqual(len(quality), 512)
        projected = [(start, end) for start, end in quality
                     if any(subject.scored(i, subject.QUALITY_RANGES) for i in range(start, end))]
        self.assertEqual((projected[0], projected[-1]), ((3064, 3072), (4088, 4096)))
        self.assertEqual([i + 1 for i in range(4096) if subject.scored(i, subject.QUALITY_RANGES)], list(range(3072, 4096)))
        with self.assertRaises(ValueError):
            list(subject.schedule('loss'))

    def test_memory_charges_full_cache_overlap_and_one_chunk(self):
        value = subject.memory_plan()
        self.assertEqual(value['inference']['cache_and_update_overlap'], 939524096)
        self.assertEqual(value['inference']['chunk_output_max_fp32'], 4861952)
        self.assertEqual(sum(value['inference'].values()), 5299813096)
        self.assertEqual(value['admitted'], 5927671248)
        self.assertLess(value['admitted'], 6 * 1024**3)
        self.assertTrue(value['cache_passes_are_sequential'])

    def test_input_identity_geometry_scoring_and_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            raw = struct.pack('<4096I', *range(4096))
            manifest = dict(format='toklet-qwen3-context4096-inputs-v1', capacity=4096,
                            numerical_parent_manifest_sha256=subject.NUMERICAL_PARENT,
                            quality_parent_manifest_sha256=subject.baseline.CONTRACT_SHA,
                            quality_global_token_start=10001, score_rows=[3071, 4095],
                            score_target_indices=[3072, 4096], targets=1024,
                            quality_mean_nll_increase_limit=0.05,
                            files={name: dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
                                   for name in ('A.u32le', 'B.u32le', 'Q.u32le')})
            for name in manifest['files']:
                (folder / name).write_bytes(raw)

            def freeze(document):
                body = json.dumps(document).encode()
                (folder / 'manifest.json').write_bytes(body)
                return hashlib.sha256(body).hexdigest()

            digest = freeze(manifest)
            self.assertEqual(subject.load_inputs(folder, 'equivalence', digest)[1], list(range(1025)))
            self.assertEqual(subject.load_inputs(folder, 'quality', digest)[1], list(range(4096)))
            with self.assertRaises(ValueError):
                subject.load_inputs(folder, 'quality', '0' * 64)
            for field, bad in [('capacity', 4097), ('targets', 1023), ('score_rows', [3072, 4096]),
                               ('score_target_indices', [3071, 4095]), ('quality_global_token_start', 10000),
                               ('quality_mean_nll_increase_limit', 0.06), ('numerical_parent_manifest_sha256', '0' * 64)]:
                changed = copy.deepcopy(manifest)
                changed[field] = bad
                with self.assertRaisesRegex(ValueError, 'recipe'):
                    subject.load_inputs(folder, 'quality', freeze(changed))
            for bad in (raw[:-4], raw + bytes(4), struct.pack('<I', 151936) + raw[4:]):
                (folder / 'Q.u32le').write_bytes(bad)
                changed = copy.deepcopy(manifest)
                changed['files']['Q.u32le'] = dict(bytes=len(bad), sha256=hashlib.sha256(bad).hexdigest())
                with self.assertRaises((ValueError, OSError)):
                    subject.load_inputs(folder, 'quality', freeze(changed))

    def test_receipt_requires_completed_same_candidate(self):
        receipt = dict(complete=True, mode='equivalence', profile=subject.PROFILE,
                       model_pins=subject.baseline.PINS, input_manifest_sha256='a' * 64,
                       producer_sha256='b' * 64, equivalence_passed=True, cleanup_errors=[])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'receipt.json'

            def check(document):
                raw = json.dumps(document).encode()
                path.write_bytes(raw)
                subject.require_equivalence(path, hashlib.sha256(raw).hexdigest(), 'a' * 64, 'b' * 64)

            check(receipt)
            for key, bad in [('complete', False), ('mode', 'quality'), ('profile', 'other'),
                             ('producer_sha256', 'c' * 64), ('input_manifest_sha256', 'c' * 64),
                             ('equivalence_passed', False), ('cleanup_errors', ['failed']), ('model_pins', {})]:
                changed = copy.deepcopy(receipt)
                changed[key] = bad
                with self.assertRaises(ValueError):
                    check(changed)
            with self.assertRaises(ValueError):
                subject.require_equivalence(None, None, 'a' * 64, 'b' * 64)

    def test_adapter_delegates_checks_before_mutation_and_returns_fresh_view(self):
        events = []
        mx = SimpleNamespace(bfloat16='bf16', eval=lambda *args: events.append(('eval', args)),
                             synchronize=lambda: events.append(('sync',)))

        class Stock:
            step = 256

            def __init__(self):
                self.offset = 0
                self.keys = 'full-keys'
                self.values = 'full-values'

            def update_and_fetch(self, keys, values):
                events.append(('stock', self.offset, keys.shape[2]))
                self.offset += keys.shape[2]
                return 'discarded-lazy-views'

            def keys_and_values(self):
                events.append(('fresh',))
                return 'fresh-views'

        fixed = subject.adapted_cache_class(mx, Stock)()
        keys = SimpleNamespace(shape=(1, 8, 8, 128), dtype='bf16')
        self.assertEqual(fixed.update_and_fetch(keys, keys), 'fresh-views')
        self.assertEqual([event[0] for event in events], ['stock', 'eval', 'sync', 'fresh'])
        self.assertEqual(events[1][1], ('full-keys', 'full-values'))
        self.assertEqual((Stock.step, fixed.step, fixed.offset), (256, 4096, 8))
        fixed.offset = 4088
        fixed.update_and_fetch(keys, keys)
        self.assertEqual(fixed.offset, 4096)
        before = list(events)
        for bad in (keys, SimpleNamespace(shape=(1, 8, 0, 128), dtype='bf16'),
                    SimpleNamespace(shape=(1, 8, 9, 128), dtype='bf16')):
            with self.assertRaises(ValueError):
                fixed.update_and_fetch(bad, bad)
        self.assertEqual((fixed.offset, events), (4096, before))
        fixed.offset = 0
        for shape, dtype in [((1, 16, 8, 128), 'bf16'), ((1, 8, 8, 64), 'bf16'), ((1, 8, 8, 128), 'fp16')]:
            bad = SimpleNamespace(shape=shape, dtype=dtype)
            with self.assertRaises(ValueError):
                fixed.update_and_fetch(bad, bad)
        self.assertEqual(fixed.offset, 0)

    def test_forward_mask_before_updates_barriers_and_whole_batch_head(self):
        events = []
        states = [SimpleNamespace(offset=253), SimpleNamespace(offset=253)]
        mx = SimpleNamespace(eval=lambda value: events.append(('eval', value)),
                             synchronize=lambda: events.append(('sync',)))

        class Embedding:
            def __call__(self, inputs):
                events.append(('embed', inputs))
                return ('whole-hidden', 8)

            def as_linear(self, hidden):
                events.append(('head', hidden))
                return 'whole-output'

        def mask(hidden, state):
            events.append(('mask', state.offset))
            return ('mask-at', state.offset)

        def layer(hidden, mask_value, state):
            events.append(('layer', mask_value, state.offset))
            state.offset += 8
            return hidden

        def norm(hidden):
            events.append(('norm', hidden))
            return hidden

        model = SimpleNamespace(model=SimpleNamespace(embed_tokens=Embedding(), layers=[layer, layer], norm=norm))
        self.assertEqual(subject.adapted_forward(model, 'inputs', states, mx, mask, True), 'whole-output')
        self.assertEqual([event[0] for event in events],
                         ['embed', 'mask', 'layer', 'eval', 'sync', 'layer', 'eval', 'sync', 'norm', 'eval', 'sync', 'head', 'eval', 'sync'])
        self.assertEqual([event for event in events if event[0] == 'layer'],
                         [('layer', ('mask-at', 253), 253), ('layer', ('mask-at', 253), 253)])
        self.assertIn(('head', ('whole-hidden', 8)), events)
        events.clear()
        self.assertIsNone(subject.adapted_forward(model, 'inputs', states, mx, mask, False))
        self.assertFalse(any(event[0] == 'head' for event in events))
        self.assertEqual([state.offset for state in states], [269, 269])

    def test_cache_storage_offsets_and_dtype_are_exact(self):
        def cache(offset, capacity, step):
            return [SimpleNamespace(offset=offset, step=step,
                                    keys=SimpleNamespace(shape=(1, 8, capacity, 128), dtype='bf16'),
                                    values=SimpleNamespace(shape=(1, 8, capacity, 128), dtype='bf16')) for _ in range(28)]
        for offset, stock_capacity in [(255, 256), (256, 256), (257, 512), (1024, 1273), (1025, 1273)]:
            subject.validate_cache(cache(offset, stock_capacity, 256), offset, False, 'bf16', stock_capacity)
            subject.validate_cache(cache(offset, 4096, 4096), offset, True, 'bf16', 4096)
        for field, bad in [('offset', 1024), ('step', 256), ('keys', SimpleNamespace(shape=(1, 8, 4096, 128), dtype='fp16'))]:
            values = cache(1025, 4096, 4096)
            setattr(values[7], field, bad)
            with self.assertRaises(ValueError):
                subject.validate_cache(values, 1025, True, 'bf16', 4096)
        before = subject.cache_shapes(cache(505, 512, 256))
        events = []
        with self.assertRaises(ValueError):
            subject.validate_cache(cache(513, 768, 256), 513, False, 'bf16', 761, before, events.append)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['event'], 'cache_validation_failed')
        self.assertEqual((events[0]['expected_offset'], events[0]['expected_capacity']), (513, 761))
        self.assertEqual(events[0]['before'][0]['keys']['shape'], [1, 8, 512, 128])
        self.assertEqual(events[0]['after'][0]['keys']['shape'], [1, 8, 768, 128])
        for wrong_capacity in (760, 762, 768):
            with self.assertRaises(ValueError):
                subject.validate_cache(cache(513, wrong_capacity, 256), 513, False, 'bf16', 761)

    def test_stock_irregular_schedule_has_exact_allocation_recurrence(self):
        # Literal growth events from the pinned step256 trim/concatenate rule.
        # This independently rejects rounding the new offset up to256.
        growth = []
        capacity = 0
        for start, end in subject.schedule('equivalence'):
            actual = subject.stock_capacity_after_update(capacity, start, end - start)
            expected = (256 if end <= 256 else 512 if end <= 505 else
                        761 if end <= 761 else 1017 if end <= 1017 else 1273)
            self.assertEqual(actual, expected, (start, end))
            if actual != capacity:
                growth.append((start, end, capacity, actual))
            capacity = actual
        self.assertEqual(growth, [(0, 8, 0, 256), (256, 257, 256, 512),
                                  (505, 513, 512, 761), (761, 769, 761, 1017),
                                  (1017, 1024, 1017, 1273)])
        self.assertEqual(capacity, 1273)
        for allocated, offset, rows, expected in [(256, 255, 1, 256), (512, 512, 1, 768),
                                                  (257, 256, 8, 513), (1273, 1024, 1, 1273)]:
            self.assertEqual(subject.stock_capacity_after_update(allocated, offset, rows), expected)
        for arguments in [(-1, 0, 1), (256, 257, 1), (0, 0, 0), (0, 0, 9),
                          (True, 0, 1), (4096, 4096, 1)]:
            with self.assertRaises(ValueError):
                subject.stock_capacity_after_update(*arguments)


if __name__ == '__main__':
    unittest.main()
