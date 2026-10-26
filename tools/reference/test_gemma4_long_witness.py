"""Frozen short parity and bounded admission for the long batch-only oracle."""
import hashlib
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock
import numpy as np
import gemma4_long_witness as long
import test_gemma4_text_math as old


class LongWitness(unittest.TestCase):
    def test_frozen_historical_source_and_every_short_state(self):
        source = Path(old.graph.__file__).read_bytes()
        self.assertEqual(hashlib.sha256(source).hexdigest(), long.ORIGINAL_SOURCE_SHA256)
        weights, manifest, raw = old.fixture()
        for staged, contract in [(False, 'math'), (True, 'staged_fp32')]:
            actual = long.evaluate(weights, manifest['token_ids'], staged)
            self.assertEqual(set(actual), set(manifest[contract]))
            for name, item in manifest[contract].items():
                body = raw[item['offset']:item['offset']+item['bytes']]
                expected = np.frombuffer(body, dtype=item['dtype']).reshape(item['shape'])
                old.gate(actual[name], expected)

    def test_uneven_projection_gather_batches_preserve_frozen_causal_prefix(self):
        weights, manifest, raw = old.fixture()
        ids = manifest['token_ids'] + [3]
        for staged, contract in [(False, 'math'), (True, 'staged_fp32')]:
            captured = {}; gathers = []; projections = []
            linear, gather = weights.linear, weights.gather
            def checked_linear(name, value, dtype):
                projections.append(len(value))
                self.assertLessEqual(len(value), 32)
                return linear(name, value, dtype)
            def checked_gather(name, values):
                gathers.append(len(values))
                self.assertLessEqual(len(values), 32)
                return gather(name, values)
            with mock.patch.object(weights, 'linear', checked_linear), mock.patch.object(weights, 'gather', checked_gather):
                self.assertEqual(long.evaluate(weights, ids, staged, sink=lambda n,x:captured.setdefault(n,x.copy())), {})
            self.assertEqual(set(gathers), {1, 32})
            self.assertEqual(set(projections), {1, 32})
            self.assertEqual({n:tuple(v.shape) for n,v in captured.items()}, long.stage_shapes(33))
            for name, item in manifest[contract].items():
                body = raw[item['offset']:item['offset']+item['bytes']]
                expected = np.frombuffer(body, dtype=item['dtype']).reshape(item['shape'])
                old.gate(old.prefix(captured[name], name, 32), expected)

    def test_invalid_and_real_profiles_reject_before_numerical_calls(self):
        weights, _, _ = old.fixture()
        for ids, staged, sink in [([],False,None),([0]*1026,False,lambda *_:None),
                                  ([True],False,None),([-1],False,None),([17],False,None),
                                  ([0],1,None),([0]*33,False,None),([0],False,'invalid')]:
            with mock.patch.object(weights, 'gather', side_effect=AssertionError('unexpected gather')):
                with self.assertRaises(ValueError): long.evaluate(weights, ids, staged, sink=sink)
        with self.assertRaisesRegex(ValueError, 'Named frozen WITNESS'):
            long.evaluate(SimpleNamespace(profile=old.graph.E2B), [2], True, sink=lambda *_:None)
        weights.profile = old.graph.E2B
        with self.assertRaisesRegex(ValueError, 'Named frozen WITNESS'):
            long.evaluate(weights, [2], True, sink=lambda *_:None)
        for invalid in [0,1026,True]:
            with self.assertRaises(ValueError): long.stage_shapes(invalid)
        self.assertEqual(long.stage_shapes(1025)['layer.1.context'], (1,2,1025,128))
        self.assertEqual(sum(long.LEDGER_MIB.values()), 512)

    def test_sink_failure_keeps_source_and_input_and_preserves_completed_events(self):
        weights, manifest, _ = old.fixture(); ids=manifest['token_ids']+[3]
        before=weights.matrices['embed_tokens.weight'].stream.getvalue(); original=list(ids); events=[]
        def broken(name, value):
            events.append((name, value.shape))
            if name=='layer.2.output': raise OSError('injected long witness sink failure')
        with self.assertRaisesRegex(OSError, 'injected long witness sink failure'):
            long.evaluate(weights, ids, True, sink=broken)
        self.assertEqual(events[-1][0], 'layer.2.output')
        self.assertNotIn('logits', [n for n,_ in events])
        self.assertEqual(ids, original)
        self.assertEqual(weights.matrices['embed_tokens.weight'].stream.getvalue(), before)


if __name__ == '__main__': unittest.main()
