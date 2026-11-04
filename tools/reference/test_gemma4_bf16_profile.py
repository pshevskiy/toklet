"""Tiny equivalence and failure controls; no model download or Apple GPU."""
from pathlib import Path
import hashlib
import tempfile
import time
import unittest
import numpy as np
import bf16_source
import gemma4_bf16_profile as profile
import gemma4_text_math as graph
import test_gemma4_text_math as fixture


class ProfileControls(unittest.TestCase):
    def test_complete_graph_bytes_and_source_chain_are_unchanged(self):
        packed, _, _ = fixture.fixture()
        raw = bytearray(); bindings = {}
        for name, shape in graph.shapes(graph.WITNESS).items():
            if len(shape) != 2:
                continue
            values = ((np.arange(np.prod(shape), dtype=np.float32) % 37 - 18) / 128).reshape(shape)
            bindings[name] = (len(raw), shape)
            raw.extend((values.view('<u4') >> 16).astype('<u2').tobytes())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'source.bin'; path.write_bytes(raw)
            expected = {}; audit = None
            for instrumented in [False, True]:
                deadline = time.monotonic()+60
                with bf16_source.Source(path, len(raw), hashlib.sha256(raw).hexdigest(), deadline=deadline) as source:
                    matrices = {name: bf16_source.Rows(source, *shape, offset, deadline)
                                for name, (offset, shape) in bindings.items()}
                    weights = graph.Weights(graph.WITNESS, matrices, packed.vectors)
                    ledger = profile.Ledger()
                    if instrumented:
                        weights = profile.Weights(weights, ledger)
                    for staged in [False, True]:
                        actual = {}
                        graph.evaluate(weights, [2, 7, 13, 7], staged,
                                       sink=lambda name, value: actual.update({name: value.tobytes()}))
                        if instrumented:
                            self.assertEqual(actual, expected[staged])
                        else:
                            expected[staged] = actual
                    if instrumented:
                        self.assertEqual(source.audit(), audit)
                        rows = ledger.records()
                        self.assertEqual(sum(v['bytes_returned'] for v in rows if v['operation']=='read_guard_hash'), audit['read_bytes'])
                        self.assertTrue(all(v['elapsed_ns'] >= 0 and v['failed_calls']==0 for v in rows))
                        self.assertTrue(any(v['operation']=='tied_head_inclusive' for v in rows))
                        self.assertTrue(any(v['operation']=='projection_fp64_conversion_compute_framework' for v in rows))
                        self.assertTrue(all(a.inner is matrices[n] for n, a in weights.matrices.items()))
                        self.assertTrue(all(a is packed.vectors[n] for n, a in weights.vectors.items()))
                    else:
                        audit = source.audit()

    def test_gather_and_rejected_request_preserve_reader_behavior(self):
        raw = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype='<f4')
        body = (raw.view('<u4') >> 16).astype('<u2').tobytes()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'source.bin'; path.write_bytes(body)
            audits = []
            for instrumented in [False, True]:
                deadline = time.monotonic()+60
                with bf16_source.Source(path, len(body), hashlib.sha256(body).hexdigest(), deadline=deadline) as source:
                    rows = bf16_source.Rows(source, 3, 3, 0, deadline)
                    ledger = profile.Ledger()
                    if instrumented:
                        rows = profile.Rows('tiny', rows, ledger)
                    np.testing.assert_array_equal(rows.gather([2, 0, 2]), raw[[2, 0, 2]])
                    before = source.audit()
                    with self.assertRaises(ValueError):
                        rows.read_rows(0, 4)
                    self.assertEqual(source.audit(), before)
                    self.assertFalse(source.poisoned)
                    audits.append(before)
                    if instrumented:
                        read = next(v for v in ledger.records() if v['operation']=='read_guard_hash')
                        self.assertEqual((read['calls'], read['failed_calls'], read['bytes_returned']), (2, 1, 18))
            self.assertEqual(audits[0], audits[1])


if __name__ == '__main__':
    unittest.main()
