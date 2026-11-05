import hashlib
import json
import unittest

import numpy as np

import check_gemma_attention_boundary as witness


class AttentionBoundaryTests(unittest.TestCase):
    @staticmethod
    def fixture():
        body = bytearray()
        events = [('REAL_ATTENTION_BOUNDARY', witness.METADATA.copy())]
        for index, (name, start, rows, count) in enumerate(witness.ARRAYS):
            values = np.full(count, 1 if name.endswith(('.v', '.context')) else 0, dtype='<f4').tobytes()
            spec = dict(record=index, name=name, before=start, count=rows,
                        offset=len(body), bytes=len(values), values=count)
            events.append(('ARRAY_ATTEMPT', spec))
            events.append(('ARRAY', dict(spec, sha256=hashlib.sha256(values).hexdigest())))
            body.extend(values)
        events.append(('REAL_ATTENTION_BOUNDARY_COMPLETE',
                       dict(committed=448, fault_preserved_prefix=True, committed_operands_equal=True)))
        events.append(('CAPTURE', dict(records=len(witness.ARRAYS), bytes=len(body),
                                       sha256=hashlib.sha256(body).hexdigest(), whole_e2b_qualified=False)))
        return events, bytes(body)

    @staticmethod
    def transcript(events):
        return '\n'.join(f'GEMMA_HYBRID_{name} {json.dumps(value)}' for name, value in events).encode()

    def test_exact_operands_and_corrupted_value_replay(self):
        class Projection:
            def linear(self, name, context, dtype):
                if name != 'layers.12.self_attn.o_proj.weight' or not np.all(context == 1):
                    raise AssertionError('Projection must consume the unchanged actual context')
                return np.zeros((1, 1536), dtype=dtype)

        events, body = self.fixture()
        arrays = witness.native_operands(self.transcript(events), body)
        projection = Projection()
        clean = witness.compare(arrays, projection)
        self.assertTrue(all(stage['failed_values'] == 0 for item in clean.values() for stage in item.values()))
        corrupted = np.frombuffer(arrays['attention.0.v'], dtype='<f4').copy()
        corrupted[0] = 64.
        arrays['attention.0.v'] = corrupted.tobytes()
        bad = witness.compare(arrays, projection)
        self.assertTrue(all(item['context']['failed_values'] == 8 for item in bad.values()))
        self.assertTrue(all(item['projection']['failed_values'] == 0 for item in bad.values()))

    def test_shifted_span_wrong_producer_wrong_head_and_raw_corruption_rejected(self):
        for mutation in ('span', 'producer', 'head', 'raw'):
            with self.subTest(mutation=mutation):
                events, body = self.fixture()
                if mutation == 'span':
                    events[7][1]['before'] += 1
                elif mutation == 'producer':
                    events[0][1]['producer'] = 13
                elif mutation == 'head':
                    events[0][1]['kv_heads'] = 2
                else:
                    body = bytes([1]) + body[1:]
                with self.assertRaises(ValueError):
                    witness.native_operands(self.transcript(events), body)
