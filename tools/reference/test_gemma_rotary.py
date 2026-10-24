"""Offline mathematical contract tests; no NumPy or Apple runtime required."""
import copy
import hashlib
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest

import gemma_rotary as oracle


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class GemmaRotary(unittest.TestCase):
    def test_hand_pairs_and_full_dimension_denominator(self):
        x = [0.] * 16
        x[0], x[8], x[1], x[9] = 2., 3., 4., 5.
        before = list(x)
        frequencies, y = oracle.math_values(x, 1, 1, 16, 4, 2, 256., 1.)
        self.assertEqual(frequencies, [1., 2.])
        # Exact chosen exponents make angles 2 and 1, with independent literals.
        cosine2, sine2 = -0.4161468365471424, 0.9092974268256817
        cosine1, sine1 = 0.5403023058681398, 0.8414709848078965
        for index, expected in [(0, 2*cosine2-3*sine2), (8, 2*sine2+3*cosine2),
                (1, 4*cosine1-5*sine1), (9, 4*sine1+5*cosine1)]:
            self.assertAlmostEqual(y[index], expected, places=14)
        self.assertEqual(x, before)
        self.assertGreater(abs(y[1] - (4*math.cos(.125)-5*math.sin(.125))), 1.)
        self.assertEqual(y[2:8], x[2:8])
        self.assertEqual(y[10:], x[10:])

    def test_empty_zero_rotation_and_zero_position_copy(self):
        frequencies, output = oracle.math_values([], 0, 2, 8, 2, 0, 1e6, 1.)
        self.assertEqual(frequencies, [1.])
        self.assertEqual(output, [])
        x = [float(i-8) for i in range(16)]
        self.assertEqual(oracle.math_values(x, 1, 2, 8, 0, 131071, 1e6, 1.)[1], x)
        self.assertEqual(oracle.math_values(x, 1, 2, 8, 8, 0, 1e6, 1.)[1], x)
        x[2], x[10] = -0., -0.
        for rotated_dims in (0, 2):
            y = oracle.math_values(x, 1, 2, 8, rotated_dims, 17, 1e6, 1.)[1]
            for index in (2, 10):
                self.assertEqual(struct.pack('<d', y[index]), struct.pack('<d', x[index]))

    def test_chunk_offsets_and_multiple_heads(self):
        x = [(((i*11+3)%29)-14)*.125 for i in range(5*3*16)]
        before = list(x)
        full = oracle.math_values(x, 5, 3, 16, 4, 131067, 1e6, 2.5)[1]
        for chunk in (1, 2, 3):
            joined = []
            for first in range(0, 5, chunk):
                count = min(chunk, 5-first)
                joined += oracle.math_values(x[first*48:(first+count)*48], count, 3, 16, 4, 131067+first, 1e6, 2.5)[1]
            self.assertEqual(joined, full)
        wrong = oracle.math_values(x[48:96], 1, 3, 16, 4, 131067, 1e6, 2.5)[1]
        self.assertGreater(max(abs(a-b) for a,b in zip(wrong, full[48:96])), .01)
        self.assertEqual(x, before)

    def test_all_fixture_pairs_preserve_norm_and_inactive_coordinates(self):
        data = json.loads(oracle.export())
        for row in data['cases']:
            x = row['arrays']['input']['values']
            y = row['arrays']['output']['values']
            dims, pairs = row['dims'], row['rotated_dims']//2
            for first in range(0, len(x), dims):
                for pair in range(dims//2):
                    a, b = first+pair, first+pair+dims//2
                    if pair >= pairs:
                        self.assertEqual([x[a], x[b]], [y[a], y[b]])
                    else:
                        expected = x[a]*x[a]+x[b]*x[b]
                        self.assertLessEqual(abs(y[a]*y[a]+y[b]*y[b]-expected), 1e-13+1e-13*expected)
            self.assertTrue(all(math.isfinite(v) for v in y))

    def test_geometry_positions_precision_and_nonfinite_inputs_reject(self):
        good = [1, 1, 16, 4, 0, 256., 1.]
        bad = {0: [-1, 9, True, 1.], 1: [0, 5, True], 2: [0, 3, 514],
            3: [-2, 3, 18], 4: [-1, (1<<53)+1, True],
            5: [0., .5, .1, float('nan'), float('inf'), True, 10**400],
            6: [0., .1, 9., float('-inf'), True]}
        for index, cases in bad.items():
            for value in cases:
                args = list(good)
                args[index] = value
                with self.subTest(index=index, value=str(value)), self.assertRaises(ValueError):
                    oracle.math_values([1.]*16, *args)
        with self.assertRaises(ValueError):
            oracle.math_values([1.]*48, 3, 1, 16, 4, 1<<53, 256., 1.)
        for values in ([1.]*15, [1.]*17, tuple([1.]*16)):
            with self.assertRaises(ValueError):
                oracle.math_values(values, *good)
        for value in (float('nan'), float('inf'), float('-inf'), True, .1, 10**400):
            x = [1.]*16
            x[-1] = value
            with self.assertRaises(ValueError):
                oracle.math_values(x, *good)

    def test_bounded_external_fixture_identity_and_format(self):
        raw = oracle.export()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'math.json'
            path.write_bytes(raw)
            self.assertEqual(len(oracle.verify(path, sha(raw))['cases']), len(oracle.SPECS))
            with self.assertRaises(ValueError):
                oracle.verify(path, '0'*64)
            base = json.loads(raw)
            mutations = []
            for key,value in [('upstream_parity_qualified', True), ('extra', 1), ('precision', 'other')]:
                altered = copy.deepcopy(base)
                altered[key] = value
                mutations.append(altered)
            for target,key,value in [('row','tokens',True), ('row','dims',15), ('array','shape',[True,1,8]),
                    ('array','values',[.1]*8), ('array','dtype','<f8')]:
                altered = copy.deepcopy(base)
                row = altered['cases'][1]
                (row if target=='row' else row['arrays']['input'])[key] = value
                mutations.append(altered)
            altered = copy.deepcopy(base)
            altered['cases'][1]['arrays']['output']['values'][0] = 10**400
            mutations.append(altered)
            for altered in mutations:
                changed = oracle.encoded(altered)
                path.write_bytes(changed)
                with self.assertRaises(ValueError):
                    oracle.verify(path, sha(changed))
            path.write_bytes(b' '*(oracle.MAX_JSON+1))
            with self.assertRaises(ValueError):
                oracle.verify(path, sha(path.read_bytes()))
            alias = path.parent/'alias'
            alias.symlink_to(path)
            with self.assertRaises(ValueError):
                oracle.verify(alias, sha(path.read_bytes()))


if __name__ == '__main__':
    unittest.main()
