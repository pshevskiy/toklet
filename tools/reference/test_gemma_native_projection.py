"""Explicit cold qualification; set TOKLET_GEMMA_NATIVE_PROJECTION_LIBRARY."""
from fractions import Fraction
import math
import os
import struct
import unittest
import gemma_native_projection as subject
from check_gemma_affine_admission import rounded


def binary(a, b, multiply=False):
    value = Fraction(a)*Fraction(b) if multiply else Fraction(a)+Fraction(b)
    if value == 0:
        negative = ((math.copysign(1, a)*math.copysign(1, b) < 0) if multiply
                    else a == b == 0 and math.copysign(1, a) < 0 and math.copysign(1, b) < 0)
        return -0. if negative else 0.
    return rounded(value)


def fused(a, b, c):
    value = Fraction(a)*Fraction(b)+Fraction(c)
    if value == 0:
        return binary(binary(a, b, True), c)
    return rounded(value)


def exact_replay(codes, scales, biases, inputs, tokens, rows, columns, lanes, residual=True):
    """Rational RNE at every operation, no native library or target output."""
    outputs = []
    for token in range(tokens):
        for row in range(rows):
            high, low = [0.]*lanes, [0.]*lanes
            def add(lane, value):
                a = high[lane]
                next_value = binary(a, value)
                correction = (binary(binary(a, -next_value), value) if abs(a) >= abs(value)
                              else binary(binary(value, -next_value), a))
                low[lane] = binary(low[lane], correction)
                high[lane] = next_value
            for lane in range(lanes):
                for group in range(columns//64):
                    index = row*(columns//64)+group
                    for local in range(lane, 64, lanes):
                        column = group*64+local
                        weight = fused(scales[index], codes[row*columns+column], biases[index])
                        value = inputs[token*columns+column]
                        product = binary(weight, value, True)
                        add(lane, product)
                        if residual:
                            add(lane, fused(weight, value, -product))
            hop = 1
            while hop < lanes:
                for lane in range(0, lanes-hop, hop*2):
                    add(lane, high[lane+hop])
                    add(lane, low[lane+hop])
                hop *= 2
            outputs.append(binary(high[0], low[0]))
    return struct.pack('<'+'f'*len(outputs), *outputs)


def packed(codes, scales, biases, inputs):
    return (bytes(a+16*b for a, b in zip(codes[::2], codes[1::2], strict=True)),
            b''.join(struct.pack('<H', struct.unpack('<I', struct.pack('<f', s))[0] >> 16) for s in scales),
            b''.join(struct.pack('<H', struct.unpack('<I', struct.pack('<f', b))[0] >> 16) for b in biases),
            struct.pack('<'+'f'*len(inputs), *inputs))


class ProjectionControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        library = os.environ.get('TOKLET_GEMMA_NATIVE_PROJECTION_LIBRARY')
        if not library:
            raise unittest.SkipTest('Explicit compiled projection oracle library required')
        cls.project = subject.Projection(library)

    def check(self, codes, scales, biases, inputs, tokens, rows, columns):
        wire = packed(codes, scales, biases, inputs)
        for lanes in [1, 2, 8, 16, 32, 64]:
            with self.subTest(lanes=lanes):
                expected = exact_replay(codes, scales, biases, inputs, tokens, rows, columns, lanes)
                self.assertEqual(self.project(*wire, tokens=tokens, rows=rows, columns=columns, lanes=lanes), expected)

    def test_literal_nibbles_groups_rows_and_tokens(self):
        codes = [(i*7+i//64*3) % 16 for i in range(256)]
        self.check(codes, [.125, -.25, .5, -.125], [.25, -.5, 1., .5],
                   [((i*11)%31-15)/16 for i in range(256)], 2, 2, 128)
        wire = packed(codes, [.125]*4, [.25]*4, [1.]*128)
        self.assertEqual(wire[0][:4], bytes.fromhex('705e3c1a'))

    def test_product_residual_is_required_under_cancellation(self):
        codes, scales, biases = [0]*64, [0.], [1.0078125]
        inputs = [1+2**-23, -1.]+[0.]*62
        self.check(codes, scales, biases, inputs, 1, 1, 64)
        correct = exact_replay(codes, scales, biases, inputs, 1, 1, 64, 32)
        self.assertNotEqual(correct, exact_replay(codes, scales, biases, inputs, 1, 1, 64, 32, False))

    def test_signed_zero_subnormal_and_rounding_boundaries(self):
        for inputs in [[-0.]*64, [2**-149]+[0.]*63,
                       [1., 2**-24]+[0.]*62, [1., 3*2**-24]+[0.]*62]:
            self.check([0]*64, [0.], [1.], inputs, 1, 1, 64)

    def test_invalid_shape_bytes_nonfinite_and_arithmetic_overflow_reject(self):
        wire = packed([0]*64, [0.], [1.], [1.]*64)
        options = dict(tokens=1, rows=1, columns=64, lanes=32)
        for change in [dict(lanes=3), dict(lanes=128), dict(tokens=33), dict(rows=129), dict(columns=65), dict(tokens=True)]:
            with self.assertRaises(ValueError):
                self.project(*wire, **dict(options, **change))
        for index, bad in [(0, wire[0][:-1]), (1, bytes.fromhex('807f')), (3, struct.pack('<f', math.nan)+wire[3][4:])]:
            blocks = list(wire)
            blocks[index] = bad
            with self.assertRaises(ValueError):
                self.project(*blocks, **options)
        overflow = packed([0]*64, [0.], [2.**100], [2.**100]*64)
        with self.assertRaises(ValueError):
            self.project(*overflow, **options)
