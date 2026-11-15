"""Independent literals and exact dyadic controls; no model, GPU or converter run."""
import builtins
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import qwen35_conversion_math as arithmetic
import qwen35_parent_correspondence as driver

FIXTURES = Path(__file__).resolve().parents[2] / 'tests/fixtures/qwen35-huihui-conversion-v1'
AFFINE = json.loads((FIXTURES / 'affine-goldens.json').read_bytes())['groups']
BF16 = json.loads((FIXTURES / 'bf16-goldens.json').read_bytes())
PROFILES = ('cpu_rint', 'metal_round')
COVERAGE = dict(parent_arrays=426, matrices=249, unquantized_arrays=177,
                child_descriptors=924, groups=65699840, matrix_row_blocks=10820,
                unquantized_elements=961536)


def encode(value):
    return json.dumps(value, separators=(',', ':')).encode()


def words(values):
    return np.array([int(v, 16) for v in values], dtype=np.uint16)


def exact_norm_sum(word):
    """Round exact BF16+1 dyadic integer units, not producer FP32 bit arithmetic."""
    exponent, fraction = (word >> 7) & 255, word & 127
    if exponent == 255:
        return word if fraction == 0 else 0x7FC0
    units = fraction if exponent == 0 else (128 + fraction) << (exponent - 1)
    if word & 0x8000:
        units = -units
    units += 1 << 133
    if units == 0:
        return 0
    sign, magnitude = (0x8000 if units < 0 else 0), abs(units)
    shift = max(0, magnitude.bit_length() - 8)
    quotient, remainder = divmod(magnitude, 1 << shift)
    twice = remainder * 2
    if twice > 1 << shift or (twice == 1 << shift and quotient % 2):
        quotient += 1
    if quotient == 256:
        quotient, shift = 128, shift + 1
    if shift == 0 and quotient < 128:
        return sign | quotient
    exponent = shift + 1
    return sign | (0x7F80 if exponent >= 255 else (exponent << 7) | (quotient - 128))


class BFloat16Tests(unittest.TestCase):
    def test_literal_rne_ties_neighbors_signed_zero_boundaries_and_nan_policy(self):
        source = np.array([int(row['f32_bits'], 16) for row in BF16['f32_to_bf16']], dtype=np.uint32)
        expected = words([row['bf16_bits'] for row in BF16['f32_to_bf16']])
        saved = source.copy()
        actual = arithmetic.f32_to_bf16(source.view(np.float32))
        self.assertEqual(actual.dtype, np.dtype(np.uint16))
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(source, saved)

    def test_bit_widening_preserves_sign_subnormal_infinity_and_nan_payload_bits(self):
        source = words(['0000', '8000', '0001', '007f', '0080', '3f80', '7f7f', 'ff7f',
                        '7f80', 'ff80', '7f81', 'ff81', '7fff', 'ffff'])
        expected = np.array([0x00000000, 0x80000000, 0x00010000, 0x007F0000,
            0x00800000, 0x3F800000, 0x7F7F0000, 0xFF7F0000, 0x7F800000, 0xFF800000,
            0x7F810000, 0xFF810000, 0x7FFF0000, 0xFFFF0000], dtype=np.uint32)
        actual = arithmetic.bf16_to_f32(source)
        self.assertEqual(actual.dtype, np.dtype(np.float32))
        np.testing.assert_array_equal(actual.view(np.uint32), expected)

    def test_literal_typed_norm_shift_ties_cancellation_and_software_nonfinite(self):
        source = words([row['input'] for row in BF16['add_one']])
        expected = words([row['output'] for row in BF16['add_one']])
        with np.errstate(invalid='ignore'):
            actual = arithmetic.add_one_bf16(source)
        np.testing.assert_array_equal(actual, expected)

    def test_all_65280_finite_norm_words_match_independent_exact_dyadic_reference(self):
        finite = [word for word in range(65536) if word & 0x7F80 != 0x7F80]
        self.assertEqual(len(finite), 65280)
        expected = np.array([exact_norm_sum(word) for word in finite], dtype=np.uint16)
        source = np.array(finite, dtype=np.uint16)
        saved = source.copy()
        np.testing.assert_array_equal(arithmetic.add_one_bf16(source), expected)
        np.testing.assert_array_equal(source, saved)

    def test_all_256_nonfinite_words_have_explicit_software_add_policy(self):
        nonfinite = [word for word in range(65536) if word & 0x7F80 == 0x7F80]
        self.assertEqual(len(nonfinite), 256)
        source = np.array(nonfinite, dtype=np.uint16)
        expected = np.array([word if word & 127 == 0 else 0x7FC0 for word in nonfinite], dtype=np.uint16)
        with np.errstate(invalid='ignore'):
            np.testing.assert_array_equal(arithmetic.add_one_bf16(source), expected)

    def test_scalar_empty_and_strided_conversion_shape_is_preserved_without_mutation(self):
        for source in [np.array(0x3F80, dtype=np.uint16), np.empty((2, 0), dtype=np.uint16),
                       np.full((4, 6), 0x3F80, dtype=np.uint16)[::2, ::-2]]:
            before = source.copy()
            widened = arithmetic.bf16_to_f32(source)
            shifted = arithmetic.add_one_bf16(source)
            self.assertEqual(widened.shape, source.shape)
            self.assertEqual(shifted.shape, source.shape)
            np.testing.assert_array_equal(source, before)
            np.testing.assert_array_equal(arithmetic.f32_to_bf16(widened), source)

    def test_exact_dtypes_and_no_implicit_materialization(self):
        class NoCast:
            def __array__(self, *args, **kwargs):
                raise AssertionError('public boundary must reject before implicit casting')
        for function, good in [(arithmetic.bf16_to_f32, np.uint16),
                               (arithmetic.f32_to_bf16, np.float32),
                               (arithmetic.add_one_bf16, np.uint16)]:
            for invalid in [[], [0], NoCast(), np.array([0], dtype=np.uint8),
                            np.array([0], dtype=np.float64), np.array([0], dtype=np.int16)]:
                with self.subTest(function=function.__name__, invalid=type(invalid)), self.assertRaises((ValueError, TypeError)):
                    function(invalid)
            self.assertEqual(function(np.zeros((1,), dtype=good)).shape, (1,))

    def test_conversion_element_bound_is_checked(self):
        for function, dtype in [(arithmetic.bf16_to_f32, np.uint16),
                                (arithmetic.f32_to_bf16, np.float32),
                                (arithmetic.add_one_bf16, np.uint16)]:
            with self.subTest(function=function.__name__), self.assertRaises((ValueError, TypeError)):
                function(np.zeros((65537,), dtype=dtype))


class AffineGroupTests(unittest.TestCase):
    def check_group(self, source, profile, golden):
        saved = source.copy()
        result = arithmetic.quantize_groups(source, profile)
        self.assertIs(type(result), tuple)
        self.assertEqual(len(result), 3)
        codes, scales, biases = result
        self.assertEqual(codes.dtype, np.dtype(np.uint32))
        self.assertEqual(scales.dtype, np.dtype(np.uint16))
        self.assertEqual(biases.dtype, np.dtype(np.uint16))
        self.assertEqual(codes.shape, (source.shape[0], 8))
        self.assertEqual(scales.shape, biases.shape)
        self.assertEqual(scales.shape, (source.shape[0],))
        pattern = [int(value, 16) for value in golden['word_pattern']]
        expected = np.array((pattern * (8 // len(pattern))), dtype=np.uint32)
        np.testing.assert_array_equal(codes, np.tile(expected, (source.shape[0], 1)))
        np.testing.assert_array_equal(scales, np.full(source.shape[0], int(golden['scale'], 16), dtype=np.uint16))
        np.testing.assert_array_equal(biases, np.full(source.shape[0], int(golden['bias'], 16), dtype=np.uint16))
        np.testing.assert_array_equal(source, saved)
        return result

    def test_literal_cpu_metal_codes_signed_metadata_negative_only_epsilon_and_packing(self):
        for row in AFFINE:
            pattern = row['input_pattern']
            source = words(pattern * (64 // len(pattern))).reshape(1, 64)
            source.flags.writeable = False
            for profile in PROFILES:
                with self.subTest(group=row['name'], profile=profile):
                    codes, _, _ = self.check_group(source, profile, row[profile])
                    if row['name'] == 'code_half_and_unrounded_metadata':
                        self.assertEqual((int(codes[0, 0]) >> 8) & 15, 8)
                        self.assertNotEqual((int(codes[0, 0]) >> 8) & 15, 7)
                    if row['name'] == 'low_to_high_nibble_and_word_order':
                        self.assertEqual(codes.astype('<u4').tobytes().hex(), 'efcdab8967452301' * 4)

    def test_group_batch_and_row_tail_boundaries_preserve_independent_groups(self):
        row = AFFINE[0]
        for count in [1, 7, 32, 128, 160, 256]:
            source = np.tile(words(row['input_pattern'] * 16), (count, 1))
            for profile in PROFILES:
                with self.subTest(groups=count, profile=profile):
                    self.check_group(source, profile, row[profile])
        # Interleaved backing storage verifies values,not accidental contiguous casts.
        backing = np.empty((3, 128), dtype=np.uint16)
        backing[:, ::2] = np.tile(words(row['input_pattern'] * 16), (3, 1))
        backing[:, 1::2] = 0x4000
        self.check_group(backing[:, ::2], 'cpu_rint', row['cpu_rint'])

    def test_mixed_group_metadata_resets_at_each_group(self):
        selected = [AFFINE[0], AFFINE[1], AFFINE[2]]
        source = np.stack([words(row['input_pattern'] * (64 // len(row['input_pattern']))) for row in selected])
        for profile in PROFILES:
            codes, scales, biases = arithmetic.quantize_groups(source, profile)
            for index, row in enumerate(selected):
                expected = [int(x, 16) for x in row[profile]['word_pattern']] * 8
                np.testing.assert_array_equal(codes[index], expected)
                self.assertEqual(int(scales[index]), int(row[profile]['scale'], 16))
                self.assertEqual(int(biases[index]), int(row[profile]['bias'], 16))

    def test_profiles_dtypes_shapes_and_group_resource_bounds_are_rejected(self):
        good = np.zeros((1, 64), dtype=np.uint16)
        for profile in [None, '', 'cpu', 'metal', 'mixed', True]:
            with self.subTest(profile=profile), self.assertRaises((ValueError, TypeError)):
                arithmetic.quantize_groups(good, profile)
        invalid = [[], [[0] * 64], np.zeros((1, 64), dtype=np.float32),
                   np.zeros((1, 64), dtype=np.int16), np.zeros((0, 64), dtype=np.uint16),
                   np.zeros((257, 64), dtype=np.uint16), np.zeros((64,), dtype=np.uint16),
                   np.zeros((1, 63), dtype=np.uint16), np.zeros((1, 65), dtype=np.uint16),
                   np.zeros((1, 1, 64), dtype=np.uint16)]
        for source in invalid:
            with self.subTest(shape=getattr(source, 'shape', None)), self.assertRaises((ValueError, TypeError)):
                arithmetic.quantize_groups(source, 'cpu_rint')

    def test_every_nonfinite_source_and_finite_fp32_range_overflow_is_rejected(self):
        for word in range(65536):
            if word & 0x7F80 != 0x7F80:
                continue
            source = np.full((1, 64), word, dtype=np.uint16)
            for profile in PROFILES:
                with self.subTest(word=hex(word), profile=profile), self.assertRaises(ValueError):
                    arithmetic.quantize_groups(source, profile)
        extreme = words(['7f7f', 'ff7f'] * 32).reshape(1, 64)
        for profile in PROFILES:
            with self.subTest(profile=profile, overflow=True), self.assertRaises(ValueError):
                arithmetic.quantize_groups(extreme, profile)


class ReaderAndCoverageTests(unittest.TestCase):
    def test_decode_rejects_duplicate_nonfinite_nonobject_wrong_type_and_byte_bound(self):
        self.assertEqual(driver.decode_document(b'{"x":1}'), {'x': 1})
        for raw in [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}', b'{"x":1e999}',
                    b'[]', b'null', b'', b'\xff', '{"x":1}', bytearray(b'{}')]:
            with self.subTest(raw=repr(raw)), self.assertRaises((ValueError, UnicodeError)):
                driver.decode_document(raw)
        with self.assertRaises(ValueError):
            driver.decode_document(b'{} ', limit=2)
        for limit in [True, 0, -1, 2**32]:
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                driver.decode_document(b'{}', limit=limit)

    def test_tiny_header_has_exact_complete_nonoverlapping_dtype_spans(self):
        raw = (FIXTURES / 'reader-header.json').read_bytes()
        expected = json.loads(raw)
        self.assertEqual(driver.validate_header(raw, 8 + len(raw) + 12), expected)
        for malformed in [True, -1, 8, 2**63]:
            with self.subTest(container_bytes=malformed), self.assertRaises(ValueError):
                driver.validate_header(raw, malformed)

    def test_header_rejects_aliases_gaps_shapes_metadata_and_descriptor_extensions(self):
        header = json.loads((FIXTURES / 'reader-header.json').read_bytes())
        variants = []
        for field, value in [('dtype', 'I8'), ('shape', []), ('shape', [0]), ('shape', [-1]),
                             ('shape', [True]), ('shape', [1] * 9), ('shape', [2**31]),
                             ('data_offsets', [False, 4]), ('data_offsets', [0, 3])]:
            row = copy.deepcopy(header)
            row['a'][field] = value
            variants.append(row)
        for offsets in [[2, 6], [6, 10]]:
            row = copy.deepcopy(header)
            row['b']['data_offsets'] = offsets
            variants.append(row)
        row = copy.deepcopy(header)
        row['a']['extra'] = 'uninterpreted'
        variants.append(row)
        row = copy.deepcopy(header)
        row['__metadata__']['scope'] = 1
        variants.append(row)
        for row in variants:
            raw = encode(row)
            with self.subTest(header=row), self.assertRaises(ValueError):
                driver.validate_header(raw, 8 + len(raw) + 12)
        too_many = {f'x{i}': dict(dtype='BF16', shape=[1], data_offsets=[2*i, 2*i+2]) for i in range(926)}
        raw = encode(too_many)
        with self.assertRaises(ValueError):
            driver.validate_header(raw, 8 + len(raw) + 1852)

    def test_reader_uses_exact_tiny_ranges_and_maximal_64k_window(self):
        self.assertEqual(driver.read_exact_at(io.BytesIO(b'0123456789'), 2, 4), b'2345')
        self.assertEqual(driver.read_exact_at(io.BytesIO(b'0123456789'), 0, 0), b'')
        self.assertEqual(driver.read_exact_at(io.BytesIO(b'x' * 65536), 0, 65536), b'x' * 65536)

    def test_invalid_read_arguments_are_rejected_before_seek_or_read(self):
        class NoIO:
            def seekable(self):
                raise AssertionError('invalid range must be checked before I/O')
        for offset, count in [(-1, 1), (True, 1), (0, True), (1.0, 1), (0, -1),
                              (0, 65537), (2**63, 1), (2**63-1, 1)]:
            with self.subTest(offset=offset, count=count), self.assertRaises(ValueError):
                driver.read_exact_at(NoIO(), offset, count)

    def test_short_wrong_type_unseekable_and_io_errors_are_finite_rejections(self):
        with self.assertRaises(ValueError):
            driver.read_exact_at(io.BytesIO(b'x'), 0, 2)
        class Source:
            def __init__(self, kind):
                self.kind = kind
            def seekable(self):
                return self.kind != 'unseekable'
            def seek(self, offset):
                if self.kind == 'ioerror':
                    raise OSError('synthetic I/O error')
                return offset + 1 if self.kind == 'seek' else offset
            def read(self, count):
                return bytearray(b'x') if self.kind == 'type' else b'xx'
        for kind in ['unseekable', 'ioerror', 'seek', 'type', 'long']:
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                driver.read_exact_at(Source(kind), 0, 1)

    def test_whole_artifact_profile_requires_full_coverage_and_cannot_mix_tensors(self):
        self.assertEqual(driver.complete_profiles(dict(cpu_rint=0, metal_round=0), 0, COVERAGE), list(PROFILES))
        self.assertEqual(driver.complete_profiles(dict(cpu_rint=0, metal_round=1), 0, COVERAGE), ['cpu_rint'])
        self.assertEqual(driver.complete_profiles(dict(cpu_rint=1, metal_round=0), 0, COVERAGE), ['metal_round'])
        self.assertEqual(driver.complete_profiles(dict(cpu_rint=1, metal_round=1), 0, COVERAGE), [])
        self.assertEqual(driver.complete_profiles(dict(cpu_rint=0, metal_round=0), 1, COVERAGE), [])
        # Positive tiny helper return values above are not actual source admission.

    def test_incomplete_coverage_and_counter_types_cannot_claim_correspondence(self):
        for key in COVERAGE:
            changed = dict(COVERAGE)
            changed[key] -= 1
            with self.subTest(coverage=key), self.assertRaises(ValueError):
                driver.complete_profiles(dict(cpu_rint=0, metal_round=0), 0, changed)
        for counters in [dict(cpu_rint=0), dict(cpu_rint=0, metal_round=True),
                         dict(cpu_rint=-1, metal_round=0), dict(cpu_rint=0, metal_round=0, guessed=0)]:
            with self.subTest(counters=counters), self.assertRaises(ValueError):
                driver.complete_profiles(counters, 0, COVERAGE)
        with self.assertRaises(ValueError):
            driver.complete_profiles(dict(cpu_rint=0, metal_round=0), True, COVERAGE)

    def test_mapping_rejects_empty_unknown_and_duplicated_parent_descriptors(self):
        descriptor = dict(dtype='BF16', shape=[1, 64], data_offsets=[0, 128])
        for parents, child in [({}, {}), ({'one': {'unknown.weight': descriptor}}, {}),
            ({'one': {'model.language_model.embed_tokens.weight': descriptor},
              'two': {'model.language_model.embed_tokens.weight': descriptor}}, {})]:
            with self.subTest(parents=list(parents)), self.assertRaises(ValueError):
                driver.derive_mapping(parents, child)

    def test_fixed_source_authentication_cannot_be_fabricated_from_success_flags(self):
        for receipt in [{}, {'passed': True, 'completed': True},
            dict(format='toklet-qwen4-source-content-authentication-v2', passed=True, completed=True,
                 failure=None, model_executed=False, records=[],
                 plan_sha256='50e6994b9ead0df5730da424732f73770c498d0cb29af8ea42af811379c0ee07',
                 numerical_or_quality_or_fit_qualification=False,
                 individual_tensor_values_interpreted=False, whole_goal_complete=False)]:
            with self.subTest(receipt=receipt), self.assertRaises(ValueError):
                driver.validate_authentication(receipt, Path('/no-model-files-are-created'))

    def test_cli_preflight_bad_pin_and_existing_output_stop_before_array_import_or_io(self):
        with tempfile.TemporaryDirectory(prefix='toklet-correspondence-preflight-') as folder:
            root = Path(folder)
            receipt, output = root / 'receipt.json', root / 'report.json'
            receipt.write_bytes(b'{}\n')
            output.write_bytes(b'never overwrite\n')
            original_import = builtins.__import__
            def no_arrays(name, *args, **kwargs):
                if name == 'numpy' or name.endswith('qwen35_conversion_math'):
                    raise AssertionError('preflight must not import numerical modules')
                return original_import(name, *args, **kwargs)
            args = ['--artifact-root', str(root), '--authentication-receipt', str(receipt),
                    '--authentication-sha256', 'BAD', '--output', str(output)]
            for checksum, reason in [('BAD', 'Explicit authentication receipt SHA'),
                                     (hashlib.sha256(receipt.read_bytes()).hexdigest(), 'Fresh report')]:
                args[5] = checksum
                stderr = io.StringIO()
                with mock.patch('builtins.__import__', new=no_arrays), contextlib.redirect_stderr(stderr):
                    self.assertEqual(driver.main(args), 2)
                self.assertIn(reason, stderr.getvalue())
                self.assertEqual(output.read_bytes(), b'never overwrite\n')
                self.assertEqual(sorted(path.name for path in root.iterdir()), ['receipt.json', 'report.json'])


if __name__ == '__main__':
    unittest.main()
