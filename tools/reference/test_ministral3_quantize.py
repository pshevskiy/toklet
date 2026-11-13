"""Portable Ministral tile/probe controls using only stdlib and small fakes."""
import copy
import builtins
from contextlib import redirect_stdout
import hashlib
import io
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import granite_quantize as legacy
import minicpm5_quantize as minicpm
import ministral3_quantize as quant
import ministral3_quant_probe as probe


ROOT = Path(__file__).resolve().parents[2]
CONTROL_FILE = ROOT / 'tests/fixtures/ministral3-conversion-v1/controls.json'
CONTROL_SHA = 'b3f2f2d401de49a38a5f3cfb3bea1995dff4b1f140a461d253a24b2ca7a6643d'


def f32(value):
    try:
        return struct.unpack('<f', struct.pack('<f', value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


class Mask:
    def __init__(self, values):
        self.values = values

    def all(self):
        return all(self.values)


class Array:
    """Small one-row fake for the exact vector operations under test."""
    def __init__(self, values, dtype='u2', shape=None):
        self.values = list(values)
        self.dtype = dtype
        self.shape = shape or (len(self.values),)

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        shape = list(shape)
        if -1 in shape:
            known = math.prod(n for n in shape if n != -1)
            shape[shape.index(-1)] = len(self.values) // known
        if math.prod(shape) != len(self.values):
            raise ValueError('fake reshape extent')
        return Array(self.values, self.dtype, tuple(shape))

    def __getitem__(self, key):
        if key != (slice(None), slice(None), None):
            raise ValueError('unsupported fake index')
        return self

    def __and__(self, mask):
        return Array([v & mask for v in self.values], self.dtype, self.shape)

    def __eq__(self, value):
        return Mask([v == value for v in self.values])

    def __lshift__(self, shift):
        return Array([v << shift for v in self.values], self.dtype, self.shape)

    def __rshift__(self, shifts):
        return Array([v >> shift for v in self.values for shift in shifts.values], self.dtype)

    def __mul__(self, scalar):
        return Array([v * scalar for v in self.values], self.dtype, self.shape)

    def astype(self, dtype, copy=True):
        del copy
        dtype = {'uint32': 'u4', 'float32': 'f4'}.get(dtype, dtype.lstrip('<'))
        return Array(self.values, dtype, self.shape)

    def view(self, dtype):
        if dtype != 'float32':
            raise ValueError('unsupported fake view')
        return Array([struct.unpack('<f', struct.pack('<I', v))[0] for v in self.values], 'f4', self.shape)

    def tobytes(self):
        code = {'u2': 'H', 'u4': 'I', 'f4': 'f'}[self.dtype]
        return b''.join(struct.pack('<' + code, value) for value in self.values)


class NumpyFake:
    uint32 = 'uint32'
    float32 = 'float32'

    def __init__(self):
        self.operations = []

    def frombuffer(self, raw, dtype):
        code = {'<u2': 'H', '<u4': 'I'}[dtype]
        return Array([v for (v,) in struct.iter_unpack('<' + code, raw)], dtype[1:])

    def asarray(self, tensor):
        dtype = {'uint32': '<u4', 'uint16': '<u2'}[tensor.dtype]
        return self.frombuffer(tensor.raw, dtype).reshape(*tensor.shape)

    @staticmethod
    def any(mask):
        return any(mask.values)

    @staticmethod
    def arange(count, dtype):
        return Array(range(count), dtype)

    @staticmethod
    def repeat(array, repeats, axis):
        if axis != 1:
            raise ValueError('unsupported fake repeat')
        return Array([v for v in array.values for _ in range(repeats)], 'f4')

    @staticmethod
    def empty(shape, dtype):
        return Array([0.] * math.prod(shape), dtype, shape)

    def multiply(self, first, second, out, dtype):
        del dtype
        self.operations.append('multiply')
        out.values = [f32(a * b) for a, b in zip(first.values, second.values)]

    def add(self, first, second, out, dtype):
        del dtype
        self.operations.append('add')
        out.values = [f32(a + b) for a, b in zip(first.values, second.values)]

    @staticmethod
    def isfinite(array):
        return Mask([math.isfinite(value) for value in array.values])


class Tensor:
    def __init__(self, raw, dtype, shape):
        self.raw, self.dtype, self.shape = raw, dtype, shape

    def view(self, dtype, stream):
        del stream
        return Tensor(self.raw, dtype, self.shape)


class MetalFake:
    cpu, gpu = 'cpu', 'gpu'
    bfloat16, uint32, uint16 = 'bfloat16', 'uint32', 'uint16'

    def __init__(self, buffers, rows=1, columns=9216):
        self.buffers = buffers
        self.rows, self.columns = rows, columns
        self.calls = []
        self.packed_dtype = self.uint32

    def array(self, values):
        self.calls.append('array')
        return Tensor(values.tobytes(), self.uint16, values.shape)

    def quantize(self, source, group_size, bits, mode, stream):
        if (source.dtype, source.shape, group_size, mode) != (
                self.bfloat16, (self.rows, self.columns), 64, 'affine'):
            raise ValueError('fake quantization contract')
        self.calls.append(('quantize', bits, stream))
        return (Tensor(self.buffers['weight'], self.packed_dtype, (self.rows, self.columns * bits // 32)),
                Tensor(self.buffers['scales'], self.bfloat16, (self.rows, self.columns // 64)),
                Tensor(self.buffers['biases'], self.bfloat16, (self.rows, self.columns // 64)))

    def eval(self, *values):
        self.calls.append(('eval', len(values)))

    def synchronize(self, stream):
        self.calls.append(('synchronize', stream))


def repeated_buffers(case, device, columns=9216):
    return {k: bytes.fromhex(v) * (columns // 64) for k, v in case[device].items()}


class GeometryAndLegacy(unittest.TestCase):
    def test_geometry_and_tail_bounds_do_not_widen_existing_helpers(self):
        self.assertEqual((quant.ROWS, quant.MAX_COLUMNS, quant.GROUP), (224, 9216, 64))
        for rows in (1, 32, 64, 128, 160, 224):
            raw = bytes(rows * 9216 * 2)
            for bits in (4, 8):
                self.assertEqual(quant.geometry(raw, rows, 9216, bits), rows * 9216)
                with self.assertRaises(ValueError):
                    legacy.geometry(raw, rows, 9216, bits)
            with self.assertRaises(ValueError):
                minicpm.geometry(raw, rows, 9216)
        self.assertEqual((legacy.ROWS, legacy.MAX_COLUMNS), (256, 8192))
        self.assertEqual((minicpm.ROWS, minicpm.MAX_COLUMNS), (256, 6144))
        self.assertLessEqual(quant.ROWS * quant.MAX_COLUMNS * 2, 4 * 1024**2)

    def test_invalid_shapes_and_extents_reject_before_numerical_access(self):
        for rows, columns, bits in [(True, 64, 4), (1, True, 4), (0, 64, 4),
                                    (225, 64, 4), (1, 65, 4), (1, 9280, 4),
                                    (1, 0, 4), (1, 64, True), (1, 64, 2),
                                    (1.0, 64, 4), (1, 9216.0, 8)]:
            with self.subTest(shape=(rows, columns, bits)), self.assertRaises(ValueError):
                quant.quantize(bytes(128), rows, columns, bits, 'cpu', None, None)
        for raw in (b'', bytes(127), bytes(129), bytearray(128), '00'):
            with self.assertRaises(ValueError):
                quant.geometry(raw, 1, 64, 4)
        for device in (None, 'implicit', True):
            with self.assertRaises(ValueError):
                quant.quantize(bytes(128), 1, 64, 4, device, None, None)

    def test_qualified_columns_delegate_only_after_new_preflight(self):
        raw = bytes(128)
        with patch.object(legacy, 'quantize', return_value='delegated') as call:
            self.assertEqual(quant.quantize(raw, 1, 64, 4, 'gpu', None, None), 'delegated')
            call.assert_called_once_with(raw, 1, 64, 4, 'gpu', None, None)
            with self.assertRaises(ValueError):
                quant.quantize(bytes(225 * 64 * 2), 225, 64, 4, 'gpu', None, None)
            self.assertEqual(call.call_count, 1)
        buffers = dict(weight=bytes(32), scales=bytes(2), biases=bytes(2))
        with patch.object(legacy, 'reconstruct_f32', return_value=b'reused') as call:
            self.assertEqual(quant.reconstruct_f32(buffers, 1, 64, 4, None), b'reused')
            call.assert_called_once_with(buffers, 1, 64, 4, None)

    def test_reconstruction_buffers_and_nonfinite_metadata_reject_early(self):
        for columns in (64, 9216):
            complete = dict(weight=bytes(columns // 2), scales=bytes(columns // 32), biases=bytes(columns // 32))
            for key in complete:
                broken = dict(complete)
                broken[key] = broken[key][:-1]
                with self.assertRaises(ValueError):
                    quant.reconstruct_f32(broken, 1, columns, 4, None)
            for broken in ({}, dict(complete, extra=b''), dict(complete, scales=None)):
                with self.assertRaises(ValueError):
                    quant.reconstruct_f32(broken, 1, columns, 4, None)
            for word in (0x7F80, 0xFF80, 0x7FC1, 0xFFFF):
                for key in ('scales', 'biases'):
                    broken = dict(complete)
                    broken[key] = struct.pack('<H', word) + broken[key][2:]
                    with self.assertRaises(ValueError):
                        quant.reconstruct_f32(broken, 1, columns, 4, None)


class WideBranch(unittest.TestCase):
    def test_wide_fake_cpu_gpu_quantization_preserves_completed_literal_bytes(self):
        for bits in (4, 8):
            case = next(c for c in quant.controls() if c['bits'] == bits and c['name'] == 'round_ties')
            for device in ('cpu', 'gpu'):
                buffers = repeated_buffers(case, device)
                np, mx = NumpyFake(), MetalFake(buffers)
                raw = bytes.fromhex(case['input_bf16']) * 144
                with patch.object(legacy, 'quantize', side_effect=AssertionError('wide delegation')):
                    self.assertEqual(quant.quantize(raw, 1, 9216, bits, device, np, mx), buffers)
                self.assertEqual(mx.calls[-2:], [('eval', 3), ('synchronize', device)])

    def test_wide_nonfinite_source_rejects_before_metal_access(self):
        for word in (0x7F80, 0xFF80, 0x7FC0):
            np, mx = NumpyFake(), MetalFake({})
            with self.assertRaises(ValueError):
                quant.quantize(struct.pack('<H', word) + bytes(9216 * 2 - 2),
                               1, 9216, 4, 'gpu', np, mx)
            self.assertEqual(mx.calls, [])

    def test_wide_upstream_dtype_and_metadata_fail_closed(self):
        case = quant.controls()[0]
        buffers = repeated_buffers(case, 'gpu')
        mx = MetalFake(buffers)
        mx.packed_dtype = mx.uint16
        with self.assertRaises(ValueError):
            quant.quantize(bytes(9216 * 2), 1, 9216, case['bits'], 'gpu', NumpyFake(), mx)
        broken = dict(buffers, scales=struct.pack('<H', 0x7F80) + buffers['scales'][2:])
        with self.assertRaises(ValueError):
            quant.quantize(bytes(9216 * 2), 1, 9216, case['bits'], 'gpu', NumpyFake(), MetalFake(broken))

    def test_independent_wide_fake_decoder_matches_all_prospective_literals(self):
        for case in quant.controls():
            for device in ('cpu', 'gpu'):
                np = NumpyFake()
                with patch.object(legacy, 'reconstruct_f32', side_effect=AssertionError('wide delegation')):
                    actual = quant.reconstruct_f32(repeated_buffers(case, device), 1, 9216, case['bits'], np)
                self.assertEqual(actual, quant.literal_reconstruction(case, device) * 144)
                self.assertEqual(np.operations, ['multiply', 'add'])

    def test_finite_bf16_metadata_can_still_overflow_f32_reconstruction(self):
        buffers = dict(weight=b'\xff' * 9216, scales=struct.pack('<H', 0x7F7F) * 144, biases=bytes(288))
        with self.assertRaises(ValueError):
            quant.reconstruct_f32(buffers, 1, 9216, 8, NumpyFake())


class LiteralsAndProbe(unittest.TestCase):
    def test_frozen_controls_keep_all_device_words_and_six_exact_shapes(self):
        raw = CONTROL_FILE.read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), CONTROL_SHA)
        controls = probe.load_controls(CONTROL_FILE, CONTROL_SHA)
        self.assertEqual(controls['cases'], legacy.controls())
        self.assertEqual(quant.controls(), legacy.controls())
        self.assertEqual(len(controls['cases']), 12)
        self.assertEqual(controls['shapes'], [list(s) for s in probe.SHAPES])
        self.assertEqual(len(controls['cases']) * len(probe.SHAPES) * 2, 144)
        for case in controls['cases']:
            for device in ('cpu', 'gpu'):
                self.assertEqual(quant.literal_reconstruction(case, device), legacy.literal_reconstruction(case, device))

    def test_control_hash_schema_type_and_expected_byte_mutations_reject(self):
        original = json.loads(CONTROL_FILE.read_bytes())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'controls.json'
            for mutate in (
                lambda c: c.update(extra=1),
                lambda c: c['shapes'][0].__setitem__(0, True),
                lambda c: c['shapes'].pop(),
                lambda c: c['cases'][0].update(bits=True),
                lambda c: c['cases'][0]['gpu'].update(weight='ff' * 64),
            ):
                changed = copy.deepcopy(original)
                mutate(changed)
                raw = json.dumps(changed).encode()
                path.write_bytes(raw)
                with self.assertRaises(ValueError):
                    probe.load_controls(path, hashlib.sha256(raw).hexdigest())
            path.write_bytes(CONTROL_FILE.read_bytes())
            with self.assertRaises(ValueError):
                probe.load_controls(path, '0' * 64)
            for raw in (b'{"format":1,"format":2}', b'{"format":NaN}'):
                path.write_bytes(raw)
                with self.assertRaises(ValueError):
                    probe.load_controls(path, hashlib.sha256(raw).hexdigest())

    def test_execute_case_returns_scalar_receipts_and_retains_failure(self):
        case = quant.controls()[0]
        actual = {key: bytes.fromhex(value) for key, value in case['gpu'].items()}
        decoded = quant.literal_reconstruction(case, 'gpu')
        with patch.object(quant, 'quantize', return_value=actual), \
                patch.object(quant, 'reconstruct_f32', return_value=decoded), \
                patch.object(probe, 'upstream_f32', return_value=decoded):
            record = probe.execute_case(case, 1, 64, 'gpu', None, None)
            self.assertTrue(record['passed'])
            self.assertEqual(record['independent_F32_sha256'], hashlib.sha256(decoded).hexdigest())
            self.assertEqual(record['input_bytes'], 128)
            json.dumps(record, allow_nan=False)
            self.assertNotIn('input_bf16', record)
        broken = dict(actual, weight=b'\xff' + actual['weight'][1:])
        with patch.object(quant, 'quantize', return_value=broken), \
                patch.object(quant, 'reconstruct_f32', return_value=decoded), \
                patch.object(probe, 'upstream_f32', return_value=decoded):
            record = probe.execute_case(case, 1, 64, 'gpu', None, None)
            self.assertFalse(record['passed'])
            self.assertEqual(record['buffers']['weight']['mismatching_bytes'], 1)

    def test_source_inventory_contains_all_new_and_transitive_helpers(self):
        names = {p.name for p in probe.sources()}
        self.assertEqual(len(names), len(probe.sources()))
        self.assertTrue(all(p.is_file() for p in probe.sources()))
        self.assertTrue({'ministral3_quantize.py', 'ministral3_quant_probe.py', 'test_ministral3_quantize.py',
                         'granite_quantize.py', 'granite_quant_probe.py', 'granite_quant_runtime.py',
                         'minicpm5_quantize.py', 'minicpm5_quant_runtime.py',
                         'reference_runtime.py', 'text_reference_runtime.py'} <= names)

    def test_cold_imports_do_not_import_numerical_runtime(self):
        script = ('import sys; import ministral3_quant_probe; '
                  'assert not any(n == p or n.startswith(p+".") '
                  'for n in sys.modules for p in ("numpy","mlx"))')
        result = subprocess.run([sys.executable, '-s', '-B', '-c', script],
                                cwd=Path(__file__).parent, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr.decode())

    def test_fresh_output_rejects_alias_existing_and_missing_parent_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            self.assertEqual(probe.check_output(root / 'new'), root / 'new')
            for path in (Path('relative'), root, root / 'missing' / 'new', root / '..' / 'new'):
                with self.assertRaises(ValueError):
                    probe.check_output(path)
            alias = root / 'alias'
            alias.symlink_to(root, target_is_directory=True)
            with self.assertRaises(ValueError):
                probe.check_output(alias / 'new')
            dangling = root / 'dangling'
            dangling.symlink_to(root / 'absent')
            with self.assertRaises(ValueError):
                probe.check_output(dangling)

    def test_cli_existing_output_rejects_before_controls_or_runtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            command = [sys.executable, '-s', '-B', str(Path(probe.__file__))]
            for option in ('environment-root', 'runtime-manifest', 'runtime-sha256', 'controls-file', 'controls-sha256'):
                command.extend(['--' + option, 'absent'])
            command.extend(['--fresh-output', temporary])
            result = subprocess.run(command, capture_output=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b'Never replace', result.stderr)
            self.assertNotIn(b'absent', result.stderr)
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_main_fake_run_checks_144_unique_scalar_records_and_endpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            calls = []
            original_import = builtins.__import__
            core = SimpleNamespace(set_cache_limit=lambda size: calls.append(('cache', size)),
                                   metal=SimpleNamespace(device_info=lambda: {'fake': True}))

            def admitted_import(name, *args, **kwargs):
                if name in ('numpy', 'mlx.core'):
                    self.assertIn('admit', calls)
                    calls.append(('fake_import', name))
                    return SimpleNamespace() if name == 'numpy' else SimpleNamespace(core=core)
                return original_import(name, *args, **kwargs)

            def admit(*args):
                del args
                calls.append('admit')
                return {'fake': True}, lambda: calls.append('endpoint') or {'fake': True}

            def execute(case, rows, columns, device, np, mx):
                del np, mx
                return dict(bits=case['bits'], name=case['name'], rows=rows,
                            columns=columns, device=device, passed=True)

            output = root / 'controls'
            args = ['--environment-root', str(ROOT), '--runtime-manifest', 'fake',
                    '--runtime-sha256', '0' * 64, '--controls-file', str(CONTROL_FILE),
                    '--controls-sha256', CONTROL_SHA, '--fresh-output', str(output)]
            with patch.object(probe.runtime, 'bind_environment', side_effect=lambda _: calls.append('bind')), \
                    patch.object(probe.runtime, 'admit', side_effect=admit), \
                    patch.object(probe, 'execute_case', side_effect=execute), \
                    patch('builtins.__import__', side_effect=admitted_import), redirect_stdout(io.StringIO()):
                self.assertEqual(probe.main(args), 0)
            result = json.loads((output / 'observation.json').read_bytes())
            self.assertTrue(result['passed'])
            self.assertEqual(result['executions'], 144)
            self.assertEqual(len((output / 'records.jsonl').read_text().splitlines()), 144)
            self.assertEqual(calls[:2], ['bind', 'admit'])
            self.assertEqual(calls[-1], 'endpoint')
            self.assertEqual(calls.count('endpoint'), 1)
            self.assertNotIn('numpy', sys.modules)
            self.assertNotIn('mlx.core', sys.modules)

    def test_failed_runtime_admission_never_imports_numerical_packages(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary).resolve() / 'failed'
            args = ['--environment-root', str(ROOT), '--runtime-manifest', 'fake',
                    '--runtime-sha256', '0' * 64, '--controls-file', str(CONTROL_FILE),
                    '--controls-sha256', CONTROL_SHA, '--fresh-output', str(output)]
            with patch.object(probe.runtime, 'bind_environment'), \
                    patch.object(probe.runtime, 'admit', side_effect=ValueError('rejected runtime')):
                with self.assertRaisesRegex(ValueError, 'rejected runtime'):
                    probe.main(args)
            self.assertEqual(list(output.iterdir()), [])
            self.assertNotIn('numpy', sys.modules)
            self.assertNotIn('mlx.core', sys.modules)


if __name__ == '__main__':
    unittest.main()
