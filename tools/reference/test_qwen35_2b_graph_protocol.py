"""Cold stdlib controls; no numerical runtime, native model, or large payload."""
import copy
import hashlib
import io
import json
import os
import subprocess
from pathlib import Path
import struct
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import bf16_source
import qwen35_2b_graph_protocol as p
import qwen35_2b_graph_check as check
import qwen35_2b_graph_reference as reference
import qwen35_2b_weights as weights


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encode(value):
    return json.dumps(value, separators=(',', ':'), allow_nan=False).encode()


PINS = dict(config_bytes=3000, config_sha256='a' * 64, header_bytes=100000,
            header_sha256='b' * 64, container_bytes=1059415912, container_sha256='c' * 64)


class ProtocolControls(unittest.TestCase):
    def test_no_numerical_runtime_on_import(self):
        command = [sys.executable, '-s', '-c',
            'import sys; import qwen35_2b_weights, qwen35_2b_graph_protocol, '
            'qwen35_2b_graph_reference, qwen35_2b_graph_check; '
            'blocked = [n for n in (\"numpy\", \"mlx\", \"torch\") if n in sys.modules]; '
            'exec(\"if blocked: raise RuntimeError(str(blocked))\")']
        result = subprocess.run(command, env={**os.environ, 'PYTHONPATH': str(Path(__file__).parent)},
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        self.assertEqual(result.returncode, 0, result.stdout.decode())

    def test_exact_typed_values(self):
        self.assertFalse(p.exact(True, 1))
        self.assertFalse(p.exact({'a': [1]}, {'a': [True]}))
        self.assertTrue(p.exact({'a': [1, 1.0]}, {'a': [1, 1.0]}))

    def test_full_geometry_states_and_shapes(self):
        shapes = p.shapes(1)
        self.assertEqual(len(shapes), 669)
        self.assertEqual(shapes[0, 'delta_state', None], [16, 128, 128])
        self.assertEqual(shapes[0, 'delta_history', None], [3, 6144])
        self.assertEqual(p.shapes(512)[3, 'attention_k_cache', 1], [512, 256])
        self.assertEqual(p.shapes(512)[23, 'attention_probabilities', 7], [512])
        for count in (True, 0, 513, 1025):
            with self.assertRaises(ValueError):
                p.shapes(count)

    def test_frozen_cases_phase_plan_and_unsupported_capacity(self):
        plan = p.plan()
        self.assertEqual((plan['arrays_per_profile'], plan['values_per_profile']), (8028, 79593504))
        self.assertEqual((plan['reference_rows_bytes'], plan['native_rows_bytes']), (955122048, 318374016))
        self.assertLessEqual(plan['reference_reserved_bytes'], 512 * 1024**2)
        self.assertEqual(plan['streamed_capture_reservation_bytes'],
                         2 * plan['journal_reservation_bytes'] + 32 * 1024**2)
        self.assertEqual([len(c['tokens']) for c in p.cases('context17')], [17, 17, 17])
        self.assertEqual(len(p.cases('fixed255')[0]['tokens']), 255)
        self.assertEqual(len(p.cases('context512')[0]['tokens']), 512)
        self.assertEqual(p.cases('fixed255')[0]['tokens'][:17], list(p.TOKENS17))
        for phase in ('context1025', 'request128_128', 'quality', None):
            with self.assertRaises(ValueError):
                p.cases(phase)

    def test_every_coordinate_unchanged_gate_and_nonfinite(self):
        native = struct.pack('<3f', 0., 0.5, 1.)
        good = struct.pack('<3d', 0., 0.5, 1.)
        self.assertEqual(p.compare(native, good, 'F64')['failed_values'], 0)
        bad = struct.pack('<3d', 0., 0.5, 1.01)
        self.assertEqual(p.compare(native, bad, 'F64')['failed_values'], 1)
        for raw, dtype in [(struct.pack('<f', float('nan')), 'F32'),
                           (struct.pack('<d', float('inf')), 'F64'), (b'', 'F32')]:
            with self.assertRaises(ValueError):
                p.finite(raw, dtype)
        self.assertTrue(abs(1e-5) <= p.GATE['atol'] + p.GATE['rtol'] * 0.)

    def test_json_duplicate_nonfinite_and_symlink_controls(self):
        for raw in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":1e999}'):
            with self.assertRaises(ValueError):
                weights.decode(raw)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'data'
            path.write_bytes(b'{}')
            link = Path(folder) / 'link'
            link.symlink_to(path)
            with self.assertRaises(ValueError):
                weights.captured(link, 2)
            with self.assertRaises(ValueError):
                weights.captured(path, 1)
            self.assertEqual(weights.captured(path, 2, sha(b'{}')), b'{}')

    def test_metadata_above_single_pread_limit_streams_bounded_chunks(self):
        raw = b'x' * (bf16_source.MAX_READ_BYTES + 17)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'journal'
            path.write_bytes(raw)
            self.assertEqual(weights.captured(path, len(raw), sha(raw)), raw)

    def test_actual_manifest_types_source_and_qualification(self):
        value = dict(format='toklet-qwen35-2b-affine-child-v1', passed=True,
            artifact_pins=PINS, source=weights.SOURCE, storage=weights.STORAGE,
            qualification=weights.QUALIFICATION, lineage_certificate_sha256='d' * 64,
            conversion_manifest_sha256='e' * 64)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'child-manifest.json'
            raw = encode(value)
            path.write_bytes(raw)
            self.assertEqual(weights.artifact_manifest(path, sha(raw)), value)
            for update in [dict(passed=1), dict(extra=True), dict(source={**weights.SOURCE, 'revision': 'bad'}),
                           dict(storage={**weights.STORAGE, 'body_bytes': 1}),
                           dict(qualification={**weights.QUALIFICATION, 'diagnostic_only': False}),
                           dict(lineage_certificate_sha256='0' * 64),
                           dict(artifact_pins={**PINS, 'header_bytes': True}),
                           dict(artifact_pins={**PINS, 'container_sha256': '0' * 64})]:
                raw = encode({**value, **update})
                path.write_bytes(raw)
                with self.subTest(update=update), self.assertRaises(ValueError):
                    weights.artifact_manifest(path, sha(raw))

    def test_independent_complete_inventory_matches_producer_metadata_only(self):
        import qwen35_2b_layout as producer
        planned = producer.selected_layout()
        table = weights.validate_table(planned['header'], planned['origin'], planned['file_bytes'])
        self.assertEqual(len(table), 694)
        self.assertEqual(sum(item['dtype'] == 'F32' for item in table.values()), 18)
        self.assertEqual(sum(item['dtype'] == 'U32' for item in table.values()), 187)
        for mutate in ('missing', 'dtype', 'shape', 'alias', 'tail', 'metadata'):
            altered = copy.deepcopy(planned['header'])
            names = [k for k in altered if k != '__metadata__']
            first = names[0]
            if mutate == 'missing':
                altered.pop(first)
            elif mutate == 'dtype':
                name = next(k for k in names if k.endswith('.A_log'))
                altered[name]['dtype'] = 'BF16'
            elif mutate == 'shape':
                altered[first]['shape'][0] = True
            elif mutate == 'alias':
                altered[names[1]]['data_offsets'] = altered[first]['data_offsets'][:]
            elif mutate == 'tail':
                altered[names[-1]]['data_offsets'][1] -= 1
            else:
                altered['__metadata__']['format'] = 'other'
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                weights.validate_table(altered, planned['origin'], planned['file_bytes'])

    def test_public_weights_preserves_frozen_recipe_metadata_through_bind(self):
        """Real full recipe, bounded synthetic values; source auth is separate."""
        import qwen35_2b_layout as producer
        planned = producer.selected_layout()
        header = planned['prefix'][8:]
        self.assertEqual(len(header), 89240)
        self.assertEqual(sha(header), 'ea3c847a30459cfd565f0abd8c8474b3d3818c2fd612a4bd0f220acec4965469')
        self.assertEqual(weights.decode(header)['__metadata__'], {'format': 'mlx'})
        config_path = Path(__file__).resolve().parents[2] / 'tests/fixtures/qwen35-2b-source-v1/source-config.json'
        config = producer.converted_config(config_path.read_bytes())
        pins = dict(PINS, config_bytes=len(config), config_sha256=sha(config),
            header_bytes=len(header), header_sha256=sha(header), container_bytes=planned['file_bytes'])
        # Stub only the already-authenticated owner, not weights()/bind(). This
        # reaches the public path that previously stripped metadata twice. It
        # opens no model and reconstructs no matrix or actual source values.
        owner = bf16_source.Source.__new__(bf16_source.Source)
        owner.size, owner.sha256 = planned['file_bytes'], pins['container_sha256']
        small = {planned['origin'] + item['data_offsets'][0]:
                 item['data_offsets'][1] - item['data_offsets'][0]
                 for name, item in planned['header'].items()
                 if name != '__metadata__' and name in weights.logical_specs()
                 and weights.logical_specs()[name][0] != 'matrix'}
        calls = []
        def read(offset, count, *, deadline):
            calls.append((offset, count))
            if offset == 0 and count == 8:
                return planned['prefix'][:8]
            if offset == 8 and count == len(header):
                return header
            if small.get(offset) == count:
                return bytes(count)
            raise ValueError('Unexpected synthetic source read')
        owner.read = read
        matrices, vectors = weights.weights(owner, config, pins, time.monotonic() + 30)
        self.assertEqual((len(matrices), len(vectors)), (187, 133))
        self.assertEqual(len(calls), 135)
        self.assertEqual({offset for offset, _ in calls[2:]}, set(small))

    def test_typed_reader_preserves_f32_alog_on_unaligned_body(self):
        g = dict(p.GEOMETRY, hidden=64, intermediate=64, vocab=32, layers=1,
            query_heads=2, kv_heads=1, head_dim=32, rotated_dims=8,
            key_heads=2, value_heads=4, key_dim=16, value_dim=16)
        table, body = {'__metadata__': {'format': 'mlx'}}, bytearray()
        actual = struct.unpack('<f', struct.pack('<f', 0.1001))[0]
        for name, (dtype, shape) in sorted(weights.physical_specs(g).items()):
            count = __import__('math').prod(shape)
            raw = (struct.pack('<f', actual) * count if dtype == 'F32'
                   else bytes(count * (2 if dtype == 'BF16' else 4)))
            start = len(body)
            body.extend(raw)
            table[name] = dict(dtype=dtype, shape=shape, data_offsets=[start, len(body)])
        raw = b'x' + body
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'weights'
            path.write_bytes(raw)
            with bf16_source.Source(path, len(raw), sha(raw), deadline=time.monotonic() + 30) as owner:
                matrices, vectors = weights.bind(owner, table, 1, time.monotonic() + 30, g)
                name = next(n for n in vectors if n.endswith('.A_log'))
                self.assertEqual(vectors[name], [actual] * 4)
                self.assertEqual(len(matrices), 9)
            path.write_bytes(raw + b'extra')
            with self.assertRaises(ValueError):
                bf16_source.Source(path, len(raw), sha(raw), deadline=time.monotonic() + 30)


class StreamControls(unittest.TestCase):
    """Small independent fixed-source fixture, with three complete arrays."""
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.native = self.root / 'native'
        self.native.mkdir()
        self.case = dict(name='fixed255', tokens=[1], position_offset=0)
        self.shapes = {(None, 'embedding', None): [2], (0, 'delta_state', None): [2],
                       (None, 'logits', None): [3]}
        self.plan = dict(arrays_per_profile=3, values_per_profile=7, native_rows_bytes=28,
                         reference_rows_bytes=84)
        self.patchers = [patch.object(p, 'cases', return_value=[self.case]),
            patch.object(p, 'shapes', return_value=self.shapes), patch.object(p, 'plan', return_value=self.plan)]
        for patcher in self.patchers:
            patcher.start()
        self.addCleanup(self.tmp.cleanup)
        for patcher in self.patchers:
            self.addCleanup(patcher.stop)

    def fixture(self, mutate=None):
        values = [[0.25, -0.5], [0.125, 0.75], [-1.0, 0.0, 1.0]]
        rows, blocks, offset = [], [], 0
        for ((layer, stage, head), shape), numbers in zip(self.shapes.items(), values, strict=True):
            block = struct.pack('<' + 'f' * len(numbers), *numbers)
            rows.append(dict(profile='native_fp32', case='fixed255', token=0, position=0,
                layer=layer, stage=stage, head=head, shape=shape, dtype='F32', count=len(numbers),
                offset=offset, bytes=len(block), sha256=sha(block)))
            blocks.append(block)
            offset += len(block)
        result = reference.summary('native', 'fixed255', 'f' * 64, PINS)
        result['capture_mode'] = 'stream'
        result['completed'] = True
        trace = b''.join(blocks)
        controls = [dict(case='fixed255', schedule=schedule, arrays=3, values=7,
            trace_sha256=sha(trace), state_sha256=sha(blocks[1]),
            logits_sha256=sha(blocks[-1]), history=[1]) for schedule in ('chunk1', 'chunk3', 'reset')]
        if mutate is not None:
            mutate(rows, blocks, result, controls)
        journal = b''.join(encode(row) + b'\n' for row in rows)
        control_raw = b''.join(encode(row) + b'\n' for row in controls)
        result.update(rows_sha256=sha(b''.join(blocks)), journal_bytes=len(journal),
            journal_sha256=sha(journal), controls=dict(bytes=len(control_raw), sha256=sha(control_raw)))
        (self.native / 'arrays.jsonl').write_bytes(journal)
        (self.native / 'controls.jsonl').write_bytes(control_raw)
        (self.native / 'result.json').write_bytes(encode(result))
        framed = b''.join(struct.pack('<I', len(encode(row))) + encode(row) + block
                          for row, block in zip(rows, blocks, strict=True))
        trailer = encode(result)
        framed += bytes(4) + struct.pack('<I', len(trailer)) + trailer
        return framed, values

    def run_fixture(self, mutate=None, suffix=b'', truncate=0):
        framed, values = self.fixture(mutate)
        if truncate:
            framed = framed[:-truncate]
        framed += suffix
        out = self.root / 'checked'
        checker = check.StreamComparison(io.BytesIO(framed), out, 'ideal_fp64', 'fixed255', 'f' * 64, PINS)
        try:
            for ((layer, stage, _), _shape), numbers in zip(self.shapes.items(), values, strict=True):
                checker.save(self.case, dict(token=0, layer=layer, stage=stage, values=numbers))
            return checker.finish(self.native)
        finally:
            checker.close()

    def test_complete_fixture_all_values_controls_and_hashes(self):
        result = self.run_fixture()
        self.assertEqual((result['arrays'], result['values'], result['failed_values']), (3, 7, 0))
        self.assertEqual(result['schedule_reset_controls'], 3)
        self.assertEqual(sha((self.root / 'checked/comparisons.jsonl').read_bytes()), result['journal_sha256'])

    def test_changed_numeric_value_reports_and_retains_failure(self):
        def change(rows, blocks, _result, _controls):
            blocks[0] = struct.pack('<2f', 0.25, 5.)
            rows[0]['sha256'] = sha(blocks[0])
            # Authentic control of the changed actual source bytes, independent
            # of the reference, so only the original numerical gate can fail.
            for control in _controls:
                control['trace_sha256'] = sha(b''.join(blocks))
        result = self.run_fixture(change)
        self.assertFalse(result['passed'])
        self.assertEqual(result['failed_values'], 1)
        self.assertEqual(result['retained_failure_bytes'], 24)
        self.assertEqual((self.root / 'checked/failure-0.native.f32le').read_bytes(), struct.pack('<2f', 0.25, 5.))

    def test_omitted_array(self):
        with self.assertRaises(ValueError):
            self.run_fixture(lambda rows, blocks, _r, _c: (rows.pop(1), blocks.pop(1)))

    def test_duplicate_array(self):
        with self.assertRaises(ValueError):
            self.run_fixture(lambda rows, _b, _r, _c: rows[1].update(rows[0]))

    def test_aliased_array(self):
        with self.assertRaises(ValueError):
            self.run_fixture(lambda rows, _b, _r, _c: rows[1].update(offset=0))

    def test_bad_dtype(self):
        with self.assertRaises(ValueError):
            self.run_fixture(lambda rows, _b, _r, _c: rows[0].update(dtype='BF16'))

    def test_nonfinite_array(self):
        def change(rows, blocks, _r, _c):
            blocks[0] = struct.pack('<2f', float('inf'), 0.)
            rows[0]['sha256'] = sha(blocks[0])
        with self.assertRaises(ValueError):
            self.run_fixture(change)

    def test_shape_key_phase_gate_source_protocol_mutations(self):
        changes = [lambda rows, _b, _r, _c: rows[0].update(shape=[1, 2]),
            lambda rows, _b, _r, _c: rows[0].update(stage='unknown'),
            lambda rows, _b, _r, _c: rows[0].update(case='missing'),
            lambda _a, _b, result, _c: result.update(phase='context17'),
            lambda _a, _b, result, _c: result.update(gate=dict(atol=1., rtol=1.)),
            lambda _a, _b, result, _c: result.update(source_manifest_sha256='0' * 64),
            lambda _a, _b, result, _c: result.update(protocol_sha256='0' * 64)]
        for index, change in enumerate(changes):
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.run_fixture(change)
            import shutil
            shutil.rmtree(self.root / 'checked')

    def test_omitted_duplicate_and_changed_controls(self):
        for change in (lambda _a, _b, _r, c: c.pop(),
                       lambda _a, _b, _r, c: c.append(c[0]),
                       lambda _a, _b, _r, c: c[0].update(state_sha256='0' * 64)):
            with self.assertRaises(ValueError):
                self.run_fixture(change)
            import shutil
            shutil.rmtree(self.root / 'checked')

    def seal_result(self, result):
        output = self.root / 'checked'
        raw = encode(result)
        (output / 'result.json').write_bytes(raw)
        return check.audit_stream_profile(output, sha(raw))

    def test_offline_complete_metadata_audit(self):
        result = self.run_fixture()
        self.assertEqual(self.seal_result(result), result)

    def test_offline_rehashed_omission_and_duplicate_reject(self):
        result = self.run_fixture()
        output = self.root / 'checked'
        path = output / 'comparisons.jsonl'
        original = path.read_bytes().splitlines(keepends=True)
        for lines in (original[:-1], original + [original[0]]):
            raw = b''.join(lines)
            path.write_bytes(raw)
            changed = {**result, 'journal_bytes': len(raw), 'journal_sha256': sha(raw)}
            with self.assertRaises(ValueError):
                self.seal_result(changed)

    def test_offline_failed_array_replay_and_tamper(self):
        def change(rows, blocks, _result, controls):
            blocks[0] = struct.pack('<2f', 0.25, 5.)
            rows[0]['sha256'] = sha(blocks[0])
            for control in controls:
                control['trace_sha256'] = sha(b''.join(blocks))
        result = self.run_fixture(change)
        self.assertEqual(self.seal_result(result)['failed_values'], 1)
        (self.root / 'checked/failure-0.native.f32le').write_bytes(bytes(8))
        with self.assertRaises(ValueError):
            self.seal_result(result)

    def test_truncation_extra_bytes_and_complete_profile_pair(self):
        with self.assertRaises(ValueError):
            self.run_fixture(truncate=1)
        import shutil
        shutil.rmtree(self.root / 'checked')
        with self.assertRaises(ValueError):
            self.run_fixture(suffix=b'extra')
        shutil.rmtree(self.root / 'checked')
        ideal = self.run_fixture()
        staged = dict(ideal, profile='staged_fp32')
        self.assertTrue(check.pair_stream_profiles(ideal, staged)['passed'])
        for update in (dict(completed=False), dict(native_trace_sha256='0' * 64),
                       dict(values=6), dict(gate=dict(atol=1., rtol=1.))):
            with self.assertRaises(ValueError):
                check.pair_stream_profiles(ideal, {**staged, **update})


if __name__ == '__main__':
    unittest.main()
