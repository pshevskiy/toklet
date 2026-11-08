"""Small synthetic protocol controls; no checkpoint, GPU or model execution."""
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import check_gemma4_heretic_reference as check


class HereticReferenceCheckTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        small = patch.object(check, 'VOCAB', 8)
        small.start()
        self.addCleanup(small.stop)
        self.base = [0.] + [1.] * 7
        self.build()

    def write(self, path, value):
        path.write_text(json.dumps(value) + '\n')

    def build(self):
        self.native = self.root / 'native'
        (self.native / 'capture').mkdir(parents=True, exist_ok=True)
        ids, _ = check.reference.inputs('full512')
        rows = []
        for position in check.NATIVE_POSITIONS:
            raw = struct.pack('<8f', *self.base)
            name = f'position-{position}.f32le'
            (self.native / 'capture' / name).write_bytes(raw)
            rows.append(dict(position=position, file=name, bytes=len(raw), sha256=check.digest(raw), greedy_id=self.base.index(max(self.base))))
        source = dict(container_sha256=list(bytes.fromhex(check.source.FILE_SHA256)),
                      config_sha256=list(bytes.fromhex(check.source.CONFIG_SHA256)),
                      header_sha256=list(bytes.fromhex(check.source.HEADER_SHA256)),
                      container_bytes=check.source.FILE_BYTES, matched_text_components=1094)
        self.write(self.native / 'result.json', dict(format='toklet-heretic-context-fidelity-v1', passed=True, tokens=ids,
                   positions=check.NATIVE_POSITIONS, source=source, model_bytes=100, session_allocated_bytes=100,
                   caller_reserve_bytes=100, memory=dict(lifetime_peak_bytes=1000), rows=rows))
        self.write(self.native / 'checks.json', dict(passed=True, exit_code=0, changed_inputs=[]))
        self.folders = {}
        for phase in check.reference.PHASES:
            self.make_reference(phase)

    def make_reference(self, phase, *, change=None):
        folder = self.root / phase
        (folder / 'capture').mkdir(parents=True, exist_ok=True)
        ids, positions = check.reference.inputs(phase)
        body, arrays, contracts = bytearray(), [], {}
        for contract, code, size in [('math_fp64', 'd', 8), ('staged_fp32', 'f', 4)]:
            values, greedy = [], []
            for position in positions:
                row = self.base.copy()
                if change is not None:
                    change(contract, position, row)
                values.extend(row)
                greedy.append(row.index(max(row)))
            raw = struct.pack('<' + str(len(values)) + code, *values)
            name = contract + '.bin'
            (folder / 'capture' / name).write_bytes(raw)
            arrays.append(dict(contract=contract, name='logits', positions=positions,
                               shape=[1, len(positions), 8], dtype='<f' + str(size), offset=len(body),
                               bytes=len(raw), logits_file=name, sha256=check.digest(raw)))
            body.extend(raw)
            contracts[contract] = dict(greedy_ids=greedy)
        (folder / 'capture/states.bin').write_bytes(body)
        data = dict(format='toklet-gemma-heretic-math-v1', completed=True, phase=phase, tokens=ids,
                    selected_positions=positions, repository=check.source.REPOSITORY, revision=check.source.REVISION,
                    gate=dict(atol=1e-5, rtol=1e-4), source=dict(sha256=check.source.FILE_SHA256,
                    bytes=check.source.FILE_BYTES, closed=True, poisoned=False), source_endpoint_sha256=check.source.FILE_SHA256,
                    planned_capture_bytes=len(body), states_sha256=check.digest(body), arrays=arrays, contracts=contracts,
                    engine_executed=False, held_out_quality_scored=False, full_model_fidelity_qualified=False,
                    managed_adoption_qualified=False)
        self.write(folder / 'capture/result.json', data)
        self.write(folder / 'checks.json', dict(collection_passed=True, exit_code=0, changed_sources=[], error=None, lifetime_peak_physical_bytes=1000))
        self.folders[phase] = folder

    def test_complete_protocol_recomputes_every_selected_row(self):
        result = check.compare(self.native, self.folders)
        self.assertTrue(result['selected_context_fidelity_passed'])
        self.assertEqual((result['row_comparisons'], result['compared_values']), (22, 176))

    def test_staged_signed_zero_prefix_difference_is_not_bitwise_exact(self):
        def change(contract, position, row):
            if contract == 'staged_fp32' and position == 0:
                row[0] = -0.
        self.make_reference('pilot64', change=change)
        result = check.compare(self.native, self.folders)
        self.assertTrue(result['numerical_rows_passed'])
        self.assertFalse(result['causal_prefixes_passed'])
        self.assertFalse(result['selected_context_fidelity_passed'])

    def test_near_tie_is_reported_for_investigation(self):
        self.base = [1.] * 8
        self.build()
        def change(_contract, position, row):
            if position == 0:
                row[0] = 1. - 5e-8
        for phase in check.reference.PHASES:
            self.make_reference(phase, change=change)
        result = check.compare(self.native, self.folders)
        self.assertTrue(result['numerical_rows_passed'])
        self.assertTrue(result['causal_prefixes_passed'])
        self.assertTrue(result['unresolved_greedy_mismatches'])
        self.assertFalse(result['selected_context_fidelity_passed'])

    def test_incomplete_mutated_or_nonfinite_captures_reject(self):
        path = self.folders['compat32'] / 'capture/states.bin'
        path.write_bytes(path.read_bytes() + b'!')
        with self.assertRaisesRegex(ValueError, 'Complete reference'):
            check.compare(self.native, self.folders)
        self.make_reference('compat32')
        path = self.native / 'capture/position-0.f32le'
        path.write_bytes(b'!')
        with self.assertRaisesRegex(ValueError, 'length/digest'):
            check.native_rows(self.native)
        self.build()
        def change(_contract, position, row):
            if position == 0:
                row[0] = float('nan')
        self.make_reference('compat32', change=change)
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            check.reference_rows(self.folders['compat32'], 'compat32')

    def test_typed_state_phase_coverage_local_files_and_json_reject(self):
        path = self.native / 'result.json'
        data = json.loads(path.read_text())
        data['memory']['lifetime_peak_bytes'] = True
        self.write(path, data)
        with self.assertRaisesRegex(ValueError, 'memory'):
            check.native_rows(self.native)
        with self.assertRaisesRegex(ValueError, 'local'):
            check.captured(self.root, '../outside', 1, '0' * 64)
        with self.assertRaisesRegex(ValueError, 'All named'):
            check.compare(self.native, {})
        path = self.root / 'duplicate.json'
        path.write_text('{"x":1,"x":2}')
        with self.assertRaises(ValueError):
            check.document(path)


if __name__ == '__main__':
    unittest.main()
