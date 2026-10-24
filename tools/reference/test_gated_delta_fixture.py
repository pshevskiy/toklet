"""Portable stdlib fixture-reader tests, separate from numerical qualification."""
import copy
import hashlib
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest

import gated_delta_fixture as fixture


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def synthetic():
    # Format-only zeros. This is not a generated or mathematically qualified
    # recurrence reference. Hand-computed math tests live in the oracle suite.
    binary = bytearray()
    cases = []
    for name, tokens, hk, hv, dk, dv in fixture.CASE_SPECS:
        geometry = dict(tokens=tokens, key_heads=hk, value_heads=hv, key_dim=dk, value_dim=dv, epsilon=1e-6)
        arrays = {}
        for key, (dtype, shape) in fixture.array_layout(geometry).items():
            binary.extend(b'\0' * ((-len(binary)) % 8))
            payload = b'\0' * (math.prod(shape) * (4 if dtype == '<f4' else 8))
            arrays[key] = dict(offset=len(binary), bytes=len(payload), dtype=dtype, shape=shape, sha256=sha(payload))
            binary.extend(payload)
        cases.append(dict(name=name, geometry=geometry, arrays=arrays))
    value = dict(format=fixture.FORMAT, producer_sha256='1' * 64,
        upstream_revision=fixture.UPSTREAM_REVISION, upstream_source_sha256=fixture.UPSTREAM_SHA256,
        upstream_source_executed=False, upstream_numerical_parity_checked=False,
        masked_outputs_supported=False, native_graph_implemented=False, numerical_device_qualified=False,
        precision='FP32 stored inputs, FP64 normalization/gates/recurrence/output/state',
        normalization='sum-of-squares epsilon; Dk**-0.5 folded into Q exactly once',
        head_mapping='key_head = value_head // (value_heads // key_heads)',
        state_layout='[value_heads,value_dim,key_dim] contiguous key_dim',
        binary=dict(path='arrays.bin', bytes=len(binary), sha256=sha(binary), alignment=8), cases=cases)
    return value, bytes(binary)


class FixtureReader(unittest.TestCase):
    def write(self, folder, value, binary):
        raw = (json.dumps(value, allow_nan=False) + '\n').encode()
        (folder / 'manifest.json').write_bytes(raw)
        (folder / 'arrays.bin').write_bytes(binary)
        return sha(raw), sha(binary)

    def test_full_span_inventory_and_external_identities(self):
        value, binary = synthetic()
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            pins = self.write(folder, value, binary)
            self.assertEqual(fixture.verify(folder, *pins), value)
            for changed in [('0' * 64, pins[1]), (pins[0], '0' * 64)]:
                with self.assertRaisesRegex(ValueError, 'External fixture identity'):
                    fixture.verify(folder, *changed)
            (folder / 'extra').write_bytes(b'')
            with self.assertRaisesRegex(ValueError, 'inventory'):
                fixture.verify(folder, *pins)

    def test_mutated_shapes_types_offsets_and_array_names_reject(self):
        value, binary = synthetic()
        changed = []
        for key, mutation in [('offset', -1), ('offset', False), ('bytes', 1),
                              ('dtype', '>f4'), ('shape', [True, 3, 8]), ('sha256', '0' * 64)]:
            item = copy.deepcopy(value)
            item['cases'][0]['arrays']['initial_state'][key] = mutation
            changed.append(item)
        item = copy.deepcopy(value)
        del item['cases'][0]['arrays']['final_state']
        changed.append(item)
        item = copy.deepcopy(value)
        item['cases'][1]['geometry']['tokens'] = True
        changed.append(item)
        for item in changed:
            with tempfile.TemporaryDirectory() as temporary:
                folder = Path(temporary)
                pins = self.write(folder, item, binary)
                with self.assertRaises(ValueError):
                    fixture.verify(folder, *pins)

    def test_scope_and_numerical_contract_cannot_be_promoted(self):
        value, binary = synthetic()
        for key, replacement in [('upstream_numerical_parity_checked', True),
                                  ('masked_outputs_supported', True), ('native_graph_implemented', True),
                                  ('head_mapping', 'alternating'), ('state_layout', '[Hv,Dk,Dv]')]:
            changed = copy.deepcopy(value)
            changed[key] = replacement
            with tempfile.TemporaryDirectory() as temporary:
                folder = Path(temporary)
                pins = self.write(folder, changed, binary)
                with self.assertRaises(ValueError):
                    fixture.verify(folder, *pins)

    def test_nonfinite_payload_and_trailing_bytes_reject_even_with_updated_identities(self):
        value, original = synthetic()
        for kind in ['nan', 'trailing']:
            changed = copy.deepcopy(value)
            binary = bytearray(original)
            if kind == 'nan':
                array = changed['cases'][0]['arrays']['initial_state']
                offset = array['offset']
                binary[offset:offset + 4] = struct.pack('<f', float('nan'))
                array['sha256'] = sha(binary[offset:offset + array['bytes']])
            else:
                binary.extend(b'\0')
            changed['binary'].update(bytes=len(binary), sha256=sha(binary))
            with tempfile.TemporaryDirectory() as temporary:
                folder = Path(temporary)
                pins = self.write(folder, changed, binary)
                with self.assertRaisesRegex(ValueError, 'Nonfinite|trailing'):
                    fixture.verify(folder, *pins)

    def test_symlink_root_files_and_oversized_manifest_reject(self):
        value, binary = synthetic()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / 'fixture'
            folder.mkdir()
            pins = self.write(folder, value, binary)
            alias = root / 'alias'
            alias.symlink_to(folder, target_is_directory=True)
            with self.assertRaises(ValueError):
                fixture.verify(alias, *pins)
            outside = root / 'outside.bin'
            (folder / 'arrays.bin').replace(outside)
            (folder / 'arrays.bin').symlink_to(outside)
            with self.assertRaises(ValueError):
                fixture.verify(folder, *pins)
            (folder / 'arrays.bin').unlink()
            outside.replace(folder / 'arrays.bin')
            (folder / 'manifest.json').write_bytes(b' ' * (256 * 1024 + 1))
            with self.assertRaisesRegex(ValueError, 'bound'):
                fixture.verify(folder, '0' * 64, pins[1])

    def test_duplicate_and_nonfinite_json_numbers_reject(self):
        for raw in [b'{"format":"x","format":"y"}', b'{"bad":NaN}', b'{"bad":1e999}']:
            with tempfile.TemporaryDirectory() as temporary:
                folder = Path(temporary)
                (folder / 'manifest.json').write_bytes(raw)
                (folder / 'arrays.bin').write_bytes(b'')
                with self.assertRaisesRegex(ValueError, 'Duplicate|Nonfinite'):
                    fixture.verify(folder, sha(raw), sha(b''))


if __name__ == '__main__':
    unittest.main()
