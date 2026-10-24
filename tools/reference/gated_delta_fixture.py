"""Stdlib-only gated-delta math-fixture reader; no target/NumPy imports."""
import hashlib
import json
import math
from pathlib import Path
import re
import struct

FORMAT = 'toklet-gated-delta-math-reference-v1'
UPSTREAM_REVISION = '0d4d3d04f8ab7ce4a51e7209e8615755a48ba94d'
UPSTREAM_SHA256 = 'f69e837ea68641dbb06fdb5dcc024a9a7e1d9353f08ae58c05275253a0c39428'
MAX_ARRAY_BYTES = 32 * 1024 * 1024
CASE_SPECS = (('empty', 0, 1, 1, 8, 3), ('single', 1, 1, 1, 1, 1),
              ('grouped', 7, 2, 4, 8, 3), ('kernel-width32', 5, 1, 2, 32, 5),
              ('state-width128', 3, 2, 4, 128, 128))


def array_layout(geometry):
    t, hk, hv, dk, dv = [geometry[key] for key in ('tokens', 'key_heads', 'value_heads', 'key_dim', 'value_dim')]
    return dict(raw_q=('<f4', [t, hk, dk]), raw_k=('<f4', [t, hk, dk]),
        v=('<f4', [t, hv, dv]), a=('<f4', [t, hv]), b=('<f4', [t, hv]),
        a_log=('<f4', [hv]), dt_bias=('<f4', [hv]), initial_state=('<f4', [hv, dv, dk]),
        normalized_q=('<f8', [t, hk, dk]), normalized_k=('<f8', [t, hk, dk]),
        decay=('<f8', [t, hv]), beta=('<f8', [t, hv]), output=('<f8', [t, hv, dv]),
        states=('<f8', [t, hv, dv, dk]), final_state=('<f8', [hv, dv, dk]))


def _sha(value):
    if not isinstance(value, str) or re.fullmatch('[a-f0-9]{64}', value) is None:
        raise ValueError('Expected an external lowercase SHA256 identity')
    return value


def _bounded(path, maximum):
    if path.is_symlink() or not path.is_file():
        raise ValueError('Expected a regular fixture file')
    with path.open('rb') as stream:
        raw = stream.read(maximum + 1)
    if len(raw) > maximum:
        raise ValueError('Fixture file exceeds its bound')
    return raw


def _unique(items):
    value = {}
    for key, item in items:
        if key in value:
            raise ValueError('Duplicate fixture JSON field')
        value[key] = item
    return value


def _bad_number(value):
    raise ValueError('Nonfinite fixture JSON number: ' + value)


def _finite_number(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError('Nonfinite fixture JSON number: ' + value)
    return number


def verify(directory, expected_manifest_sha256, expected_binary_sha256):
    """External identities authenticate bytes before schema, spans or values.

    These checks establish a bounded mathematical fixture, never upstream parity
    or numerical/device qualification. The caller must supply both identities
    from its independently accepted record, not from this fixture's own fields.
    """
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError('Expected a regular fixture directory')
    if {path.name for path in directory.iterdir()} != {'manifest.json', 'arrays.bin'}:
        raise ValueError('Fixture inventory differs')
    raw = _bounded(directory / 'manifest.json', 256 * 1024)
    binary = _bounded(directory / 'arrays.bin', MAX_ARRAY_BYTES)
    if (hashlib.sha256(raw).hexdigest() != _sha(expected_manifest_sha256)
            or hashlib.sha256(binary).hexdigest() != _sha(expected_binary_sha256)):
        raise ValueError('External fixture identity differs')
    manifest = json.loads(raw, object_pairs_hook=_unique, parse_constant=_bad_number, parse_float=_finite_number)
    if not isinstance(manifest, dict):
        raise ValueError('Expected a fixture JSON object')
    if (manifest.get('format') != FORMAT or manifest.get('upstream_revision') != UPSTREAM_REVISION
            or manifest.get('upstream_source_sha256') != UPSTREAM_SHA256
            or manifest.get('upstream_source_executed') is not False
            or manifest.get('upstream_numerical_parity_checked') is not False
            or manifest.get('masked_outputs_supported') is not False
            or manifest.get('native_graph_implemented') is not False
            or manifest.get('numerical_device_qualified') is not False):
        raise ValueError('Mathematical-reference scope differs')
    _sha(manifest.get('producer_sha256'))
    text_contract = dict(
        precision='FP32 stored inputs, FP64 normalization/gates/recurrence/output/state',
        normalization='sum-of-squares epsilon; Dk**-0.5 folded into Q exactly once',
        head_mapping='key_head = value_head // (value_heads // key_heads)',
        state_layout='[value_heads,value_dim,key_dim] contiguous key_dim')
    if any(manifest.get(key) != value for key, value in text_contract.items()):
        raise ValueError('Reference numerical contract differs')
    descriptor = manifest.get('binary')
    if (not isinstance(descriptor, dict) or descriptor.get('path') != 'arrays.bin'
            or type(descriptor.get('bytes')) is not int or descriptor['bytes'] != len(binary)
            or type(descriptor.get('alignment')) is not int or descriptor['alignment'] != 8
            or descriptor.get('sha256') != expected_binary_sha256):
        raise ValueError('Binary fixture descriptor differs')
    cases = manifest.get('cases')
    if not isinstance(cases, list) or len(cases) != len(CASE_SPECS):
        raise ValueError('Fixture case inventory differs')
    cursor = 0
    for row, spec in zip(cases, CASE_SPECS):
        name, tokens, hk, hv, dk, dv = spec
        expected = dict(tokens=tokens, key_heads=hk, value_heads=hv, key_dim=dk, value_dim=dv, epsilon=1e-6)
        if not isinstance(row, dict) or set(row) != {'name', 'geometry', 'arrays'} or row['name'] != name:
            raise ValueError('Fixture case descriptor differs')
        geometry = row['geometry']
        if not isinstance(geometry, dict) or geometry != expected:
            raise ValueError('Fixture geometry differs')
        if any(type(geometry[key]) is not int for key in expected if key != 'epsilon'):
            raise ValueError('Expected integer fixture dimensions')
        layout = array_layout(expected)
        if not isinstance(row['arrays'], dict) or set(row['arrays']) != set(layout):
            raise ValueError('Array inventory differs')
        for key, (dtype, shape) in layout.items():
            array = row['arrays'][key]
            if (not isinstance(array, dict) or set(array) != {'offset', 'bytes', 'dtype', 'shape', 'sha256'}
                    or array['dtype'] != dtype or not isinstance(array['shape'], list)
                    or array['shape'] != shape or any(type(n) is not int for n in array['shape'])):
                raise ValueError('Array type or shape differs')
            size = math.prod(shape) * (4 if dtype == '<f4' else 8)
            offset = cursor + (-cursor) % 8
            if (type(array['offset']) is not int or type(array['bytes']) is not int
                    or array['offset'] != offset or array['bytes'] != size
                    or offset + size > len(binary) or any(binary[cursor:offset])):
                raise ValueError('Array span, alignment or padding differs')
            payload = binary[offset:offset + size]
            if hashlib.sha256(payload).hexdigest() != _sha(array['sha256']):
                raise ValueError('Array payload identity differs')
            unpack = '<f' if dtype == '<f4' else '<d'
            if any(not math.isfinite(value[0]) for value in struct.iter_unpack(unpack, payload)):
                raise ValueError('Nonfinite fixture array')
            cursor = offset + size
    if cursor != len(binary):
        raise ValueError('Unaccounted trailing fixture bytes')
    return manifest


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--expected-manifest-sha256', required=True)
    parser.add_argument('--expected-binary-sha256', required=True)
    options = parser.parse_args()
    verify(options.directory, options.expected_manifest_sha256, options.expected_binary_sha256)
    print('Bound mathematical fixture verified; upstream numerical parity remains separate.')
