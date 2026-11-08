"""Recompute selected native/independent context-row numerical gates offline.

Consistency and complete row comparison complement the owned execution receipts;
they do not independently prove execution, lineage, held-out quality or adoption.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import check_gemma_hybrid32 as prior
import gemma4_heretic_reference as reference
import gemma4_heretic_weights as source

VOCAB = 262144
NATIVE_POSITIONS = [0, 15, 30, 31, 63, 255, 511]


def need(value, message):
    if not value:
        raise ValueError(message)


def read(path, maximum):
    need(not path.is_symlink() and path.is_file(), 'Regular captured file required')
    with path.open('rb') as file:
        raw = file.read(maximum + 1)
    need(len(raw) <= maximum, 'Capture exceeds bound')
    return raw


def document(path):
    return prior.bounded.document(read(path, 4 * 1024**2))


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def captured(folder, name, size, sha):
    need(type(name) is str and Path(name).name == name and name not in ['', '.', '..'], 'Captured filename must be local')
    raw = read(folder / name, size)
    need(len(raw) == size and digest(raw) == sha, 'Capture body length/digest differs')
    return raw


def native_rows(folder):
    data, receipt = document(folder / 'result.json'), document(folder / 'checks.json')
    ids, _ = reference.inputs('full512')
    need(data['format'] == 'toklet-heretic-context-fidelity-v1' and data['passed'] is True
         and receipt['passed'] is True and receipt['exit_code'] == 0 and receipt['changed_inputs'] == [], 'Completed native capture required')
    need(data['tokens'] == ids and all(type(v) is int for v in data['tokens'])
         and data['positions'] == NATIVE_POSITIONS and all(type(v) is int for v in data['positions']), 'Native frozen tokens/positions differ')
    need(data['source']['container_sha256'] == list(bytes.fromhex(source.FILE_SHA256))
         and data['source']['config_sha256'] == list(bytes.fromhex(source.CONFIG_SHA256))
         and data['source']['header_sha256'] == list(bytes.fromhex(source.HEADER_SHA256))
         and data['source']['container_bytes'] == source.FILE_BYTES
         and data['source']['matched_text_components'] == 1094, 'Native selected source identity differs')
    need(all(type(data[k]) is int and 0 < data[k] <= 6 * 1024**3 for k in ['model_bytes', 'session_allocated_bytes', 'caller_reserve_bytes'])
         and data['model_bytes'] + data['session_allocated_bytes'] + data['caller_reserve_bytes'] <= 6 * 1024**3
         and type(data['memory']['lifetime_peak_bytes']) is int
         and 0 < data['memory']['lifetime_peak_bytes'] <= 6 * 1024**3, 'Native memory guard differs')
    need([v['position'] for v in data['rows']] == NATIVE_POSITIONS, 'Native row coverage differs')
    rows = {}
    for spec in data['rows']:
        need(spec['bytes'] == VOCAB * 4 and spec['file'] == f"position-{spec['position']}.f32le", 'Native row descriptor differs')
        raw = captured(folder / 'capture', spec['file'], VOCAB * 4, spec['sha256'])
        row = struct.unpack('<' + str(VOCAB) + 'f', raw)
        # The frozen gate checks every element for finiteness, including a
        # native row not subsequently selected by one particular reference.
        prior.fractions(row, row)
        need(max(range(VOCAB), key=row.__getitem__) == spec['greedy_id'], 'Native greedy digest differs')
        rows[spec['position']] = row
    return rows


def reference_rows(folder, phase):
    data, receipt = document(folder / 'capture/result.json'), document(folder / 'checks.json')
    ids, positions = reference.inputs(phase)
    need(data['format'] == 'toklet-gemma-heretic-math-v1' and data['completed'] is True
         and receipt['collection_passed'] is True and receipt['exit_code'] == 0
         and receipt['changed_sources'] == [] and receipt['error'] is None, 'Completed independent capture required')
    need(data['phase'] == phase and data['tokens'] == ids and all(type(v) is int for v in data['tokens'])
         and data['selected_positions'] == positions and all(type(v) is int for v in data['selected_positions'])
         and data['repository'] == source.REPOSITORY and data['revision'] == source.REVISION
         and data['gate'] == dict(atol=1e-5, rtol=1e-4), 'Independent frozen contract differs')
    need(data['source']['sha256'] == data['source_endpoint_sha256'] == source.FILE_SHA256
         and data['source']['bytes'] == source.FILE_BYTES and data['source']['closed'] is True
         and data['source']['poisoned'] is False, 'Independent source read/endpoint differs')
    need(type(receipt['lifetime_peak_physical_bytes']) is int and 0 < receipt['lifetime_peak_physical_bytes'] <= 6 * 1024**3,
         'Independent memory guard differs')
    need(data['engine_executed'] is False and data['held_out_quality_scored'] is False
         and data['full_model_fidelity_qualified'] is False and data['managed_adoption_qualified'] is False,
         'Reference scope differs')
    body = read(folder / 'capture/states.bin', reference.MAX_CAPTURE)
    need(len(body) == data['planned_capture_bytes'] == len(positions) * VOCAB * 12
         and digest(body) == data['states_sha256'] and len(data['arrays']) == 2, 'Complete reference body required')
    rows, offset = {}, 0
    for spec, contract, code, size in zip(data['arrays'], ['math_fp64', 'staged_fp32'], ['d', 'f'], [8, 4], strict=True):
        count = len(positions) * VOCAB * size
        need(spec['contract'] == contract and spec['name'] == 'logits' and spec['positions'] == positions
             and spec['shape'] == [1, len(positions), VOCAB] and spec['dtype'] == '<f' + str(size)
             and spec['offset'] == offset and spec['bytes'] == count, 'Reference row descriptor differs')
        raw = captured(folder / 'capture', spec['logits_file'], count, spec['sha256'])
        need(raw == body[offset:offset + count], 'Reference duplicate body differs')
        values = struct.unpack('<' + str(VOCAB * len(positions)) + code, raw)
        for index, position in enumerate(positions):
            row = values[index * VOCAB:(index + 1) * VOCAB]
            prior.fractions(row, row)
            need(max(range(VOCAB), key=row.__getitem__) == data['contracts'][contract]['greedy_ids'][index], 'Reference greedy digest differs')
            rows[contract, position] = row
        offset += count
    need(offset == len(body), 'Reference body has trailing bytes')
    return rows


def compare(native, folders):
    need(set(folders) == set(reference.PHASES), 'All named context phases required')
    actual = native_rows(native)
    comparisons, prefix_checks, seen = [], [], {}
    for phase in reference.PHASES:
        rows = reference_rows(folders[phase], phase)
        for (contract, position), expected in rows.items():
            result = prior.fractions(actual[position], expected)
            result.update(phase=phase, contract=contract, position=position,
                          maximum_absolute_error=max(abs(a - b) for a, b in zip(actual[position], expected)),
                          native_greedy_id=max(range(VOCAB), key=actual[position].__getitem__),
                          reference_greedy_id=max(range(VOCAB), key=expected.__getitem__))
            comparisons.append(result)
            key = contract, position
            if key in seen:
                old = seen[key]
                exact = (struct.pack('<' + str(VOCAB) + 'f', *expected) == struct.pack('<' + str(VOCAB) + 'f', *old)
                         if contract == 'staged_fp32' else expected == old)
                maximum = max(abs(a - b) for a, b in zip(expected, old))
                passed = exact if contract == 'staged_fp32' else maximum <= 1e-10
                prefix_checks.append(dict(contract=contract, position=position, exact=exact,
                                          maximum_absolute_error=maximum, passed=passed))
            else:
                seen[key] = expected
    mismatches = [dict(phase=v['phase'], contract=v['contract'], position=v['position'])
                  for v in comparisons if v['native_greedy_id'] != v['reference_greedy_id']]
    numerical = all(v['failed_values'] == 0 for v in comparisons)
    prefixes = all(v['passed'] for v in prefix_checks)
    return dict(selected_context_fidelity_passed=numerical and prefixes and not mismatches,
                numerical_rows_passed=numerical, causal_prefixes_passed=prefixes,
                unresolved_greedy_mismatches=mismatches, comparisons=comparisons, prefix_checks=prefix_checks,
                gate=dict(atol=1e-5, rtol=1e-4), row_comparisons=len(comparisons),
                compared_values=sum(v['values'] for v in comparisons),
                scope='Selected native distributions through512 against both independent contracts. Near-tied greedy differences require investigation. Owned source/runtime execution proofs remain separate; no full held-out quality, managed adoption or iPhone qualification.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('native', type=Path)
    for phase in reference.PHASES:
        parser.add_argument('--' + phase, type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.native, {p: getattr(args, p) for p in reference.PHASES})
    print(json.dumps(result, indent=2, allow_nan=False))
    raise SystemExit(0 if result['selected_context_fidelity_passed'] else 1)
