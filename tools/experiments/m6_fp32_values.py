"""Producer-typed FP32 winner restoration; no tolerance or raw-evidence edits."""
import math
import struct

import m6_long_quality_audit as quality


def fp32(value):
    quality.number(value)
    try:
        packed = struct.pack('<f', value)
        restored = struct.unpack('<f', packed)[0]
    except (OverflowError, struct.error) as error:
        raise ValueError('Saved winner value outside finite FP32') from error
    if not math.isfinite(restored):
        raise ValueError('Saved winner value outside finite FP32')
    return restored, f'{struct.unpack("<I", packed)[0]:08x}'


def restore(records):
    """Copy only quality phase/row containers; never mutate saved raw inputs."""
    normalized, diagnostics = [], []
    values = changed = 0
    for record in records:
        if record.get('plan') != 'quality':
            normalized.append(record)
            continue
        clone = dict(record)
        clone['worker'] = dict(record['worker'])
        phases = []
        clone['worker']['phases'] = phases
        for phase in record['worker']['phases']:
            phase_copy = dict(phase)
            phases.append(phase_copy)
            events = []
            phase_copy['events'] = events
            for event in phase['events']:
                if event.get('event') != 'comparison_row':
                    events.append(event)
                    continue
                row = dict(event)
                events.append(row)
                exact, bits, original = {}, {}, {}
                for key in ('reference_winner_logits', 'online_winner_logits'):
                    pair = event.get(key)
                    if type(pair) is not list or len(pair) != 2:
                        raise ValueError('Invalid saved FP32 winner pair')
                    unpacked = [fp32(v) for v in pair]
                    row[key] = [v for v, _ in unpacked]
                    values += 2
                    changed += sum(a != b for a, b in zip(pair, row[key], strict=True))
                    original[key], exact[key], bits[key] = list(pair), list(row[key]), [b for _, b in unpacked]
                if row.get('reference_greedy') != row.get('online_greedy'):
                    diagnostics.append(dict(model=record['model'], depth=record['depth'],
                        target_index=row['target_index'], prefix_tokens=row['prefix_tokens'],
                        original_json_pairs=original, exact_fp32_pairs=exact,
                        exact_fp32_bits_hex=bits, investigation_complete=False))
        normalized.append(clone)
    return normalized, dict(restored_winner_values=values, decimal_values_changed=changed,
        changed_prediction_values=diagnostics,
        policy='Only four f32 winner values per comparison row restored; f64 margin/error/NLL and all frozen gates unchanged')
