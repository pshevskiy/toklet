"""Exact current-native bridge to already authenticated complete Qwen3 quality.

The caller authenticates journals, model/tokenizer, baseline, input, grader and
executable identities. This module runs no model or generated code. Timing and
resource observations are checked separately and cannot establish continuity.
"""
import json
import math

TIMINGS = frozenset(('first_token_seconds', 'prefill_seconds', 'decode_seconds',
                     'sampling_seconds', 'total_seconds'))
READY_RESOURCES = frozenset(('memory', 'admitted_bytes'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    if isinstance(value, dict):
        for item in value.values():
            finite(item)
    elif isinstance(value, list):
        for item in value:
            finite(item)
    elif isinstance(value, float):
        require(math.isfinite(value), 'nonfinite journal value')


def parse(raw):
    require(isinstance(raw, bytes) and 0 < len(raw) <= 32*1024**2
            and raw.endswith(b'\n'), 'complete bounded newline-terminated journal required')
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate journal key')
            result[key] = value
        return result
    def constant(_):
        raise ValueError('nonfinite JSON constant')
    rows = [json.loads(line, object_pairs_hook=pairs, parse_constant=constant)
            for line in raw.splitlines()]
    require(all(isinstance(row, dict) for row in rows), 'journal events must be objects')
    for row in rows:
        finite(row)
    return rows


def normalized(row):
    result = dict(row)
    if row.get('event') == 'ready':
        for key in READY_RESOURCES:
            result.pop(key, None)
    if row.get('event') == 'task_done':
        report = dict(row['report'])
        for key in TIMINGS:
            require(key in report and type(report[key]) in (int, float)
                    and math.isfinite(report[key]) and report[key] >= 0,
                    'complete finite timing observations required')
            del report[key]
        result['report'] = report
    return json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=False)


def complete(rows, mode):
    require(mode in ('loss', 'tasks') and len(rows) >= 2, 'complete explicit quality mode required')
    ready, done = rows[0], rows[-1]
    require(ready.get('event') == 'ready'
            and ready.get('format') == 'toklet-qwen3-quality-diagnostic-events-v1'
            and ready.get('mode') == mode and ready.get('full_quality_workload') is True
            and ready.get('debug_assertions') is False
            and ready.get('projection') == 'mlx_affine'
            and ready.get('normalization') == 'compensated'
            and ready.get('attention') == 'reference'
            and ready.get('activation_dtype') == 'FP16' and ready.get('kv_dtype') == 'FP16'
            and ready.get('context_capacity') == 1025 and ready.get('prefill_chunk_size') == 128,
            'frozen native Reference profile required')
    targets, tasks = (10000, 0) if mode == 'loss' else (0, 100)
    for record in (ready, done):
        require(type(record.get('targets')) is int and record['targets'] == targets
                and type(record.get('tasks')) is int and record['tasks'] == tasks,
                'complete target/task counts required')
    require(done.get('event') == 'done', 'terminal quality completion required')
    require(sum(row.get('event') == 'ready' for row in rows) == 1
            and sum(row.get('event') == 'done' for row in rows) == 1,
            'one readiness and terminal event required')
    require(sum(row.get('event') == 'loss' for row in rows) == targets
            and sum(row.get('event') == 'task_done' for row in rows) == tasks,
            'every quality target/task must be retained')
    require(type(ready.get('admitted_bytes')) is int
            and 0 < ready['admitted_bytes'] <= 6*1024**3
            and isinstance(ready.get('memory'), dict), 'bounded native resource admission required')
    for row in rows:
        finite(row)


def compare(current, historical, mode):
    """Compare all deterministic events; the historical journal must be trusted.

    Complete old protocol, per-target correctness and isolated grading are facts
    of the caller's authenticated historical evidence, not recreated by this
    adapter. Exact normalized event identity transfers those results to the
    current run. Unknown fields are retained, never heuristically ignored.
    """
    complete(current, mode)
    complete(historical, mode)
    differences = []
    unequal = 0
    for index in range(max(len(current), len(historical))):
        actual = normalized(current[index]) if index < len(current) else None
        expected = normalized(historical[index]) if index < len(historical) else None
        if actual != expected:
            unequal += 1
            if len(differences) < 16:
                differences.append(dict(index=index,
                    current_event=current[index].get('event') if index < len(current) else None,
                    historical_event=historical[index].get('event') if index < len(historical) else None))
    return dict(mode=mode, complete=True, identical=unequal == 0,
                current_events=len(current), historical_events=len(historical),
                different_events=unequal, first_differences=differences,
                targets=10000 if mode == 'loss' else 0,
                tasks=100 if mode == 'tasks' else 0,
                excluded_task_report_fields=sorted(TIMINGS),
                excluded_ready_resource_fields=sorted(READY_RESOURCES))
