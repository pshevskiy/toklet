"""Interpret optional vLLM 0.33 request clocks from a sealed HTTP capture.

This offline extension preserves the collector's client clocks and qualification
flags. Scheduler timestamps are server observations, not GPU operator clocks.
"""
import argparse
import base64
from contextlib import contextmanager
import hashlib
import math
import os
from pathlib import Path
import stat
import sys

import lfm3_http_metrics as capture

FIELDS = ('time_to_first_token_ms', 'generation_time_ms', 'queue_time_ms',
          'mean_itl_ms', 'tokens_per_second')
MAX_REPORT = 8 * 1024**2
MAX_JOURNAL = 96 * 1024**2
MAX_ENTRY = 8 * 1024**2
MAX_MS = capture.MAX_U64 / 1_000_000
REL_TOL = 1e-12
ABS_MS_TOL = 1e-9
OBSERVATION_KEYS = ('engine','stream_complete','frames','last_frame_ns','first_generated_text_ns',
                    'first_token_batch_ns','terminal_ns','text','output_token_ids','prompt_token_ids',
                    'reported_prompt_tokens','reported_output_tokens','cached_prompt_tokens','finish_reason','stop_reason')


def ratio(tokens, milliseconds):
    if milliseconds is None:
        return None, 'Required server duration is unavailable'
    if milliseconds <= 0:
        return None, 'A positive server interval is required'
    value = tokens * 1000.0 / milliseconds
    if not math.isfinite(value):
        return None, 'Rate exceeds finite floating-point range'
    return value, None


def interpret(metrics, output_tokens):
    capture.integer(output_tokens, 'completion tokens', 32)
    capture.require(metrics is None or type(metrics) is dict, 'Invalid per-request metrics object')
    values = {key: None for key in FIELDS}
    for key in FIELDS:
        value = None if metrics is None else metrics.get(key)
        if value is not None:
            capture.require(type(value) in (int, float) and 0 <= value <= sys.float_info.max
                            and math.isfinite(value),
                            'Invalid server metric: '+key)
            if key != 'tokens_per_second':
                capture.require(value <= MAX_MS, 'Server duration exceeds uint64-nanosecond range')
        values[key] = value
    ttft, generation, queue, itl, throughput = (values[key] for key in FIELDS)
    issues = []
    checks = 0
    if output_tokens <= 1 and itl is not None:
        checks += 1
        issues.append('Mean ITL requires more than one output token')
    if output_tokens > 1 and generation is not None and itl is not None:
        checks += 1
        if not math.isclose(itl, generation / (output_tokens-1), rel_tol=REL_TOL, abs_tol=ABS_MS_TOL):
            issues.append('Mean ITL differs from first-to-last interval divided by N-1')
    inference_ms = ttft + generation if ttft is not None and generation is not None else None
    if throughput is not None and inference_ms is not None:
        checks += 1
        expected, reason = ratio(output_tokens, inference_ms)
        if reason is not None or not math.isclose(throughput, expected, rel_tol=REL_TOL, abs_tol=1e-9):
            issues.append('Overall output throughput differs from N over scheduling-to-last interval')
    derived = dict(server_inference_seconds=None if inference_ms is None else inference_ms/1000,
        server_queue_plus_inference_seconds=None if queue is None or inference_ms is None else (queue+inference_ms)/1000,
        server_decode_tokens_per_second=None)
    unavailable = {}
    if output_tokens <= 1:
        unavailable['server_decode_tokens_per_second'] = 'N-1 decode rate requires more than one output token'
    else:
        derived['server_decode_tokens_per_second'], reason = ratio(output_tokens-1, generation)
        if reason is not None:
            unavailable['server_decode_tokens_per_second'] = reason
    if issues:
        derived['server_decode_tokens_per_second'] = None
        unavailable['server_decode_tokens_per_second'] = 'Server timing fields are arithmetically inconsistent'
    for key, value in values.items():
        if value is None:
            unavailable[key] = 'Not reported by the server; timestamp or metric emission may be unavailable'
    for key in ('server_inference_seconds', 'server_queue_plus_inference_seconds'):
        if derived[key] is None:
            unavailable[key] = 'One or more required server durations are unavailable'
    return dict(schema_interpretation='vllm-0.33-per-request-timing-v1', output_tokens=output_tokens,
        api_reported=values, derived=derived, unavailable=unavailable,
        arithmetic_consistent=(not issues) if checks else None,
        arithmetic_checks=checks, arithmetic_issues=issues, runtime_version_verified=False,
        boundaries=dict(time_to_first_token_ms='scheduled to first output token',
            generation_time_ms='first to last output token', queue_time_ms='queued to scheduled',
            mean_itl_ms='first-to-last interval divided by N-1',
            tokens_per_second='N over scheduling-to-last interval, including prefill/TTFT and excluding queue'),
        authority='Server-reported per-request scheduler clocks; no GPU operator or client/server clock conversion')


def identity(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


@contextmanager
def stable_file(path, maximum):
    before = path.lstat()
    capture.require(stat.S_ISREG(before.st_mode) and before.st_size <= maximum, 'Bounded regular input file required')
    descriptor = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    with os.fdopen(descriptor, 'rb') as stream:
        capture.require(identity(before) == identity(os.fstat(stream.fileno())), 'Input identity changed before read')
        yield stream
        capture.require(identity(before) == identity(os.fstat(stream.fileno())) == identity(path.lstat()),
                        'Input identity changed during read')


class Journal:
    def __init__(self, stream):
        self.stream, self.digest = stream, hashlib.sha256()
        self.bytes = self.lines = self.timestamp = 0

    def readline(self, limit):
        raw = self.stream.readline(min(MAX_ENTRY+1, MAX_JOURNAL-self.bytes+1))
        if not raw:
            return b''
        self.bytes += len(raw)
        self.lines += 1
        self.digest.update(raw)
        capture.require(len(raw) <= MAX_ENTRY and self.bytes <= MAX_JOURNAL
                        and self.lines <= capture.MAX_FRAMES*8 and raw.endswith(b'\n'),
                        'Wire journal bound or framing differs')
        row = capture.object_json(raw)
        timestamp = capture.integer(row.get('received_monotonic_ns'), 'recorded monotonic clock')
        capture.require(timestamp >= self.timestamp, 'Recorded wire clock regressed')
        self.timestamp = timestamp
        encoded = row.get('wire_base64')
        capture.require(type(encoded) is str, 'Missing exact wire bytes')
        data = base64.b64decode(encoded, validate=True)
        capture.require(bool(data), 'Empty decoded journal entry is not stream EOF')
        capture.require(len(data) <= limit and row.get('line_complete') is data.endswith(b'\n')
                        and row.get('wire_line') == data.decode('utf-8', errors='replace'),
                        'Recorded line bytes/framing differ')
        return data


class Discard:
    def write(self, data):
        return len(data)

    def flush(self):
        pass


def replay(report, journal):
    capture.require(report.get('format') == 'toklet-lfm3-http-metrics-v1'
                    and report.get('completed') is True and report.get('failure') is None
                    and report.get('http_status') == 200, 'A complete successful capture is required')
    expected = report['observation']
    capture.require(expected.get('engine') == 'vllm' and expected.get('stream_complete') is True,
                    'A complete vLLM capture is required')
    capture.require(type(expected.get('text')) is str, 'Invalid recorded completion text')
    ids = capture.token_ids(expected.get('output_token_ids'), 'recorded output IDs')
    capture.require(len(ids) <= 32, 'Recorded output IDs exceed the frozen cap')
    if expected.get('prompt_token_ids') is not None:
        capture.token_ids(expected['prompt_token_ids'], 'recorded prompt IDs')
    capture.integer(expected.get('reported_prompt_tokens'), 'recorded prompt count', capture.MAX_CONTEXT)
    capture.integer(expected.get('reported_output_tokens'), 'recorded output count', 32)
    capture.integer(expected.get('frames'), 'recorded frame count', capture.MAX_FRAMES)
    stop = expected.get('stop_reason')
    capture.require(stop is None or type(stop) in (int, str), 'Invalid recorded stop reason type')
    if type(stop) is int:
        capture.integer(stop, 'recorded stop token ID', capture.VOCAB-1)
    for key in ('last_frame_ns','first_generated_text_ns','first_token_batch_ns','terminal_ns','cached_prompt_tokens'):
        if expected.get(key) is not None:
            capture.integer(expected[key], 'recorded '+key)
    request = report['request']
    n = request.get('n', 1)
    capture.require(type(request.get('prompt')) is str and (n is None or type(n) is int and n == 1),
                    'Only one prompt and one output stream have attributable server clocks')
    capture.require(report.get('tool_sha256') == hashlib.sha256(Path(capture.__file__).read_bytes()).hexdigest(),
                    'Capture collector version differs')
    observed = capture.state('vllm')
    optional = None

    def events():
        nonlocal optional
        for data, timestamp in capture.frames(journal, 'vllm', Discard(), clock=lambda: journal.timestamp):
            if data != b'[DONE]':
                row = capture.object_json(data)
                if row.get('metrics') is not None:
                    capture.require(row.get('choices') == [] and type(row.get('usage')) is dict and optional is None,
                                    'Timing metrics must belong to the single final usage chunk')
                    optional = row['metrics']
            yield data, timestamp

    # The original start clock is neither reconstructed nor converted. Reuse
    # the validated parser with zero as an arrival lower bound, then compare
    # its exact recorded timestamps/counters/text with the sealed observation.
    capture.observe_vllm(events(), observed, 0, request['model'], [], 32)
    capture.require(journal.digest.hexdigest() == report['wire_sha256'], 'Wire journal digest differs')
    capture.require(all(observed[key] == expected[key] for key in OBSERVATION_KEYS), 'Capture observation differs from exact wire replay')
    return interpret(optional, observed['reported_output_tokens'])


def analyze(path, expected_sha):
    with stable_file(path, MAX_REPORT) as stream:
        raw = stream.read(MAX_REPORT+1)
        capture.require(len(raw) <= MAX_REPORT and hashlib.sha256(raw).hexdigest() == expected_sha,
                        'Capture report digest/size differs')
    report = capture.object_json(raw)
    del raw
    # Keep only the bounded schema needed for replay while processing the wire.
    # Unused provenance/extension objects remain in the original sealed file.
    if type(report.get('observation')) is dict:
        report['observation'] = {key:report['observation'].get(key) for key in OBSERVATION_KEYS}
    if type(report.get('request')) is dict:
        report['request'] = {key:report['request'][key] for key in ('model','prompt','n') if key in report['request']}
    report = {key:report.get(key) for key in ('format','completed','failure','http_status',
              'observation','request','tool_sha256','wire_sha256')}
    wire = path.parent/'wire.jsonl'
    with stable_file(wire, MAX_JOURNAL) as stream:
        metrics = replay(report, Journal(stream))
    return dict(format='toklet-lfm3-vllm-timeline-v1', completed=True, capture_sha256=expected_sha,
        wire_sha256=report['wire_sha256'], tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        timeline=metrics, client_clocks_preserved_in_source_capture=True,
        client_server_clock_subtraction=False, server_weight_identity_verified=False,
        model_quality_qualified=False, performance_qualified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--capture-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        report = analyze(args.capture, args.capture_sha256)
    except Exception as error:
        report = dict(format='toklet-lfm3-vllm-timeline-v1', completed=False,
            failure=str(error)[:1024], capture_sha256=args.capture_sha256,
            server_weight_identity_verified=False, model_quality_qualified=False, performance_qualified=False)
    with args.output.open('x') as stream:
        import json
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return 0 if report['completed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
