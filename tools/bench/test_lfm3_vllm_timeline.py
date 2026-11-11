import base64
import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import lfm3_http_metrics as capture
import lfm3_vllm_timeline as timeline

MISSING = object()


def metrics():
    return dict(time_to_first_token_ms=50, generation_time_ms=1500, queue_time_ms=1000,
                mean_itl_ms=500, tokens_per_second=4/1.55)


def fixture(folder, value=MISSING, *, misplaced=False, n=1, stop_reason=None):
    identity = dict(id='cmpl-fixture', object='text_completion', model='fixture-only')
    first = dict(**identity, choices=[dict(index=0, text='abcd', token_ids=[11,12,13,14],
        prompt_token_ids=[1,2], finish_reason='stop', stop_reason=stop_reason)])
    usage = dict(**identity, choices=[], usage=dict(prompt_tokens=2, completion_tokens=4,
        total_tokens=6, prompt_tokens_details=dict(cached_tokens=0)))
    if value is not MISSING:
        (first if misplaced else usage)['metrics'] = value
    wire = b''.join(b'data: '+json.dumps(row, allow_nan=False).encode()+b'\n\n' for row in [first,usage])+b'data: [DONE]\n\n'
    raw = io.StringIO()
    clocks = iter(range(100,1000))
    observed = capture.state('vllm')
    capture.observe_vllm(capture.frames(io.BytesIO(wire),'vllm',raw,clock=lambda:next(clocks)),
                         observed,0,'fixture-only',[1,2],32)
    payload = raw.getvalue().encode()
    report = dict(format='toklet-lfm3-http-metrics-v1',completed=True,failure=None,http_status=200,
        request=dict(model='fixture-only',prompt='raw fixture',n=n),observation=observed,
        tool_sha256=hashlib.sha256(Path(capture.__file__).read_bytes()).hexdigest(),
        wire_sha256=hashlib.sha256(payload).hexdigest(),provenance=dict(synthetic_fixture=True),
        server_weight_identity_verified=False,model_quality_qualified=False,performance_qualified=False)
    path = folder/'report.json'
    path.write_text(json.dumps(report,allow_nan=False));(folder/'wire.jsonl').write_bytes(payload)
    return path, report


def analyze(path):
    return timeline.analyze(path,hashlib.sha256(path.read_bytes()).hexdigest())


class TimelineTests(unittest.TestCase):
    def test_server_decode_and_overall_throughput_have_different_denominators(self):
        result=timeline.interpret(metrics(),4)
        self.assertTrue(result['arithmetic_consistent']);self.assertEqual(result['arithmetic_checks'],2)
        self.assertEqual(result['derived']['server_decode_tokens_per_second'],2)
        self.assertAlmostEqual(result['api_reported']['tokens_per_second'],4/1.55)
        self.assertAlmostEqual(result['derived']['server_inference_seconds'],1.55)
        self.assertAlmostEqual(result['derived']['server_queue_plus_inference_seconds'],2.55)

    def test_missing_and_null_metrics_stay_unavailable(self):
        for value in (None,{}):
            with self.subTest(value=value):
                result=timeline.interpret(value,4)
                self.assertTrue(all(v is None for v in result['api_reported'].values()))
                self.assertIsNone(result['arithmetic_consistent']);self.assertFalse(result['runtime_version_verified'])
                self.assertIsNone(result['derived']['server_decode_tokens_per_second'])

    def test_partial_fields_and_explicit_zero_are_not_filled_in(self):
        result=timeline.interpret(dict(queue_time_ms=0),4)
        self.assertEqual(result['api_reported']['queue_time_ms'],0)
        self.assertIsNone(result['api_reported']['generation_time_ms'])
        self.assertIsNone(result['arithmetic_consistent']);self.assertEqual(result['arithmetic_checks'],0)

    def test_zero_or_one_token_never_has_a_decode_rate(self):
        for count in (0,1):
            result=timeline.interpret(dict(generation_time_ms=0),count)
            self.assertIsNone(result['derived']['server_decode_tokens_per_second'])
            self.assertIn('N-1',result['unavailable']['server_decode_tokens_per_second'])

    def test_zero_decode_interval_has_no_infinite_rate(self):
        result=timeline.interpret(dict(time_to_first_token_ms=500,generation_time_ms=0,
            mean_itl_ms=0,tokens_per_second=8),4)
        self.assertTrue(result['arithmetic_consistent'])
        self.assertIsNone(result['derived']['server_decode_tokens_per_second'])

    def test_inconsistent_itl_preserves_reported_values_and_withholds_rate(self):
        value=metrics();value['mean_itl_ms']=600;result=timeline.interpret(value,4)
        self.assertFalse(result['arithmetic_consistent']);self.assertEqual(result['api_reported']['mean_itl_ms'],600)
        self.assertIsNone(result['derived']['server_decode_tokens_per_second'])

    def test_overall_rate_cannot_be_relabelled_as_decode_rate(self):
        value=metrics();value['tokens_per_second']=2;result=timeline.interpret(value,4)
        self.assertFalse(result['arithmetic_consistent']);self.assertTrue(result['arithmetic_issues'])
        self.assertIsNone(result['derived']['server_decode_tokens_per_second'])

    def test_itl_with_one_token_is_inconsistent(self):
        result=timeline.interpret(dict(mean_itl_ms=0),1)
        self.assertFalse(result['arithmetic_consistent'])

    def test_invalid_timing_types_and_range_rejected(self):
        for value in (True,-1,float('nan'),float('inf'),'1',10**1000,timeline.MAX_MS+1):
            with self.subTest(value=str(value)[:20]),self.assertRaises(ValueError):
                timeline.interpret(dict(generation_time_ms=value),4)

    def test_invalid_token_counts_or_metric_objects_rejected(self):
        for count in (True,-1,4.0,33):
            with self.subTest(count=count),self.assertRaises(ValueError):timeline.interpret(metrics(),count)
        for value in ([],False,'metrics'):
            with self.subTest(value=value),self.assertRaises(ValueError):timeline.interpret(value,4)

    def test_extreme_finite_rate_arithmetic_cannot_emit_infinity(self):
        result=timeline.interpret(dict(generation_time_ms=5e-324),4)
        self.assertIsNone(result['derived']['server_decode_tokens_per_second'])
        json.dumps(result,allow_nan=False)

    def test_authenticated_wire_replay_produces_additive_unqualified_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,report=fixture(Path(tmp),metrics());before=path.read_bytes();result=analyze(path)
            self.assertTrue(result['completed']);self.assertEqual(result['timeline']['derived']['server_decode_tokens_per_second'],2)
            self.assertFalse(result['client_server_clock_subtraction']);self.assertFalse(result['performance_qualified'])
            self.assertFalse(result['server_weight_identity_verified']);self.assertEqual(path.read_bytes(),before)

    def test_legacy_absent_or_null_optional_field_replays(self):
        for value in (MISSING,None,{}):
            with tempfile.TemporaryDirectory() as tmp,self.subTest(value=value):
                path,report=fixture(Path(tmp),value);result=analyze(path)
                self.assertTrue(result['completed']);self.assertIsNone(result['timeline']['api_reported']['queue_time_ms'])

    def test_metrics_must_belong_to_final_usage_and_single_stream(self):
        for misplaced,n in [(True,1),(False,2),(False,True)]:
            with tempfile.TemporaryDirectory() as tmp,self.subTest(misplaced=misplaced,n=n):
                path,report=fixture(Path(tmp),metrics(),misplaced=misplaced,n=n)
                with self.assertRaises(ValueError):analyze(path)

    def test_wire_binding_and_observation_binding_reject_tampering(self):
        for kind in ('wire','observation','collector'):
            with tempfile.TemporaryDirectory() as tmp,self.subTest(kind=kind):
                path,report=fixture(Path(tmp),metrics())
                if kind=='wire':
                    wire=path.parent/'wire.jsonl';wire.write_bytes(wire.read_bytes().replace(b'"line_complete": true',b'"line_complete" : true',1))
                else:
                    if kind=='observation':report['observation']['reported_output_tokens']=5
                    if kind=='collector':report['tool_sha256']='0'*64
                    path.write_text(json.dumps(report))
                with self.assertRaises(ValueError):analyze(path)

    def test_duplicate_nonfinite_or_partial_json_inputs_rejected(self):
        for kind in ('duplicate','nonfinite','partial'):
            with tempfile.TemporaryDirectory() as tmp,self.subTest(kind=kind):
                path,report=fixture(Path(tmp),metrics())
                if kind=='duplicate':path.write_bytes(path.read_bytes().replace(b'"http_status": 200',b'"http_status": 200, "http_status": 200',1))
                if kind=='nonfinite':path.write_bytes(path.read_bytes().replace(b'"http_status": 200',b'"http_status": NaN',1))
                if kind=='partial':
                    wire=path.parent/'wire.jsonl';wire.write_bytes(wire.read_bytes()[:-1])
                with self.assertRaises(ValueError):analyze(path)

    def test_wrong_capture_digest_and_incomplete_capture_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,report=fixture(Path(tmp),metrics())
            with self.assertRaises(ValueError):timeline.analyze(path,'0'*64)
            report['completed']=False;path.write_text(json.dumps(report))
            with self.assertRaises(ValueError):analyze(path)

    def test_base64_and_decoded_line_must_agree(self):
        for kind in ('base64','decoded','complete'):
            with tempfile.TemporaryDirectory() as tmp,self.subTest(kind=kind):
                path,report=fixture(Path(tmp),metrics());wire=path.parent/'wire.jsonl'
                lines=wire.read_bytes().splitlines();row=json.loads(lines[0])
                if kind=='base64':row['wire_base64']='%%%'
                if kind=='decoded':row['wire_line']='different\n'
                if kind=='complete':row['line_complete']=False
                lines[0]=json.dumps(row).encode();data=b'\n'.join(lines)+b'\n';wire.write_bytes(data)
                report['wire_sha256']=hashlib.sha256(data).hexdigest();path.write_text(json.dumps(report))
                with self.assertRaises(ValueError):analyze(path)

    def test_actual_read_bounds_are_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,report=fixture(Path(tmp),metrics())
            with patch.object(timeline,'MAX_REPORT',10),self.assertRaises(ValueError):analyze(path)
            with patch.object(timeline,'MAX_ENTRY',10),self.assertRaises(ValueError):analyze(path)
            with patch.object(timeline,'MAX_JOURNAL',10),self.assertRaises(ValueError):analyze(path)

    def test_symlinks_and_concurrent_identity_change_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);path,report=fixture(folder,metrics());link=folder/'link.json';link.symlink_to(path)
            with self.assertRaises(ValueError):analyze(link)
            with self.assertRaises(ValueError):
                with timeline.stable_file(path,timeline.MAX_REPORT) as stream:
                    stream.read(1);path.write_bytes(path.read_bytes()+b' ')

    def test_recorded_integer_counters_and_clocks_cannot_be_floats_or_booleans(self):
        for key,value in [('reported_output_tokens',4.0),('reported_prompt_tokens',True),
                          ('frames',3.0),('terminal_ns',False)]:
            with tempfile.TemporaryDirectory() as tmp,self.subTest(key=key):
                path,report=fixture(Path(tmp),metrics());report['observation'][key]=value
                path.write_text(json.dumps(report))
                with self.assertRaises(ValueError):analyze(path)

    def test_unused_provenance_and_extensions_remain_in_sealed_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,report=fixture(Path(tmp),metrics());report['provenance']['extension']=[{},[]]*100
            report['observation']['extension']=[{},[]]*100;report['request']['extension']=[{},[]]*100
            path.write_text(json.dumps(report));before=path.read_bytes();result=analyze(path)
            self.assertTrue(result['completed']);self.assertEqual(path.read_bytes(),before)

    def test_empty_decoded_entry_after_done_cannot_hide_unhashed_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,report=fixture(Path(tmp),metrics());wire=path.parent/'wire.jsonl'
            last=json.loads(wire.read_bytes().splitlines()[-1])
            empty=dict(received_monotonic_ns=last['received_monotonic_ns']+1,
                wire_base64='',wire_line='',line_complete=False)
            prefix=wire.read_bytes()+json.dumps(empty).encode()+b'\n'
            wire.write_bytes(prefix+b'UNHASHED TRAILING BYTES\n')
            report['wire_sha256']=hashlib.sha256(prefix).hexdigest();path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,'Empty decoded'):analyze(path)

    def test_stop_reason_metadata_must_have_the_wire_integer_type(self):
        for value in (True,1.0):
            with tempfile.TemporaryDirectory() as tmp,self.subTest(value=value):
                path,report=fixture(Path(tmp),metrics(),stop_reason=1)
                self.assertTrue(analyze(path)['completed'])
                report['observation']['stop_reason']=value;path.write_text(json.dumps(report))
                with self.assertRaisesRegex(ValueError,'recorded stop reason type'):analyze(path)


if __name__=='__main__':unittest.main()
