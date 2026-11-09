"""Portable completed-request/clock/coverage controls; no model or GPU."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import minicpm5_performance as reader

IDS=[6,978,44004,130073]
SHA='a'*64
FULL=dict(weights_bytes=2674218150,kv_bytes=22021272,scratch_bytes=21478868,
          history_bytes=2048,logits_bytes=1044480,metadata_bytes=5184,
          metadata_allowance_bytes=67108864,admitted_bytes=2785878866)
GREEDY=dict(FULL,scratch_bytes=21478900,logits_bytes=8,admitted_bytes=2784834426)


def owned():return dict(started_ns=1,completed_ns=1000000001,passed=True,exit_code=0,failure=None,owned_group_absent=True)


def fixture(mode='full-logits'):
    samples=[]
    for i in range(13):
        steps=[];event=100000
        for step in range(4):
            event+=1012000
            timing={k:.000001 for k in reader.TIMING_FIELDS if not k.startswith('gpu_')}
            timing.update(gpu_start_seconds=1.+i*.1+step*.001,
                          gpu_end_seconds=1.+i*.1+step*.001+.0001,command_count=1)
            steps.append(dict(step=step,rows=4 if step==0 else 1,committed_after=4+step,pending_after=0,
                wall_nanoseconds=1000000,sampling_nanoseconds=10000,decision_nanoseconds=event,
                selected=IDS[step],full_logits_available=mode=='full-logits',completed_timing=timing))
        samples.append(dict(sample=i,warmup=i<3,request_nanoseconds=event+20000,steps=steps,
            committed_after=7,pending_after=1,history_after=8,outcome='fixed_teacher_continuation_completed'))
    return dict(format='toklet-minicpm5-raw-request-bench-v1',repository=reader.REPOSITORY,revision=reader.REVISION,
        source_manifest_sha256=SHA,device='Apple M1 Pro',profile='MlxAffine/Compensated/Reference/Half',readback=mode,
        context_capacity=512,prefill_chunk=32,position_offset=0,prompt=[0,7,13,3],teacher_tokens=IDS.copy(),
        warmups=3,measured_samples=10,cache_state='loaded-model;reset-empty-history-per-request;warm-filesystem',
        stop_policy='fixed-four-teacher-tokens-including-stop-id;last-token-pending',
        load_timing=dict(load_seconds=.001,conversion_seconds=.0006,compilation_seconds=.0002),
        session_memory=copy.deepcopy(FULL if mode=='full-logits' else GREEDY),application_allowance_bytes=33554432,
        request_clock_scope='set_tokens through final pending append; completed evaluation and selection; excludes reset and JSON',
        decision_ttft_scope='internal raw token decision; not streamed text or visible UI',diagnostics_enabled=False,
        operator_counts_available=False,allocation_count=None,user_visible_ttft_nanoseconds=None,samples=samples,
        model_quality_qualified=False,managed_adoption_qualified=False,ordinary_telemetry_overhead_qualified=False)


def validate(raw,owner=None,mode='full-logits'):
    return reader.validate(raw,owned() if owner is None else owner,mode,IDS,FULL if mode=='full-logits' else GREEDY,SHA)


class CompletedRequests(unittest.TestCase):
    def test_report_bytes_bind_owned_receipt_and_duplicate_fields_reject(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'bench.json'
            raw=json.dumps(fixture()).encode();path.write_bytes(raw)
            owner=dict(owned(),benchmark_report_sha256=hashlib.sha256(raw).hexdigest())
            self.assertEqual(reader.read_report(path,owner)['format'],'toklet-minicpm5-raw-request-bench-v1')
            with self.assertRaises(ValueError):reader.read_report(path,dict(owner,benchmark_report_sha256='0'*64))
            raw=b'{"sample":0,"sample":1}';path.write_bytes(raw)
            owner['benchmark_report_sha256']=hashlib.sha256(raw).hexdigest()
            with self.assertRaisesRegex(ValueError,'Duplicate'):reader.read_report(path,owner)

    def test_both_readbacks_complete_exact_work_and_event_denominators(self):
        for mode in ['full-logits','greedy-token']:
            result=validate(fixture(mode),mode=mode)
            self.assertEqual(result['measured_requests'],10)
            self.assertEqual(result['warmup_requests'],3)
            self.assertEqual(result['completed_prompt_tokens'],52)
            self.assertEqual(result['completed_decode_tokens'],39)
            self.assertEqual(result['sampled_tokens'],52)
            row=result['measured'][0]
            self.assertEqual(row['decision_ttft_ns'],1112000)
            self.assertEqual(row['decode_interval_ns'],3036000)
            self.assertAlmostEqual(row['decode_tokens_per_second'],3e9/3036000)
            self.assertEqual(row['prefill_tokens_per_second'],4000)
            self.assertEqual(row['command_count'],4)
            self.assertFalse(result['ordinary_telemetry_overhead_qualified'])

    def test_owned_interval_success_and_typed_clocks_required(self):
        for change in [dict(started_ns=False),dict(started_ns=0),dict(completed_ns=1),
                       dict(completed_ns=1000000001.),dict(passed=False),dict(exit_code=True),
                       dict(failure='timeout'),dict(owned_group_absent=False)]:
            owner=owned();owner.update(change)
            with self.assertRaises(ValueError):validate(fixture(),owner)

    def test_all_warmup_and_measured_request_clocks_must_fit_owned_lifetime(self):
        raw=fixture()
        for row in raw['samples']:row['request_nanoseconds']=80000000
        # All13individual80ms requests fit a1s lifetime; their disjoint sum does not.
        with self.assertRaisesRegex(ValueError,'exceed owned'):validate(raw)

    def test_disjoint_load_and_all_requests_must_share_the_owned_envelope(self):
        raw=fixture();raw['load_timing']=dict(load_seconds=.99,conversion_seconds=.5,compilation_seconds=.4)
        # The load and all13request clocks each fit separately; together they do not.
        with self.assertRaisesRegex(ValueError,'Disjoint load'):validate(raw)
        raw['load_timing']['load_seconds']=.9
        self.assertEqual(validate(raw)['completed_requests'],13)

    def test_disjoint_host_sum_rejects_but_gpu_is_overlapping(self):
        raw=fixture();timing=raw['samples'][0]['steps'][0]['completed_timing']
        for key in reader.TIMING_FIELDS:
            if not key.startswith('gpu_'):timing[key]=.0002
        with self.assertRaisesRegex(ValueError,'Disjoint host'):validate(raw)
        raw=fixture();timing=raw['samples'][0]['steps'][0]['completed_timing']
        timing['wait_seconds']=.0009;timing['gpu_end_seconds']=timing['gpu_start_seconds']+.00095
        self.assertEqual(validate(raw)['completed_requests'],13)

    def test_duration_conversion_has_bounded_subnanosecond_grain(self):
        for value in [0,1e-9,1.5e-9,2**20]:
            self.assertLessEqual(abs(reader.duration_ns(value)-value*10**9),.5)
        for value in [True,-1,2**20+1,float('inf'),float('nan'),10**1000]:
            with self.assertRaises(ValueError):reader.duration_ns(value)

    def test_missing_reordered_untyped_or_wrong_scope_requests_reject(self):
        for change in [dict(sample=True),dict(warmup=False),dict(request_nanoseconds=0),
                       dict(request_nanoseconds=5000000.),dict(committed_after=8),dict(pending_after=0),
                       dict(history_after=7),dict(outcome='stop')]:
            raw=fixture();raw['samples'][0].update(change)
            with self.assertRaises(ValueError):validate(raw)
        raw=fixture();raw['samples'].pop()
        with self.assertRaises(ValueError):validate(raw)
        raw=fixture();raw['samples'][3],raw['samples'][4]=raw['samples'][4],raw['samples'][3]
        with self.assertRaises(ValueError):validate(raw)

    def test_phase_work_selection_availability_and_disjoint_clock_damage_rejects(self):
        for change in [dict(step=True),dict(rows=1),dict(committed_after=3),dict(pending_after=1),
                       dict(selected=7),dict(full_logits_available=1),dict(wall_nanoseconds=2000000),
                       dict(sampling_nanoseconds=10000.),dict(decision_nanoseconds=True),dict(decision_nanoseconds=10000000)]:
            raw=fixture();raw['samples'][0]['steps'][0].update(change)
            with self.assertRaises(ValueError):validate(raw)
        raw=fixture();raw['samples'][0]['steps'].pop()
        with self.assertRaises(ValueError):validate(raw)

    def test_gpu_host_counter_and_admission_damage_rejects(self):
        for change in [dict(command_count=True),dict(wait_seconds=float('nan')),dict(gpu_start_seconds=0),
                       dict(gpu_end_seconds=.5),dict(gpu_end_seconds=5),dict(host_encode_seconds=.1)]:
            raw=fixture();raw['samples'][0]['steps'][0]['completed_timing'].update(change)
            with self.assertRaises(ValueError):validate(raw)
        for change in [dict(admitted_bytes=1),dict(weights_bytes=True),dict(logits_bytes=0)]:
            raw=fixture();raw['session_memory'].update(change)
            with self.assertRaises(ValueError):validate(raw)

    def test_identity_policy_and_unavailable_qualification_relabelling_reject(self):
        for change in [dict(revision='main'),dict(prompt=[False,7,13,3]),dict(teacher_tokens=[6,978,44004,1]),
                       dict(context_capacity=512.),dict(prefill_chunk=128),dict(position_offset=True),
                       dict(warmups=True),dict(readback='greedy-token'),dict(diagnostics_enabled=True),
                       dict(operator_counts_available=True),dict(allocation_count=0),
                       dict(user_visible_ttft_nanoseconds=0),dict(model_quality_qualified=True),
                       dict(managed_adoption_qualified=True),dict(ordinary_telemetry_overhead_qualified=True),
                       dict(engine_speedup_qualified=True),dict(stop_policy='eos'),dict(cache_state='reused-prefix')]:
            raw=fixture();raw.update(change)
            with self.assertRaises(ValueError):validate(raw)
        self.assertFalse(reader.finite(10**1000))


if __name__=='__main__':unittest.main()
