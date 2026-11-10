"""Completed text clocks, multichunk work, parity and owner negative controls."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import minicpm5_text_performance as reader
from test_minicpm5_rms_performance import BASE, PARALLEL


EXPECTED = dict(sampled_ids=[42, 130073], text='OK', source_numerical_qualification_extended=False)


def contract(prompt=14):
    work = dict(prompt_ids=list(range(prompt)), maximum_new_tokens=16,
                options=dict(enable_thinking=False, add_generation_prompt=True),
                messages_sha256='a'*64, options_sha256='b'*64)
    return dict(workloads=dict(test=work), tokenizer_sha256='c'*64,
                codec_allowance_bytes=536870912, stream_allowance_bytes=4194304,
                application_allowance_bytes=33554432,
                request_clock_scope='request test scope', first_text_scope='decoded test scope',
                expected_session_memory={'serial':BASE, 'parallel-store':PARALLEL})


def owned():
    return dict(started_ns=1, completed_ns=20000000001, passed=True, exit_code=0,
                failure=None, owned_group_absent=True, final_wait4_peak_rss_bytes=4*1024**3,
                sampled_lifetime_physical_peak_bytes=3*1024**3,
                minimum_observed_free_bytes=3*1024**3, final_physical_peak_available=False)


def report(variant='serial', prompt=14, expected=None):
    expected = EXPECTED if expected is None else expected
    config=contract(prompt);work=config['workloads']['test'];samples=[]
    for i in range(13):
        chunks=[];decisions=[];committed=0;clock=60
        for step, token in enumerate(expected['sampled_ids']):
            rows=([32]*(prompt//32)+([prompt%32] if prompt%32 else [])) if step==0 else [1]
            first=len(chunks)
            for n in rows:
                committed+=n;index=len(chunks)
                timing={key:.0001 for key in reader.common.TIMING_FIELDS if not key.startswith('gpu_')}
                begin=100.+i*.5+index*.1
                timing.update(gpu_start_seconds=begin,gpu_end_seconds=begin+.09,command_count=1)
                chunks.append(dict(decision=step,rows=n,committed_after=committed,
                                   pending_after=max(0,prompt-committed),wall_ns=100000000,completed_timing=timing))
            clock+=len(rows)*100000000+500
            decisions.append(dict(step=step,selected=token,decision_ns=clock,selection_ns=100,
                                  decode_ns=100,delta_bytes=len(expected['text'].encode()) if step==0 else 0,
                                  text_bytes=len(expected['text'].encode()),first_chunk=first,chunks=len(rows),
                                  committed_after=prompt+step,pending_after=1))
            clock+=100
        count=len(decisions)
        samples.append(dict(sample=i,warmup=i<3,request_ns=clock+1000000,render_ns=10,encode_ns=20,
                            stream_setup_ns=30,stream_finish_ns=100,finish_bytes=0,
                            first_decoded_text_ns=decisions[0]['decision_ns']+50 if expected['text'] else None,
                            full_decode_validation_ns=100,input_tokens=prompt,
                            sampled_ids=expected['sampled_ids'].copy(),text=expected['text'],chunks=chunks,decisions=decisions,
                            committed_after=prompt+count-1,pending_after=1,history_after=prompt+count,
                            stop_reason='eos' if expected['sampled_ids'][-1] in [1,130073] else 'limit'))
    memory=BASE if variant=='serial' else PARALLEL
    return dict(format='toklet-minicpm5-text-request-v1',repository=reader.common.REPOSITORY,
                revision=reader.common.REVISION,device='Apple M1 Pro',profile=reader.common.PROFILE,
                normalization_implementation='serial' if variant=='serial' else 'serial-reduction-parallel-store',
                tokenizer_sha256=config['tokenizer_sha256'],options=work['options'],messages_sha256=work['messages_sha256'],
                options_sha256=work['options_sha256'],prompt_ids=work['prompt_ids'],stop_ids=[1,130073],maximum_new_tokens=16,
                capacity=512,prefill_chunk=32,position_offset=0,warmups=3,measured_samples=10,
                cache_state='one loaded model/codec/session;reset-empty-history-per-request;warm-filesystem',
                load_timing=dict(load_seconds=1.,conversion_seconds=.6,compilation_seconds=.2),
                codec_load_ns=100,preflight_text_ns=100,session_setup_ns=100,session_memory=copy.deepcopy(memory),
                codec_allowance_bytes=536870912,stream_allowance_bytes=4194304,application_allowance_bytes=33554432,
                maximum_admitted_bytes=memory['admitted_bytes']+574619648,samples=samples,
                request_clock_scope=config['request_clock_scope'],first_text_scope=config['first_text_scope'],
                diagnostics_enabled=False,operator_counts_available=False,allocation_count=None,console_publication_ns=None,
                user_visible_ttft_ns=None,model_quality_qualified=False,managed_adoption_qualified=False,
                ordinary_telemetry_overhead_qualified=False)


def parity(prompt=14):
    return dict(format='toklet-minicpm5-text-parity-v1',profile=reader.common.PROFILE,capacity=512,prefill_chunk=32,
                position_offset=0,stop_ids=[1,130073],repository=reader.common.REPOSITORY,revision=reader.common.REVISION,
                prompt_ids=list(range(prompt)),sampled_ids=EXPECTED['sampled_ids'].copy(),text='OK',maximum_new_tokens=16,
                combined_admitted_bytes=BASE['admitted_bytes']+PARALLEL['admitted_bytes']-2674218150+574619648,
                shared_model_bytes=2674218150,serial_memory=copy.deepcopy(BASE),parallel_memory=copy.deepcopy(PARALLEL),
                codec_allowance_bytes=536870912,stream_allowance_bytes=4194304,application_allowance_bytes=33554432,
                steps=[dict(step=i,logits_values=130560,logits_sha256='d'*64,all_logits_exact=True,selected=token,
                            committed_after=prompt+i,pending_after=1) for i,token in enumerate(EXPECTED['sampled_ids'])],
                pending_after=1,history_after=prompt+2,committed_after=prompt+1,model_quality_adoption_qualified=False)


def validate(raw=None, owner=None, prompt=14, variant='serial', expected=None):
    return reader.validate(report(variant,prompt) if raw is None else raw,owned() if owner is None else owner,
                           contract(prompt),'test',variant,EXPECTED if expected is None else expected)


class TextRequests(unittest.TestCase):
    def test_both_paths_and_multichunk_prefill_count_completed_work(self):
        for prompt in [14,32,38,64]:
            for variant in ['serial','parallel-store']:
                value=validate(prompt=prompt,variant=variant)
                row=value['measured'][0]
                self.assertEqual(value['measured_requests'],10)
                self.assertEqual(value['sampled_tokens'],26)
                self.assertEqual(row['command_count'],(prompt+31)//32+1)
                self.assertAlmostEqual(row['decode_tokens_per_second'],1e9/100000600)
                self.assertFalse(value['deeper_source_numerical_qualification'])

    def test_parity_binds_all_logits_stops_state_and_shared_memory(self):
        self.assertEqual(reader.validate_parity(parity(),owned(),contract(),'test'),EXPECTED)
        changes=[dict(profile='other'),dict(prompt_ids=[0]),dict(stop_ids=[1]),dict(pending_after=0),
                 dict(model_quality_adoption_qualified=True),dict(shared_model_bytes=0),dict(sampled_ids=[1,130073]),
                 dict(combined_admitted_bytes=1),dict(maximum_new_tokens=True)]
        for change in changes:
            raw=parity();raw.update(change)
            with self.assertRaises(ValueError):reader.validate_parity(raw,owned(),contract(),'test')
        for change in [dict(logits_values=1),dict(all_logits_exact=1),dict(logits_sha256='z'*64),
                       dict(selected=7),dict(committed_after=0)]:
            raw=parity();raw['steps'][0].update(change)
            with self.assertRaises(ValueError):reader.validate_parity(raw,owned(),contract(),'test')

    def test_owned_success_and_real_resource_scope_required(self):
        for change in [dict(passed=False),dict(exit_code=True),dict(owned_group_absent=False),dict(completed_ns=True),
                       dict(final_wait4_peak_rss_bytes=6*1024**3+1),dict(sampled_lifetime_physical_peak_bytes=0),
                       dict(minimum_observed_free_bytes=2*1024**3-1),dict(final_physical_peak_available=True)]:
            owner=owned();owner.update(change)
            with self.assertRaises(ValueError):validate(owner=owner)

    def test_input_identity_types_and_unavailable_metrics_cannot_be_relabelled(self):
        for change in [dict(normalization_implementation='parallel'),dict(prompt_ids=[0]),dict(options_sha256='z'*64),
                       dict(options=dict(enable_thinking=0,add_generation_prompt=True)),dict(warmups=True),
                       dict(console_publication_ns=1),dict(user_visible_ttft_ns=1),dict(allocation_count=0),
                       dict(ordinary_telemetry_overhead_qualified=True),dict(extra=1)]:
            raw=report();raw.update(change)
            with self.assertRaises(ValueError):validate(raw)

    def test_expected_native_sequence_is_typed_bounded_and_not_source_qualification(self):
        for change in [dict(sampled_ids=[True,130073]),dict(sampled_ids=[130560,130073]),dict(sampled_ids=[]),
                       dict(sampled_ids=[1,130073]),dict(sampled_ids=[42]),dict(text='x'*65537),
                       dict(source_numerical_qualification_extended=True)]:
            expected=dict(EXPECTED,**change)
            with self.assertRaises(ValueError):validate(expected=expected)

    def test_balanced_ledger_still_requires_exact_individual_owners(self):
        raw=report();raw['session_memory']['kv_bytes']-=1;raw['session_memory']['metadata_allowance_bytes']+=1
        with self.assertRaises(ValueError):validate(raw)
        raw=report();raw['maximum_admitted_bytes']-=1
        with self.assertRaises(ValueError):validate(raw)

    def test_sample_work_history_and_stop_are_complete(self):
        for change in [dict(sample=True),dict(warmup=False),dict(sampled_ids=[42]),dict(text='NO'),dict(stop_reason='limit'),
                       dict(input_tokens=1),dict(committed_after=16),dict(history_after=15),dict(pending_after=0)]:
            raw=report();raw['samples'][0].update(change)
            with self.assertRaises(ValueError):validate(raw)
        raw=report();raw['samples'].pop()
        with self.assertRaises(ValueError):validate(raw)

    def test_multichunk_rows_and_committed_prefix_cannot_be_collapsed(self):
        for change in [dict(rows=38),dict(pending_after=0),dict(committed_after=38),dict(decision=True)]:
            raw=report(prompt=38);raw['samples'][0]['chunks'][0].update(change)
            with self.assertRaises(ValueError):validate(raw,prompt=38)
        raw=report(prompt=38);raw['samples'][0]['decisions'][0]['chunks']=1
        with self.assertRaises(ValueError):validate(raw,prompt=38)

    def test_disjoint_host_phase_sum_and_event_windows_are_enclosed(self):
        raw=report();timing=raw['samples'][0]['chunks'][0]['completed_timing']
        for key in reader.common.TIMING_FIELDS:
            if not key.startswith('gpu_'):timing[key]=.02
        with self.assertRaisesRegex(ValueError,'host sum'):validate(raw)
        raw=report();raw['samples'][0]['decisions'][0]['selection_ns']=10000
        with self.assertRaisesRegex(ValueError,'event window'):validate(raw)
        raw=report();raw['samples'][0]['stream_finish_ns']=2000000
        with self.assertRaisesRegex(ValueError,'request phase'):validate(raw)

    def test_gpu_commands_are_serial_and_whole_lifetime_is_bounded(self):
        raw=report();timing=raw['samples'][0]['chunks'][1]['completed_timing'];timing['gpu_start_seconds']-=.1
        with self.assertRaisesRegex(ValueError,'serial order'):validate(raw)
        raw=report()
        for chunk in raw['samples'][-1]['chunks']:
            chunk['completed_timing']['gpu_start_seconds']+=3600
            chunk['completed_timing']['gpu_end_seconds']+=3600
        with self.assertRaisesRegex(ValueError,'whole GPU series'):validate(raw)

    def test_load_setup_validation_and_all_request_clocks_share_owned_lifetime(self):
        raw=report();raw['codec_load_ns']=18*10**9
        with self.assertRaisesRegex(ValueError,'Disjoint load'):validate(raw)
        raw=report()
        for sample in raw['samples']:sample['full_decode_validation_ns']=2*10**9
        with self.assertRaisesRegex(ValueError,'Disjoint load'):validate(raw)

    def test_byte_counters_and_first_text_have_honest_event_scope(self):
        for change in [dict(first_decoded_text_ns=1),dict(first_decoded_text_ns=None),dict(finish_bytes=1)]:
            raw=report();raw['samples'][0].update(change)
            with self.assertRaises(ValueError):validate(raw)
        raw=report();raw['samples'][0]['decisions'][0]['text_bytes']=1
        with self.assertRaises(ValueError):validate(raw)
        expected=dict(EXPECTED,text='')
        self.assertIsNone(validate(report(expected=expected),expected=expected)['measured'][0]['first_decoded_text_ns'])
        expected=dict(EXPECTED,sampled_ids=[130073],text='')
        value=validate(report(expected=expected),expected=expected)
        self.assertIsNone(value['measured'][0]['decode_tokens_per_second'])

    def test_owned_report_bytes_and_duplicate_keys_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'text.json';raw=json.dumps(report()).encode();path.write_bytes(raw)
            owner=dict(owned(),text_report_sha256=hashlib.sha256(raw).hexdigest())
            self.assertEqual(reader.read(path,owner)['format'],'toklet-minicpm5-text-request-v1')
            with self.assertRaises(ValueError):reader.read(path,dict(owner,text_report_sha256='0'*64))
            raw=b'{"sample":0,"sample":1}';path.write_bytes(raw);owner['text_report_sha256']=hashlib.sha256(raw).hexdigest()
            with self.assertRaisesRegex(ValueError,'Duplicate'):reader.read(path,owner)

    def test_first_text_excludes_later_work_and_preserves_finish_only_emission(self):
        raw=report();sample=raw['samples'][0];following=sample['decisions'][1]
        high=following['decision_ns']-sample['chunks'][1]['wall_ns']-following['selection_ns']
        for value in [high+1,following['decision_ns']]:
            broken=copy.deepcopy(raw);broken['samples'][0]['first_decoded_text_ns']=value
            with self.assertRaisesRegex(ValueError,'First decoded text'):validate(broken)
        sample['first_decoded_text_ns']=high
        self.assertEqual(validate(raw)['measured_requests'],10)
        raw=report();sample=raw['samples'][0]
        sample['decisions'][0].update(delta_bytes=0,text_bytes=0)
        sample['decisions'][1].update(delta_bytes=2,text_bytes=2)
        sample['first_decoded_text_ns']=sample['request_ns']-sample['stream_finish_ns']+1
        with self.assertRaisesRegex(ValueError,'First decoded text'):validate(raw)
        sample['first_decoded_text_ns']-=1
        self.assertEqual(validate(raw)['measured_requests'],10)
        sample['decisions'][1].update(delta_bytes=0,text_bytes=0)
        sample.update(finish_bytes=2,first_decoded_text_ns=sample['request_ns']-50)
        self.assertEqual(validate(raw)['measured_requests'],10)
        sample['first_decoded_text_ns']=sample['decisions'][-1]['decision_ns']
        with self.assertRaisesRegex(ValueError,'First decoded text'):validate(raw)


if __name__=='__main__':unittest.main()
