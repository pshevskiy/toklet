"""Portable strict deeper report, partition, ownership and row corruption controls."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import check_qwen35_9b_context256 as check
from test_check_qwen35_9b_context17 import ABI,report as shallow_report


def report():
    r=shallow_report();recovery=r.pop('recovery')
    for k in ['observed_work','arrays','reset_allocations','all_logits_bits_exact','greedy_lowest_id']:r.pop(k)
    r.update(phase='context256',tokens=check.producer.plan()[0],capture_bytes=158707712,capture_arrays=[1051,1051],
        capture_values=[18773504,20903424],diagnostic_allowance_bytes=335544320,diagnostic_requested_bytes=254345296,
        all_logit_bytes=254279680,all_logits_sha256='0'*64,
        wrapper_memory=dict(numerical=copy.deepcopy(r['memory']),pending_history_bytes=2048,wrapper_metadata_bytes=208,admitted_bytes=5538791868),
        validation_ns=1000,repair_ns=1,repair_peak_admitted_bytes=5580214680,
        prefix_tokens=128,prefix_preserved=True,stale_future_nan_masked=True,repair_overlap_minus_one_rejected=True,
        retry_all_logits_bits_exact=True,retry_warmed_rust_allocations=0,reset_exact=True,
        request128plus128_required_capacity=256,request512plus128_rejected=True,request1024plus128_rejected=True,
        request_metrics_qualified=False,instrumentation_overhead_qualified=False,
        allocator_environment=dict(MallocLargeCache='0',MallocNanoZone=None,MallocAggressiveMadvise=None),
        final_work=check.shallow.work(0,'Idle'),
        checkpoints=[dict(inputs=n,snapshot=str(i)*64,greedy_lowest_id=7) for i,n in enumerate([128,256])],
        partitions=[dict(chunk=n,calls=([32]*8 if n==32 else ([17]*7+[9])*2 if n==17 else [1]*256),
            full_logits_bits_exact=True,checkpoint_state_bits_exact=True,warmed_rust_allocations=0,forward_ns=1) for n in [32,17,1]])
    for k in ['failed_work','cancelled_work','poisoned_work','blocked_work','rejected_work']:r[k]=recovery[k]
    return r


def small_rows():
    wanted={k:1 for k in check.oracle.adapter.expected_native().keys()};items=[];raw=bytearray()
    for profile,width,fmt in [('ideal_fp64',8,'<d'),('staged_fp32',4,'<f')]:
        for n in [128,256]:
            for layer,stage,head in wanted:
                block=struct.pack(fmt,.125)
                items.append(dict(checkpoint=n,profile=profile,token=n-1,layer=layer,
                    stage=stage if head is None else f'{stage}.{head}',count=1,offset=len(raw),bytes=width,
                    sha256=hashlib.sha256(block).hexdigest()));raw.extend(block)
    return dict(arrays=items),raw,wanted


def native_receipts(folder):
    (folder/'capture').mkdir();(folder/'capture/rows.f32le').write_bytes(b'')
    (folder/'console.log').write_text('Synthetic receipt fixture; no inference samples\n')
    (folder/'time-and-errors.log').write_text('1 peak memory footprint\n')
    (folder/'process.json').write_text(json.dumps(dict(owned_pid=7,group=7,parent_pid=2)))
    (folder/'inputs.json').write_text('{}\n')
    usage=dict(resident_size=1,phys_footprint=1,lifetime_max_phys_footprint=1)
    (folder/'native-memory.jsonl').write_text(json.dumps(dict(elapsed_seconds=.1,
        members=[dict(pid=7,ppid=2,usage=usage)],collector=usage,conservative_charge_sum_bytes=2))+'\n')
    primary=dict(passed=True,exit_code=0,error=None,cleanup_errors=[],group_absent=True,owned_pid=7,elapsed_seconds=1.,
        lifetime_peak_physical_bytes=1,observed_native_lifetime_peak_bytes=1,observed_conservative_charge_sum_peak_bytes=2,
        native_snapshots=1,native_usage_sha256='0'*64,
        console_sha256=check.sha(folder/'console.log'),time_log_sha256=check.sha(folder/'time-and-errors.log'),
        native_memory_log_sha256=check.sha(folder/'native-memory.jsonl'),group_probe_fallbacks=[],
        live_guard_scope='Conservative sum of native physical charge counters for the owned group plus collector; not a deduplicated whole-machine footprint.')
    final=dict(primary,format='toklet-qwen35-9b-context256-native-v1',cohort=1,binary_sha256='0'*64,changed_inputs=[],
        prelaunch_inputs_sha256='0'*64,report={},postprocess_errors=[])
    return primary,final


def seal_native_receipts(folder,primary,final):
    (folder/'owned-checks.json').write_text(json.dumps(primary)+'\n')
    final['owned_checks_sha256']=check.sha(folder/'owned-checks.json')
    final['files_sha256']={str(p.relative_to(folder)):check.sha(p) for p in folder.rglob('*') if p.is_file() and p.name!='checks.json'}
    (folder/'checks.json').write_text(json.dumps(final)+'\n')


class Controls(unittest.TestCase):
    def reject(self,change):
        r=report();change(r)
        with self.assertRaises(ValueError):check.report_facts(r,ABI,1.)

    def test_exact_source_workload_and_scope(self):
        check.report_facts(report(),ABI,1.)
        for change in [lambda r:r.update(phase='context17'),lambda r:r.update(tokens=[1]),
            lambda r:r.update(position_offset=0),lambda r:r.update(capacity=True),lambda r:r.update(chunk=33),
            lambda r:r.update(source_endpoint_sha256='0'*64),lambda r:r.update(distinct_head=False),
            lambda r:r['config'].update(hidden=2560),lambda r:r.update(quality_qualified=True),
            lambda r:r.update(request_metrics_qualified=True),lambda r:r.update(managed_adoption_qualified=True),
            lambda r:r.update(whole_gpu_graph_qualified=True),lambda r:r.update(extra=True)]:self.reject(change)

    def test_partition_boundaries_exact_outputs_and_state_are_required(self):
        self.assertEqual(check.partition_calls(17),([17]*7+[9])*2)
        self.assertEqual(check.partition_calls(32),[32]*8)
        for bad in [True,0,16,33,None]:
            with self.assertRaises(ValueError):check.partition_calls(bad)
        for change in [lambda r:r['partitions'].pop(),lambda r:r['partitions'][1]['calls'].pop(),
            lambda r:r['partitions'][1]['calls'].__setitem__(7,17),lambda r:r['partitions'][0].update(chunk=True),
            lambda r:r['partitions'][0].update(full_logits_bits_exact=False),lambda r:r['partitions'][1].update(checkpoint_state_bits_exact=False),
            lambda r:r['partitions'][2].update(warmed_rust_allocations=True),lambda r:r['checkpoints'][0].update(inputs=127),
            lambda r:r['checkpoints'][1].update(snapshot='bad'),lambda r:r['checkpoints'][1].update(greedy_lowest_id=True),
            lambda r:r.update(all_logits_sha256=False)]:self.reject(change)

    def test_deep_repair_and_complete_distinct_owner_ledgers_are_required(self):
        for change in [lambda r:r.update(prefix_tokens=5),lambda r:r.update(prefix_preserved=False),
            lambda r:r.update(stale_future_nan_masked=False),lambda r:r['failed_work'].update(tokens_committed=2),
            lambda r:r['poisoned_work'].update(projections_completed=249),lambda r:r.update(retry_all_logits_bits_exact=False),
            lambda r:r.update(retry_warmed_rust_allocations=1),lambda r:r.update(repair_overlap_minus_one_rejected=False),
            lambda r:r.update(repair_peak_admitted_bytes=5580214679),lambda r:r['load_report'].update(retained_cpu_matrix_bytes=1),
            lambda r:r['memory'].update(admitted_bytes=5538789611),lambda r:r['wrapper_memory'].update(pending_history_bytes=0),
            lambda r:r.update(diagnostic_requested_bytes=254345295),lambda r:r.update(all_logit_bytes=1),
            lambda r:r['allocator_environment'].update(MallocNanoZone='0')]:self.reject(change)

    def test_completed_disjoint_and_nested_time_enclosure(self):
        for key in check.TIMES+['repair_ns']:
            for bad in [True,0,-1,1.,10**15]:self.reject(lambda r,k=key,v=bad:r.update({k:v}))
        self.reject(lambda r:r.update(validation_ns=3))
        self.reject(lambda r:r['partitions'][0].update(forward_ns=10**15))
        r=report();r.update({k:150000000 for k in check.TIMES});r['validation_ns']=150000000
        with self.assertRaises(ValueError):check.report_facts(r,ABI,1.)

    def parse(self,result,raw,wanted):
        with patch.object(check.oracle.adapter,'expected_native',return_value=wanted),\
             patch.object(check.producer,'plan',return_value=([],37,{128:1051,256:1051})):
            return check.reference_rows(result,raw)

    def test_complete_typed_reference_rows_and_all_boundaries(self):
        result,raw,wanted=small_rows();rows=self.parse(result,raw,wanted)
        self.assertEqual(set(rows),{(p,n) for p in ['ideal_fp64','staged_fp32'] for n in [128,256]})
        self.assertEqual(sum(len(a) for arrays in rows.values() for a in arrays.values()),4204)
        for change in [lambda r:r['arrays'].pop(),lambda r:r['arrays'][0].update(checkpoint=True),
            lambda r:r['arrays'][1051].update(checkpoint=128),lambda r:r['arrays'][0].update(profile='other'),
            lambda r:r['arrays'][0].update(token=0),lambda r:r['arrays'][0].update(token=True),
            lambda r:r['arrays'][0].update(count=True),lambda r:r['arrays'][0].update(offset=1),
            lambda r:r['arrays'][0].update(bytes=4),lambda r:r['arrays'][0].update(sha256='0'*64),
            lambda r:r['arrays'][0].update(extra=True)]:
            bad=copy.deepcopy(result);change(bad)
            with self.assertRaises(ValueError):self.parse(bad,raw,wanted)
        with self.assertRaises(ValueError):self.parse(result,raw+b'X',wanted)
        bad=copy.deepcopy(result);damaged=bytearray(raw);damaged[:8]=struct.pack('<d',float('nan'))
        bad['arrays'][0]['sha256']=hashlib.sha256(damaged[:8]).hexdigest()
        with self.assertRaises(ValueError):self.parse(bad,damaged,wanted)

    def test_runtime_module_and_native_image_origins_are_closed(self):
        root='/tmp/numpy/__init__.py';core='/tmp/numpy/core.so';lib='/tmp/libpython.dylib';pin=dict(bytes=1,sha256='0'*64)
        runtime=dict(files={root:pin,core:pin});native=dict(files={lib:pin});admission=dict(imports={'numpy':root})
        endpoint=dict(numpy_modules={'numpy':root,'numpy._core._multiarray_umath':core},non_system_native_images={lib:pin},system_native_images=['/usr/lib/libSystem.B.dylib'])
        check.runtime_origins(endpoint,admission,runtime,native)
        for change in [lambda e:e['numpy_modules'].update(numpy='/tmp/unadmitted.py'),
            lambda e:e['numpy_modules'].pop('numpy._core._multiarray_umath'),
            lambda e:e['non_system_native_images'].update({'/tmp/extra.so':pin}),
            lambda e:e.update(system_native_images=['/tmp/not-system.dylib']),lambda e:e.update(extra=True)]:
            bad=copy.deepcopy(endpoint);change(bad)
            with self.assertRaises(ValueError):check.runtime_origins(bad,admission,runtime,native)

    def test_separate_deep_owner_bound_does_not_widen_old_reader(self):
        self.assertEqual(check.oracle.producer.MAX_CAPTURE,256*1024**2)
        self.assertEqual(check.producer.MAX_CAPTURE,512*1024**2)
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);(folder/'capture').mkdir();(folder/'capture/rows.bin').write_bytes(b'')
            (folder/'time-and-errors.log').write_text('1 peak memory footprint\n')
            (folder/'process.json').write_text(json.dumps(dict(owned_pid=7,group=7,parent_pid=2)))
            usage=dict(resident_size=1,phys_footprint=1,lifetime_max_phys_footprint=1)
            (folder/'native-memory.jsonl').write_text(json.dumps(dict(elapsed_seconds=.1,members=[dict(pid=7,ppid=2,usage=usage)],collector=usage,conservative_charge_sum_bytes=2))+'\n')
            files={'capture/rows.bin','time-and-errors.log','process.json','native-memory.jsonl'}
            receipt=dict(format='toklet-qwen35-9b-context256-reference-owned-v1',passed=True,group_absent=True,error=None,
                cleanup_errors=[],changed_inputs=[],exit_code=0,elapsed_seconds=1.,lifetime_peak_physical_bytes=1,
                observed_native_lifetime_peak_bytes=1,observed_conservative_charge_sum_peak_bytes=2,owned_pid=7,native_snapshots=1,
                files_sha256={p:hashlib.sha256((folder/p).read_bytes()).hexdigest() for p in files})
            (folder/'checks.json').write_text(json.dumps(receipt))
            with patch.object(check,'read',wraps=check.read) as read:
                check.successful_reference_owned(folder,files)
                self.assertTrue(any(c.args==(folder/'capture/rows.bin',512*1024**2) for c in read.call_args_list))
            for key,bad in [('passed',False),('exit_code',False),('group_absent',False),('lifetime_peak_physical_bytes',True),('changed_inputs',['changed'])]:
                altered=dict(receipt,**{key:bad});(folder/'checks.json').write_text(json.dumps(altered))
                with self.assertRaises(ValueError):check.successful_reference_owned(folder,files)

    def test_native_primary_outcome_cannot_be_rewritten_by_postprocessing(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);primary,final=native_receipts(folder)
            seal_native_receipts(folder,primary,final);check.successful_native_owned(folder)
            for mutate in [lambda p,f:p.update(elapsed_seconds=.5),
                lambda p,f:p.update(extra=True),lambda p,f:f.update(extra=True),
                lambda p,f:f.update(postprocess_errors=['late parse error']),
                lambda p,f:f.update(postprocess_errors=False),lambda p,f:p.update(passed=False)]:
                p,f=copy.deepcopy(primary),copy.deepcopy(final);mutate(p,f);seal_native_receipts(folder,p,f)
                with self.assertRaises(ValueError):check.successful_native_owned(folder)
            # The raw envelope may be resealed, but its primary guard digest must
            # still describe the exact console that the owner observed.
            (folder/'console.log').write_text('Rewritten console\n');seal_native_receipts(folder,primary,final)
            with self.assertRaisesRegex(ValueError,'Primary/raw deeper native digest differs: console.log'):
                check.successful_native_owned(folder)
            primary['console_sha256']=check.sha(folder/'console.log');final['console_sha256']=primary['console_sha256']
            primary['elapsed_seconds']=final['elapsed_seconds']=check.NATIVE_DEADLINE_SECONDS+120
            seal_native_receipts(folder,primary,final)
            with self.assertRaisesRegex(ValueError,'Bounded deeper native owned lifetime required'):
                check.successful_native_owned(folder)

    def test_native_group_and_peak_observations_require_typed_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);primary,final=native_receipts(folder)
            for bad in [None,True,[dict(group=True,process_table_present=False)],
                [dict(group=8,process_table_present=False)],[dict(group=7,process_table_present=0)],
                [dict(group=7,process_table_present=False,assumed_absent=True)]]:
                p,f=copy.deepcopy(primary),copy.deepcopy(final)
                p['group_probe_fallbacks']=f['group_probe_fallbacks']=bad;seal_native_receipts(folder,p,f)
                with self.assertRaisesRegex(ValueError,'Typed deeper native group-probe observations required'):
                    check.successful_native_owned(folder)
            p,f=copy.deepcopy(primary),copy.deepcopy(final)
            p['group_probe_fallbacks']=f['group_probe_fallbacks']=[dict(group=7,process_table_present=False)]
            seal_native_receipts(folder,p,f);check.successful_native_owned(folder)
            for bad in [True,1.]:
                p,f=copy.deepcopy(primary),copy.deepcopy(final)
                p['observed_native_lifetime_peak_bytes']=f['observed_native_lifetime_peak_bytes']=bad
                seal_native_receipts(folder,p,f)
                with self.assertRaisesRegex(ValueError,'Strict bounded deeper native observed lifetime peak required'):
                    check.successful_native_owned(folder)


if __name__=='__main__':unittest.main()
