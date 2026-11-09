"""Portable strict context17 metrics and recovery report controls."""
import copy
import unittest
import check_qwen35_9b_context17 as check

ABI={'nine_b':dict(model=200,layer=232,session=1344,delta_state=144,kv=48,projection=240,gather=160,metal_store=2168,gpu_weights=112)}


def report():
    return dict(phase='context17',source_sha256=check.oracle.adapter.pin.FILE_SHA,
        config_sha256=check.oracle.adapter.pin.CONFIG_SHA,header_sha256=check.oracle.adapter.pin.HEADER_SHA,
        source_bytes=5038160953,source_endpoint_sha256=check.oracle.adapter.pin.FILE_SHA,distinct_head=True,
        tokens=list(check.TOKENS),position_offset=37,capacity=512,chunk=32,config=copy.deepcopy(check.oracle.adapter.GEOMETRY),
        load_report=copy.deepcopy(check.UPLOAD),model_owned_bytes=7624,model_learned_bytes=5174465592,
        memory=copy.deepcopy(check.MEMORY),execution_owned_bytes=41425068,layout=copy.deepcopy(ABI['nine_b']),
        observed_work=check.work(17),capture_bytes=67705856,arrays=1051,reset_allocations=[0,0,0],
        all_logits_bits_exact=True,warmed_rust_allocations=0,admission_minus_one_rejected=True,
        whole_gpu_graph_qualified=False,independent_fidelity_qualified=False,quality_qualified=False,
        managed_adoption_qualified=False,greedy_lowest_id=7,**{k:1 for k in check.TIMES},
        partitions=[dict(chunk=n,snapshot='0'*64,all_logits_bits_exact=True,warmed_rust_allocations=0,forward_ns=1) for n in [17,1,3]],
        recovery=dict(prefix_tokens=5,prefix_preserved=True,stale_nonfinite_kv_masked=True,
            failed_work=dict(outcome='Error',tokens_completed=2,tokens_committed=0,projections_completed=502,
                gathers_completed=3,convolution_steps_completed=49,recurrence_steps_completed=49,attention_heads_completed=256),
            cancelled_work=dict(outcome='Cancelled',tokens_completed=0,tokens_committed=0,projections_completed=1,
                gathers_completed=1,convolution_steps_completed=0,recurrence_steps_completed=0,attention_heads_completed=0),
            poisoned_work=dict(outcome='Error',tokens_completed=0,tokens_committed=0,projections_completed=248,
                gathers_completed=1,convolution_steps_completed=24,recurrence_steps_completed=24,attention_heads_completed=128),
            blocked_work=check.work(0,'Error'),retry_work=check.work(12),rejected_work=check.work(0,'Error'),
            repair_overlap_minus_one_rejected=True,repair_peak_admitted_bytes=5580214680,
            retry_warmed_rust_allocations=0,retry_all_logits_bits_exact=True,repair_ns=1))


class Controls(unittest.TestCase):
    def reject(self,change):
        value=report();change(value)
        with self.assertRaises(ValueError):check.report_facts(value,ABI,1.)

    def test_source_workload_geometry_and_scope_are_strict(self):
        check.report_facts(report(),ABI,1.)
        for change in [lambda r:r.update(phase='one'),lambda r:r.update(tokens=[1]),
            lambda r:r.update(position_offset=0),lambda r:r.update(capacity=True),
            lambda r:r.update(distinct_head=False),lambda r:r.update(source_endpoint_sha256='0'*64),
            lambda r:r['config'].update(hidden=2560),lambda r:r.update(managed_adoption_qualified=True),
            lambda r:r.update(whole_gpu_graph_qualified=True),lambda r:r.update(full_model_qualified=True),
            lambda r:r.update(arrays=True),lambda r:r['observed_work'].update(tokens_committed=True)]:self.reject(change)

    def test_each_partition_must_have_equal_state_and_exact_outputs(self):
        for change in [lambda r:r['partitions'].pop(),lambda r:r['partitions'][0].update(chunk=32),
            lambda r:r['partitions'][1].update(snapshot='1'*64),lambda r:r['partitions'][0].update(snapshot='bad'),
            lambda r:r['partitions'][0].update(warmed_rust_allocations=True),
            lambda r:r['partitions'][2].update(warmed_rust_allocations=1),
            lambda r:r['partitions'][2].update(all_logits_bits_exact=False),lambda r:r.update(reset_allocations=[0,True,0])]:self.reject(change)

    def test_nonzero_prefix_repair_retry_and_exact_owner_ledgers_are_required(self):
        for change in [lambda r:r['recovery'].update(prefix_tokens=1),lambda r:r['recovery'].update(prefix_preserved=False),
            lambda r:r['recovery'].update(stale_nonfinite_kv_masked=False),lambda r:r['recovery']['failed_work'].update(tokens_committed=2),
            lambda r:r['recovery']['poisoned_work'].update(projections_completed=249),
            lambda r:r['recovery']['retry_work'].update(tokens_committed=11),lambda r:r['recovery'].update(retry_all_logits_bits_exact=False),
            lambda r:r['recovery'].update(retry_warmed_rust_allocations=1),lambda r:r['recovery'].update(repair_overlap_minus_one_rejected=False),
            lambda r:r['recovery'].update(repair_peak_admitted_bytes=5580214679),lambda r:r['load_report'].update(retained_cpu_matrix_bytes=1),
            lambda r:r['memory'].update(admitted_bytes=5538789611),lambda r:r['memory'].update(scratch_bytes=42003115),
            lambda r:r.update(execution_owned_bytes=41425067),lambda r:r['layout'].update(model=184)]:self.reject(change)

    def test_disjoint_completed_phase_times_are_inside_owned_lifetime(self):
        for name in check.TIMES:
            for value in [True,0,-1,1.,10**15]:self.reject(lambda r,n=name,v=value:r.update({n:v}))
        self.reject(lambda r:r['partitions'][0].update(forward_ns=10**15))
        self.reject(lambda r:r['recovery'].update(repair_ns=10**15))
        value=report();value.update({k:100000000 for k in check.TIMES})
        for row in value['partitions']:row['forward_ns']=100000000
        value['recovery']['repair_ns']=100000000
        with self.assertRaises(ValueError):check.report_facts(value,ABI,1.)

    def test_reference_keys_use_last_committed_input_and_original_gate(self):
        row=dict(token=16,layer=3,stage='attention_k_cache.0')
        self.assertEqual(check.oracle.common.key(row,token=16),(3,'attention_k_cache',0))
        for token in [0,True,15,17]:
            with self.assertRaises(ValueError):check.oracle.common.key(dict(row,token=token),token=16)
        self.assertEqual(check.oracle.graph.GATE,dict(atol=1e-5,rtol=1e-4))
        self.assertEqual(sum(check.oracle.adapter.expected_native(17).values()),check.VALUES)
        self.assertEqual(check.work(17)['projections_completed'],4233)


if __name__=='__main__':unittest.main()
