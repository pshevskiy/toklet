"""Portable hybrid-report/accounting controls; no model download or GPU."""
import copy
import unittest
import check_qwen35_9b_metal as check

ABI={'nine_b':dict(model=200,layer=232,session=1344,delta_state=144,kv=48,projection=240,gather=160,metal_store=2168,gpu_weights=112)}


def report():
    upload=dict(matrices=250,packed_gpu_bytes=check.PACKED_BYTES+100000,small_cpu_capacity_bytes=check.SMALL_BYTES,
        retained_requested_bytes=check.PACKED_BYTES+100000+check.SMALL_BYTES+2168,
        retained_policy_bytes=check.PACKED_BYTES+100000+check.SMALL_BYTES+2168+check.POLICY,
        admitted_load_policy_bytes=5800000000,maximum_upload_source_capacity_bytes=37748808,retained_cpu_matrix_bytes=0)
    execution=50*1024**2;memory=copy.deepcopy(check.oracle.MEMORY)
    memory.update(weights_and_model_bytes=upload['retained_policy_bytes']+7624,scratch_bytes=578048+execution,metadata_bytes=5184)
    memory['admitted_bytes']=sum(v for k,v in memory.items() if k!='admitted_bytes')
    failed=dict(check.oracle.WORK,outcome='Error',tokens_completed=0,tokens_committed=0)
    value=dict(source_sha256=check.oracle.adapter.pin.FILE_SHA,source_endpoint_sha256=check.oracle.adapter.pin.FILE_SHA,
        container_bytes=5038160953,config_sha256=check.oracle.adapter.pin.CONFIG_SHA,header_sha256=check.oracle.adapter.pin.HEADER_SHA,
        tokens=[1],position_offset=0,capacity=512,chunk=32,phase='one',cpu_tensor_capacity_bytes=5040162816,
        layout=copy.deepcopy(ABI['nine_b']),distinct_head=True,arrays=1051,capture_bytes=66640896,work=copy.deepcopy(check.oracle.WORK),
        warmed_rust_allocations=0,retry_warmed_rust_allocations=0,late_head_failure_prefix_preserved=True,
        cancelled_prefix_preserved=True,retry_bits_equal=True,poisoned_prefix_preserved=True,
        repair_overlap_minus_one_rejected=True,first_token_executed=True,independent_real_source_fidelity_qualified=False,
        whole_gpu_graph_qualified=False,contextual_numerical_qualified=False,quality_qualified=False,managed_adoption_qualified=False,
        timing_scope=check.SCOPE,memory=memory,upload_report=upload,embedding_slot=1,head_slot=0,
        execution_owned_bytes=execution,late_head_failure_work=failed,poisoned_work=dict(failed,projections_completed=248),
        repair_peak_admitted_bytes=memory['admitted_bytes']+execution,greedy_lowest_id=7)
    value.update({name:1 for name in check.TIMES});return value


class Controls(unittest.TestCase):
    def test_complete_hybrid_source_shape_work_and_scope_are_strict(self):
        value=report();check.report_facts(value,ABI,1.)
        for change in [lambda r:r.update(tokens=[True]),lambda r:r.update(capacity=True),
            lambda r:r.update(source_sha256='0'*64),lambda r:r.update(distinct_head=False),
            lambda r:r.update(embedding_slot=0),lambda r:r.update(head_slot=True),
            lambda r:r.update(whole_gpu_graph_qualified=True),lambda r:r.update(contextual_numerical_qualified=True),
            lambda r:r.update(managed_adoption_qualified=True),lambda r:r.update(full_model_qualified=True),
            lambda r:r['work'].update(tokens_committed=True)]:
            broken=copy.deepcopy(value);change(broken)
            with self.assertRaises(ValueError):check.report_facts(broken,ABI,1.)

    def test_consumed_original_gpu_ownership_and_overlaps_are_required(self):
        value=report()
        for change in [lambda r:r['upload_report'].update(matrices=249),lambda r:r['upload_report'].update(retained_cpu_matrix_bytes=1),
            lambda r:r['upload_report'].update(packed_gpu_bytes=check.PACKED_BYTES-1),
            lambda r:r['upload_report'].update(retained_policy_bytes=r['upload_report']['retained_policy_bytes']-1),
            lambda r:r['upload_report'].update(admitted_load_policy_bytes=check.LIMIT),
            lambda r:r['upload_report'].update(maximum_upload_source_capacity_bytes=37748807),
            lambda r:r['memory'].update(weights_and_model_bytes=r['memory']['weights_and_model_bytes']-1),
            lambda r:r['memory'].update(scratch_bytes=r['memory']['scratch_bytes']-1),
            lambda r:r.update(repair_peak_admitted_bytes=r['repair_peak_admitted_bytes']-1)]:
            broken=copy.deepcopy(value);change(broken)
            with self.assertRaises(ValueError):check.report_facts(broken,ABI,1.)

    def test_late_head_poison_repair_and_zero_allocations_are_explicit(self):
        value=report()
        for change in [lambda r:r['poisoned_work'].update(projections_completed=249),
            lambda r:r['late_head_failure_work'].update(projections_completed=248),
            lambda r:r['poisoned_work'].update(tokens_committed=1),lambda r:r.update(poisoned_prefix_preserved=False),
            lambda r:r.update(repair_overlap_minus_one_rejected=False),lambda r:r.update(retry_bits_equal=False),
            lambda r:r.update(warmed_rust_allocations=1),lambda r:r.update(retry_warmed_rust_allocations=1)]:
            broken=copy.deepcopy(value);change(broken)
            with self.assertRaises(ValueError):check.report_facts(broken,ABI,1.)

    def test_positive_phase_times_enclose_completed_owned_work(self):
        value=report()
        for name in check.TIMES:
            for number in [True,0,-1,1.,10**20]:
                broken=copy.deepcopy(value);broken[name]=number
                with self.assertRaises(ValueError):check.report_facts(broken,ABI,1.)
        broken=copy.deepcopy(value);broken.update({name:200000000 for name in check.TIMES})
        with self.assertRaises(ValueError):check.report_facts(broken,ABI,1.)

    def test_original_dual_profile_geometry_remains_unchanged(self):
        self.assertEqual(check.oracle.graph.GATE,dict(atol=1e-5,rtol=1e-4))
        self.assertEqual(sum(check.oracle.adapter.expected_native().values()),16660224)
        self.assertEqual(len(check.oracle.adapter.expected_native()),1051)
        self.assertEqual(check.PACKED_BYTES+check.SMALL_BYTES,check.CPU_TENSOR_BYTES)


if __name__=='__main__':unittest.main()
