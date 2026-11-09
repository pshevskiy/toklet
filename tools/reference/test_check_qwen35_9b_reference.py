"""Portable report/lifetime/closure controls, independent of cached real weights."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import check_qwen35_9b_reference as check
import qwen35_9b_reference as producer


def report():
    return dict(source_sha256=check.adapter.pin.FILE_SHA,source_endpoint_sha256=check.adapter.pin.FILE_SHA,
        container_bytes=5038160953,config_sha256=check.adapter.pin.CONFIG_SHA,header_sha256=check.adapter.pin.HEADER_SHA,
        tokens=[1],position_offset=0,capacity=512,chunk=32,memory=copy.deepcopy(check.MEMORY),work=copy.deepcopy(check.WORK),
        capture_bytes=66640896,arrays=1051,distinct_head=True,warmed_rust_allocations=0,
        late_head_failure_work=dict(check.WORK,outcome='Error',tokens_completed=0,tokens_committed=0),
        late_head_failure_prefix_preserved=True,cancelled_prefix_preserved=True,retry_bits_equal=True,
        retry_warmed_rust_allocations=0,first_token_executed=True,independent_real_source_fidelity_qualified=False,
        quality_qualified=False,managed_adoption_qualified=False,greedy_lowest_id=7,
        load_ns=1,bind_ns=1,observed_forward_ns=1,plain_forward_ns=1,source_endpoint_ns=1,
        timing_scope='CPU raw-input capture/replay and late-head failure/cancel/reset; no controlled performance or independent fidelity claim')


class Controls(unittest.TestCase):
    def test_native_report_has_strict_source_scope_memory_work_and_zero_allocation(self):
        original=report();check.native_report(original,1.)
        changes=[lambda n:n.update(tokens=[True]),lambda n:n.update(arrays=True),
            lambda n:n.update(container_bytes=2367237149),lambda n:n.update(distinct_head=False),
            lambda n:n.update(managed_adoption_qualified=True),lambda n:n.update(quality_qualified=True),
            lambda n:n.update(full_model_qualified=True),lambda n:n.update(warmed_rust_allocations=1),
            lambda n:n['memory'].update(admitted_bytes=1),lambda n:n['work'].update(tokens_committed=True),
            lambda n:n['late_head_failure_work'].update(projections_completed=True),
            lambda n:n['late_head_failure_work'].update(tokens_committed=1)]
        for change in changes:
            broken=copy.deepcopy(original);change(broken)
            with self.assertRaises(ValueError):check.native_report(broken,1.)

    def test_phase_times_must_be_positive_integers_inside_owned_duration(self):
        original=report()
        for value in [True,-1,0,1.,float('nan'),10**20]:
            broken=copy.deepcopy(original);broken['observed_forward_ns']=value
            with self.assertRaises(ValueError):check.native_report(broken,1.)
        for elapsed in [True,0,-1,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):check.native_report(original,elapsed)
        broken=copy.deepcopy(original)
        for key in ['load_ns','bind_ns','observed_forward_ns','plain_forward_ns','source_endpoint_ns']:broken[key]=300_000_000
        with self.assertRaises(ValueError):check.native_report(broken,1.)

    def test_live_native_samples_fold_to_receipt_and_bind_owned_process(self):
        usage=dict(resident_size=90,phys_footprint=70,lifetime_max_phys_footprint=80)
        row=dict(elapsed_seconds=.5,members=[dict(pid=123,ppid=12,usage=usage)],collector=usage,conservative_charge_sum_bytes=140)
        receipt=dict(native_snapshots=1,owned_pid=123,elapsed_seconds=1.,observed_conservative_charge_sum_peak_bytes=140,observed_native_lifetime_peak_bytes=80)
        raw=lambda r:json.dumps(r).encode()+b'\n'
        check.live_memory(raw(row),receipt)
        for change in [lambda r:r.update(elapsed_seconds=2.),lambda r:r.update(elapsed_seconds=True),
            lambda r:r.update(conservative_charge_sum_bytes=139),lambda r:r['members'][0].update(pid=124),
            lambda r:r['collector'].update(phys_footprint=True)]:
            broken=copy.deepcopy(row);change(broken)
            with self.assertRaises(ValueError):check.live_memory(raw(broken),receipt)
        for key in ['native_snapshots','observed_conservative_charge_sum_peak_bytes','observed_native_lifetime_peak_bytes']:
            broken=dict(receipt);broken[key]+=1
            with self.assertRaises(ValueError):check.live_memory(raw(row),broken)

    def test_duplicate_nonfinite_json_and_noncanonical_head_names_reject(self):
        for raw in [b'{"a":1,"a":2}',b'{"a":NaN}',b'{"a":Infinity}']:
            with self.assertRaises(ValueError):check.common.decode(raw)
        for row in [dict(token=True,layer=0,stage='delta_state'),dict(token=0,layer=True,stage='delta_state'),
                    dict(token=0,layer=3,stage='attention_k_cache.01')]:
            with self.assertRaises(ValueError):check.common.key(row)

    def test_reference_plan_and_source_closure_stay_one_input_and_bounded(self):
        self.assertEqual(producer.plan('one'),([1],0,16660224))
        self.assertEqual(producer.plan('context17'),(list(producer.graph.TOKENS),37,16926464))
        with self.assertRaises(ValueError):producer.plan('context18')
        self.assertEqual(sum(check.adapter.expected_native().values())*12,199922688)
        self.assertLess(199922688,producer.MAX_CAPTURE)
        self.assertLess(sum(producer.LEDGER.values()),6*1024**3)
        names={p.name for p in producer.extras()}
        self.assertEqual(names,{'qwen35_9b_weights.py','qwen35_real_weights.py','affine4_large_rows.py',
            'qwen35_cpu_graph.py','bf16_source.py','check_qwen35_9b_payload.py','check_qwen35_payload.py','qwen35_9b_reference.py'})

    def test_collection_failure_closes_authenticated_owner_and_keeps_failed_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);model=root/'model';model.mkdir();raw=b'not model weights'
            (model/'model.safetensors').write_bytes(raw)
            with patch.object(producer.pin,'FILE_BYTES',len(raw)),patch.object(producer.pin,'FILE_SHA',hashlib.sha256(raw).hexdigest()), \
                 patch.object(producer.runtime_owner.runtime,'captured',return_value=b''):
                result=producer.collect(root/'capture',model,{},lambda:{'checked':True})
            self.assertFalse(result['completed']);self.assertEqual(result['primary_error']['message'],'Pinned9B config differs')
            self.assertTrue(result['source']['closed']);self.assertFalse(result['source']['poisoned'])
            self.assertEqual(result['endpoint_errors'],[]);self.assertGreater(result['collection_ns'],0)
            self.assertEqual(json.loads((root/'capture/result.json').read_text())['completed'],False)


if __name__=='__main__':unittest.main()
