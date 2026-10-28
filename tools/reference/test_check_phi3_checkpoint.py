"""CPU-only malformed capture and unchanged numerical-gate controls."""
import copy
import struct
import unittest
from unittest import mock
import check_phi3_checkpoint as check

class Checker(unittest.TestCase):
    def test_nonoverlapping_complete_rows_required(self):
        items=[];cursor=0
        for(name,start),shape in check.inventory().items():
            size=shape[0]*shape[1]*4
            items.append(dict(name=name,row_start=start,shape=shape,event='completed',dtype='<f4',offset=cursor,bytes=size,sha256='0'*64));cursor+=size
        self.assertEqual(len(items),1288)
        self.assertEqual(check.validate_inventory(items),cursor)
        bad=copy.deepcopy(items);bad[-1]=bad[-2].copy()
        with self.assertRaises(ValueError):check.validate_inventory(bad)
        bad=copy.deepcopy(items);bad[0]['row_start']=1
        with self.assertRaises(ValueError):check.validate_inventory(bad)
        bad=copy.deepcopy(items);bad[0]['shape'][1]+=1
        with self.assertRaises(ValueError):check.validate_inventory(bad)
        for key,value in [('row_start',False),('row_start',0.0),('shape',[1.0,2048])]:
            bad=copy.deepcopy(items);bad[0][key]=value
            with self.assertRaises(ValueError):check.validate_inventory(bad)
        with self.assertRaises(ValueError):check.validate_inventory(items[:-1])
    def test_gate_requires_finite_values_and_keeps_exact_failure_count(self):
        target=struct.pack('<3f',1,2,3);gate=dict(atol=.01,rtol=0.)
        result=check.qualify([1,2.005,3.02],struct.iter_unpack('<f',target),gate)
        self.assertEqual((result['values'],result['failed']),(3,1));self.assertGreater(result['max_gate_fraction'],1)
        with self.assertRaises(ValueError):check.qualify([float('nan')],[(0.,)],gate)
        with self.assertRaises(ValueError):check.qualify([0.],[(float('inf'),)],gate)
        with self.assertRaises(ValueError):check.qualify([0.,1.],[(0.,)],gate)
    def test_schedule_causal_boundary_and_selection_disagreements_reject(self):
        with mock.patch.object(check,'VOCAB',2):
            logits={(case,'chunk'+str(chunk)):[1.,2.]*len(spec['tokens'])
                    for case,spec in check.reference.CASES.items()for chunk in check.CHUNKS}
            boundary={f'boundary.chunk{chunk}.logits':[1.,2.]for chunk in [32,128]}
            initial={f'{group}.step_0.logits':[1.,2.]for group in ['greedy','native_generated']}
            selected={f'{group}.step_{step}.logits':1 for group in ['greedy','native_generated']for step in range(4)}
            report=dict(native_selected_on_teacher_forced_prefixes=[1]*4,native_generated_ids=[1]*4)
            check.exact_logits_controls(logits,boundary,initial,selected,report)
            bad=copy.deepcopy(logits);bad[('short','chunk3')][-1]=3.
            with self.assertRaisesRegex(ValueError,'schedules'):check.exact_logits_controls(bad,boundary,initial,selected,report)
            bad=copy.deepcopy(logits)
            for chunk in check.CHUNKS:bad[('future_changed','chunk'+str(chunk))][0]=3.
            with self.assertRaisesRegex(ValueError,'causal'):check.exact_logits_controls(bad,boundary,initial,selected,report)
            bad=copy.deepcopy(boundary);bad['boundary.chunk128.logits'][0]=3.
            with self.assertRaisesRegex(ValueError,'boundary'):check.exact_logits_controls(logits,bad,initial,selected,report)
            bad=copy.deepcopy(initial);bad['native_generated.step_0.logits'][0]=3.
            with self.assertRaisesRegex(ValueError,'Initial'):check.exact_logits_controls(logits,boundary,bad,selected,report)
            bad=copy.deepcopy(report);bad['native_generated_ids'][0]=0
            with self.assertRaisesRegex(ValueError,'maximum'):check.exact_logits_controls(logits,boundary,initial,selected,bad)
    def test_ownership_and_generation_reports_cannot_upgrade_scope(self):
        owners=dict(source_manifest_input_bytes=1024**2,parsed_reference_reserve_bytes=8*1024**2,
                    descriptor_tables_reserve_bytes=8*1024**2,numeric_witnesses_bytes=8*check.VOCAB*4,
                    history_reserve_bytes=64*1024,io_buffers_bytes=3*8192)
        ledger=dict(**owners,full_tensor_serialization_bytes=0,maximum_descriptors=1288,
                    snapshot_and_cpu_readback_already_in_session=True,total_extra_owner_reserve_bytes=sum(owners.values()),
                    reserved_bytes=32*1024**2,allocator_and_process_peak_measured_separately=True)
        ids=[1,2,3,4]
        controls=[dict(readback=mode,generated_ids=ids,output_limit=True,pre_cancel_preserved_state=True,
                       callback_failure_valid_prefix=True,forced_stop_id=1,natural_model_eos_tested=False,
                       text_chat_qualified=False)for mode in ['FullLogits','GreedyToken']]
        boundary=[dict(chunk=chunk,append_error='Err(Options("token ID/count outside session capacity"))',
                       set_error='Err(Options("token ID/count outside session capacity"))',append_state_equal=True,
                       set_state_equal=True,evaluated=512,pending=0)for chunk in [32,128]]
        report=dict(extra_owner_ledger=ledger,generation_controls=controls,native_generated_ids=ids,boundary_controls=boundary)
        weights=2149645198
        report.update(weights_bytes=weights,positional_bytes=0,exact_load_budget=2216819598,
                      one_byte_under_load_exact_budget_error=True,
                      resident_owner_ledger=dict(queried_half_norm_row_bytes=6144,packed_resident_bytes=2149244928,
                                                 norm_count=65,guard_bytes_per_norm=14,resident_weights_bytes=weights))
        memory=dict(weights_bytes=weights,kv_bytes=1000,scratch_bytes=1000,history_bytes=1000,
                    logits_bytes=1000,metadata_bytes=1000,metadata_allowance_bytes=64*1024**2)
        memory['admitted_bytes']=sum(memory.values())
        plans=[];budgets=[]
        for chunk in [32,128]:
            for mode in ['FullLogits','GreedyToken']:
                plans.append(dict(source_payload_bytes=2149644288,source_admitted_load_bytes=2216818688,
                                  resident_weights_bytes=weights,resident_admitted_load_bytes=2216819598,
                                  requested_capacity=512,requested_chunk_size=chunk,requested_position_offset=0,
                                  readback=mode,projection='MlxAffine',normalization='Compensated',attention='Reference',
                                  fits_source_load_budget=True,fits_resident_load_budget=True,fits_session_budget=True,
                                  fits_requested_budgets=True,payload_validation_required=True,numerical_buffers_allocated=False,
                                  qualification_evaluated=False,session_memory=memory.copy()))
                budgets.append(dict(chunk=chunk,readback=mode,required=memory['admitted_bytes'],
                                    exact_budget_constructed=True,one_byte_under_exact_budget_error=True))
        report.update(resource_plans=plans,session_budget_controls=budgets)
        check.validate_extra_controls(report)
        for key,value in [('snapshot_and_cpu_readback_already_in_session',False),('full_tensor_serialization_bytes',100),
                          ('descriptor_tables_reserve_bytes',0),('reserved_bytes',64*1024**2)]:
            bad=copy.deepcopy(report);bad['extra_owner_ledger'][key]=value
            with self.assertRaises(ValueError):check.validate_extra_controls(bad)
        for key,value in [('pre_cancel_preserved_state',False),('text_chat_qualified',True),('natural_model_eos_tested',True),
                          ('readback','other'),('forced_stop_id',0),('generated_ids',[1,2,3,5])]:
            bad=copy.deepcopy(report);bad['generation_controls'][1][key]=value
            with self.assertRaises(ValueError):check.validate_extra_controls(bad)
        for key,value in [('source_payload_bytes',0),('resident_weights_bytes',weights+1),('readback','other'),
                          ('payload_validation_required',False),('numerical_buffers_allocated',True),('fits_session_budget',False)]:
            bad=copy.deepcopy(report);bad['resource_plans'][2][key]=value
            with self.assertRaises(ValueError):check.validate_extra_controls(bad)
        bad=copy.deepcopy(report);bad['session_budget_controls'][0]['one_byte_under_exact_budget_error']=False
        with self.assertRaises(ValueError):check.validate_extra_controls(bad)
        bad=copy.deepcopy(report);bad['resource_plans'][0]['session_memory']['scratch_bytes']+=1
        with self.assertRaises(ValueError):check.validate_extra_controls(bad)
        for key,value in [('append_state_equal',False),('append_error','Err(Device)'),('evaluated',511),('pending',1)]:
            bad=copy.deepcopy(report);bad['boundary_controls'][0][key]=value
            with self.assertRaises(ValueError):check.validate_extra_controls(bad)
        bad=copy.deepcopy(report);bad['native_generated_ids']=[check.VOCAB]*4
        with self.assertRaises(ValueError):check.validate_extra_controls(bad)
if __name__=='__main__':unittest.main()
