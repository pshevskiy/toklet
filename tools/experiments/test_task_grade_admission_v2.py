"""Frozen full v2 admission and isolated grading hooks; no model or VM."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import task_grade_admission_v2 as admission
import task_grading_window_v2 as window
import test_task_grade_admission as legacy_fixtures
import test_metal_tasks_v2 as v2_fixtures
import test_task_grading_window as window_fixtures


class FullV2Admission(unittest.TestCase):
    def fixture(self):
        record,cases,raw=legacy_fixtures.GradingAdmission().fixture()
        ready=v2_fixtures.admission();record['run']['ready']=ready;raw[0]=ready
        record.update(format='toklet-q4-v2-task-collection-v1',model='larger',projection_mode='Q4CompensatedTiled',
                      normalization_mode='Compensated',weight_dtype='Q4 v2 + FP16 exclusions',packed_format_version=2,group_size=8)
        for case,item in zip(cases,record['run']['cases'],strict=True):
            for event in item['native_events']:
                if event['event']=='done':event['statistics'].update({k:v for k,v in ready.items() if k not in ('event','context_capacity')})
            item['response']=admission.producer.shared.collect_response(case,item['native_events'],ready)
        record['run']['idle_usage']=[dict(phase=phase,worker={'lifetime_max_phys_footprint':1},driver={'lifetime_max_phys_footprint':2},joint_native_lifetime_peak_bytes=3)
                                   for phase in ['ready']+[f'case_{i}' for i in range(1,101)]]
        return record,cases,raw

    def test_complete_frozen_caps_raw_and_v2_identity(self):
        record,cases,raw=self.fixture();result=admission.validate_records(record,cases,raw)
        self.assertEqual(len(result['responses']),100);self.assertEqual(result['identity']['group_size'],8)

    def test_flags_caps_exit_metadata_and_raw_are_required(self):
        for mutate in (lambda p:p.update(check_only=True),lambda p:p.update(collection_completed=False),
                       lambda p:p.update(quality_completed=True),lambda p:p.update(packed_format_version=2.0),
                       lambda p:p.update(model='small'),lambda p:p['task_ids'].reverse(),
                       lambda p:p['run'].update(worker_pid=True),lambda p:p['run'].update(model_exited=False),
                       lambda p:p['run'].update(exit_code=True),lambda p:p['run']['cases'][0].update(max_output_tokens=8)):
            record,cases,raw=self.fixture();mutate(record)
            with self.assertRaises(ValueError):admission.validate_records(record,cases,raw)
        record,cases,raw=self.fixture()
        with self.assertRaisesRegex(ValueError,'raw/idle'):admission.validate_records(record,cases,raw[:-1])

    def test_missing_duplicate_and_reordered_idle_labels_reject(self):
        for mutate in (lambda u:u.pop(),lambda u:u[1].update(phase='ready'),lambda u:u.reverse(),lambda u:u[50].update(phase='case_51')):
            record,cases,raw=self.fixture();mutate(record['run']['idle_usage'])
            with self.assertRaisesRegex(ValueError,'raw/idle'):admission.validate_records(record,cases,raw)

    def test_native_charge_types_totals_and_limit_are_required(self):
        for mutate in (lambda u:u.update(joint_native_lifetime_peak_bytes=True),lambda u:u['worker'].update(lifetime_max_phys_footprint=0),
                       lambda u:u.update(joint_native_lifetime_peak_bytes=4),lambda u:u['driver'].update(lifetime_max_phys_footprint=admission.producer.shared.LIMIT)):
            record,cases,raw=self.fixture();mutate(record['run']['idle_usage'][0])
            with self.assertRaisesRegex(ValueError,'native charge'):admission.validate_records(record,cases,raw)

    def test_idle_proof_requires_exact_completed_collection_before_any_vm(self):
        with tempfile.TemporaryDirectory() as directory:
            selected=Path(directory).resolve()/'collection.json';proof=Path(directory)/'idle.json'
            proof.write_text(json.dumps({'format':'toklet-q4-v2-task-idle-v1','collections':[str(selected)]}))
            accepted=dict(collection_path=str(selected),collection_sha256='summary',raw_sha256='raw',identity={'model':'larger'},worker_pid=1234567)
            with patch.object(admission,'validate_collection',return_value=accepted) as validate:
                result=admission.admit_idle(proof,selected);validate.assert_called_once_with(selected)
                self.assertEqual(result['collections'][0]['worker_pid'],1234567)
            proof.write_text(json.dumps({'format':'toklet-q4-v2-task-idle-v1','collections':[str(selected)]*2}))
            with patch.object(admission,'validate_collection') as validate,self.assertRaisesRegex(ValueError,'Exactly'):
                admission.admit_idle(proof,selected)
            validate.assert_not_called()
            proof.write_text(json.dumps({'format':'toklet-q4-v2-task-idle-v1','collections':[str(selected)]}))
            with patch.object(admission,'validate_collection',side_effect=ValueError('Incomplete')):
                with self.assertRaisesRegex(ValueError,'Incomplete'):admission.admit_idle(proof,selected)

    def test_wrapper_passes_only_cold_validators_and_explicit_provenance(self):
        args=SimpleNamespace()
        with patch.object(window.shared,'run',return_value=0) as run:
            self.assertEqual(window.run(args),0)
        self.assertIs(run.call_args.kwargs['collection_admission'],admission)
        self.assertIs(run.call_args.kwargs['idle_admission'],admission.admit_idle)
        self.assertIn(Path(admission.__file__).resolve(),run.call_args.kwargs['extra_source_paths'])
        self.assertIn(admission,run.call_args.kwargs['extra_grading_modules'])

class V2CollectionProvenance(unittest.TestCase):
    def fixture(self):
        producer=admission.producer;root=admission.ROOT;binary=root/producer.BINARY_REL
        source=root/'models'/producer.loss.SPECS['larger']['name'];manifest=admission.legacy.frozen_manifest()
        q={'bindings':{producer.BINARY_REL:{'sha256':'a'*64},
                      'artifacts/m5-q4-v2-cli-source-v1/src/metal/mod.rs':{'sha256':'b'*64},
                      'artifacts/m5-larger-q4-v2/manifest.json':{'sha256':producer.loss.SPECS['larger']['packed']}}}
        ids={p:'c'*64 for p in admission.inventory(binary,q,source,manifest)}
        controlled=producer.shared.controlled;spec=controlled.MODELS['larger']
        pins={source/'model.safetensors':spec[1],source/'config.json':spec[2],
              controlled.prepare.CONTRACT/'manifest.json':controlled.CONTRACT_SHA,
              root/'benchmarks/result.schema.json':controlled.SCHEMA_SHA,
              root/'benchmarks/sources.json':manifest['sources_sha256'],
              root/'tools/bench/requirements.lock':manifest['python_dependencies_sha256']}
        pins.update({controlled.prepare.CONTRACT/v['file']:v['sha256'] for v in manifest['arrays'].values()})
        pins.update(producer.compiled_pins(q));pins.update({root/producer.QUAL_REL:producer.QUAL_SHA,root/producer.LOSS_REL:producer.LOSS_SHA,
                    root/manifest['quality']['tasks_file']:producer.shared.tasks.TASK_SHA})
        pins.update({source/name:sha for name,sha in producer.cli.shared.ASSET_PINS.items()})
        pins.update({root/p:v['sha256'] for p,v in q['bindings'].items() if p.startswith('artifacts/m5-larger-q4-v2/')})
        ids.update({str(p):sha for p,sha in pins.items()})
        command=[str(binary),'--model',str(source),'--backend','metal','--metal-projection',producer.cli.ARG,
                 '--interactive','--json','--context-capacity','2048','--prefill-chunk','128','--q4-weights',str(root/'artifacts/m5-larger-q4-v2')]
        return {'run':{'command':command},'identities':ids},binary,q,manifest

    def invoke(self,record,binary,q,manifest,current=None):
        from contextlib import ExitStack
        encoded=json.dumps(record).encode();ids=record['identities'];current=ids if current is None else current
        with ExitStack() as stack:
            stack.enter_context(patch.object(admission.legacy,'bounded',side_effect=[encoded,b'']))
            stack.enter_context(patch.object(admission,'validate_records',return_value={'identity':{},'responses':[]}))
            stack.enter_context(patch.object(admission.producer.shared.tasks,'load_cases',return_value=[]))
            stack.enter_context(patch.object(admission.producer,'qualified_cli',return_value=(binary,q)))
            stack.enter_context(patch.object(admission.producer,'loss_prerequisite'))
            stack.enter_context(patch.object(admission.legacy,'frozen_manifest',return_value=manifest))
            stack.enter_context(patch.object(admission.producer.shared.metal_engine,'digest',side_effect=lambda p:current[str(p)]))
            decode=stack.enter_context(patch.object(admission.producer.shared,'finalized'))
            try: result=admission.validate_collection(admission.ROOT/'authored-collection.json')
            except ValueError:decode.assert_not_called();raise
            decode.assert_called_once();return result

    def test_exact_inventory_and_all_pinned_inputs_accept(self):
        record,binary,q,manifest=self.fixture();result=self.invoke(record,binary,q,manifest)
        self.assertEqual(result['collection_path'],str(admission.ROOT/'authored-collection.json'))

    def test_missing_or_extra_inventory_rejects_before_decode(self):
        record,binary,q,manifest=self.fixture()
        for entry in record['identities']:
            changed=copy.deepcopy(record);del changed['identities'][entry]
            with self.subTest(entry=entry),self.assertRaisesRegex(ValueError,'inventory'):self.invoke(changed,binary,q,manifest)
        changed=copy.deepcopy(record);changed['identities'][str(admission.ROOT/'extra.py')]='c'*64
        with self.assertRaisesRegex(ValueError,'inventory'):self.invoke(changed,binary,q,manifest)

    def test_saved_and_current_binary_source_receipts_and_artifact_replacement_reject(self):
        record,binary,q,manifest=self.fixture()
        for entry in (binary,admission.ROOT/'src/metal/mod.rs',admission.ROOT/admission.producer.QUAL_REL,
                      admission.ROOT/admission.producer.LOSS_REL,admission.ROOT/'artifacts/m5-larger-q4-v2/manifest.json'):
            saved=copy.deepcopy(record);saved['identities'][str(entry)]='d'*64
            with self.subTest(saved=entry),self.assertRaisesRegex(ValueError,'frozen'):self.invoke(saved,binary,q,manifest)
            current={**record['identities'],str(entry):'d'*64}
            with self.subTest(current=entry),self.assertRaisesRegex(ValueError,'endpoint'):self.invoke(record,binary,q,manifest,current)

class InjectedWindow(unittest.TestCase):
    def test_actual_shared_window_custom_admission_failure_never_starts_vm(self):
        result,record,commands,owner,start,_,grade=window_fixtures.WindowRestoration().invoke(custom_hooks=True,admission_failure=True)
        self.assertEqual(result,1);self.assertIn('V2 admission failed',record['failure']['message'])
        start.assert_not_called();grade.assert_not_called();owner.finish.assert_not_called()

    def test_actual_shared_window_custom_late_live_check_prevents_vm(self):
        result,record,commands,owner,start,_,grade=window_fixtures.WindowRestoration().invoke(custom_hooks=True,late_live=True)
        self.assertEqual(result,1);self.assertIn('became live',record['failure']['message'])
        start.assert_not_called();grade.assert_not_called();owner.finish.assert_not_called()

    def test_actual_shared_window_passes_selected_validator_provenance_and_restores(self):
        result,record,commands,owner,start,_,grade=window_fixtures.WindowRestoration().invoke(custom_hooks=True)
        self.assertEqual(result,0);self.assertTrue(record['restored_stopped'])
        kwargs=grade.call_args.kwargs
        self.assertIs(kwargs['collection_validator'],kwargs['extra_identity_modules'][0].validate_collection)
        self.assertTrue(any('held_out_task_window.py' in key for key in record['window_source_sha256']))
        start.assert_called_once();grade.assert_called_once();owner.finish.assert_called_once()

if __name__=='__main__':unittest.main()
