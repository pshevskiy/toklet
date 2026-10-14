"""Frozen v2 application cold hooks, exact pins and retained failures; no GPU."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock,patch
import metal_application_v2 as v2
import test_metal_application as legacy
import test_metal_tasks_v2 as task_fixtures
import test_task_grade_admission_v2 as inventory_fixtures


class V2Application(unittest.TestCase):
    def test_actual_v2_done_modes_stop_pending_and_reset(self):
        for eos in (False,True):
            prompt,events,ready=legacy.Application().fixture(eos);ready=task_fixtures.admission()
            for event in events:
                if event['event']=='done':event['statistics'].update(copy.deepcopy({k:v for k,v in ready.items() if k not in ('event','context_capacity')}))
            result=v2.shared.collect_response(prompt,2,events,ready)
            self.assertEqual(result['statistics']['execution']['quantization_group_size'],8)
            next(e['statistics'] for e in events if e['event']=='done')['execution']['packed_format_version']=1
            with self.assertRaises(ValueError):v2.shared.collect_response(prompt,2,events,ready)

    def test_reviewed_full_quality_required_before_application(self):
        q={'passed':True,'quality_completed':True,'model':'larger','review':{'accepted':True},'cases':100,'q4_passed':11,'fp32_passed':11,'drop_percentage_points':0}
        with patch.object(v2.producer.loss.shared,'frozen_bytes',return_value=json.dumps(q).encode()),patch.object(v2.producer,'loss_prerequisite') as loss:v2.quality_prerequisite();loss.assert_called_once()
        for mutate in (lambda q:q.update(passed=False),lambda q:q.update(cases=99),lambda q:q.update(model='small'),lambda q:q.update(q4_passed=10),lambda q:q['review'].update(accepted=False)):
            changed=copy.deepcopy(q);mutate(changed)
            with patch.object(v2.producer.loss.shared,'frozen_bytes',return_value=json.dumps(changed).encode()),patch.object(v2.producer,'loss_prerequisite') as loss,self.assertRaisesRegex(ValueError,'quality required'):v2.quality_prerequisite()
            loss.assert_not_called()

    def test_custom_admission_failure_uses_exact_v2_args_and_never_decodes_live_model(self):
        real=subprocess.Popen;children=[];commands=[]
        def launch(command,**kw):
            commands.append(command);child=real([sys.executable,'-c','print(\'{"event":"wrong"}\',flush=True);import time;time.sleep(60)'],**kw);child.wait=Mock(wraps=child.wait);children.append(child);return child
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)
            with (p/'raw').open('x') as raw,(p/'stderr').open('xb') as stderr,patch.object(v2.shared.subprocess,'Popen',launch),patch.object(v2.shared.tasks,'finalized') as decode:
                result=v2.shared.child(p/'binary',p,'larger',[],[],raw,stderr,True,admission_check=v2.producer.ready,projection_argument=v2.producer.cli.ARG,extra_args=('--q4-weights','/packed'))
                decode.assert_not_called()
        self.assertFalse(result['passed']);self.assertFalse(result['model_exited']);self.assertEqual(result['cleanup_errors'],[])
        self.assertEqual(commands[0][commands[0].index('--metal-projection')+1],v2.producer.cli.ARG);self.assertEqual(commands[0][-2:],['--q4-weights','/packed'])
        children[0].wait.assert_called_once_with(timeout=5);self.assertIsNotNone(children[0].returncode)

    def invoke(self,path,replace_initial=None,replace_endpoint=None):
        from contextlib import ExitStack
        record,binary,q,manifest=inventory_fixtures.V2CollectionProvenance().fixture();ids=record['identities']
        ids.update({str(v2.ROOT/v2.QUALITY_REL):v2.QUALITY_SHA,str(v2.shared.controlled.prepare.CONTRACT/'application_trace.json'):manifest['authored_text_sha256']['application_trace.json']})
        after=False
        def digest(p):
            if p==replace_initial or after and p==replace_endpoint:return 'replacement'
            return ids.get(str(p),'c'*64)
        def child(*args,**kwargs):
            nonlocal after
            after=True
            return {'passed':True,'model_exited':True,'samples':[{'scenario_seconds':1.0}],'cleanup_errors':[]}
        with ExitStack() as stack:
            stack.enter_context(patch.object(v2,'quality_prerequisite'))
            stack.enter_context(patch.object(v2.producer,'qualified_cli',return_value=(binary,q)))
            stack.enter_context(patch.object(v2.shared.admission,'frozen_manifest',return_value=manifest))
            stack.enter_context(patch.object(v2.shared,'trace_inputs',return_value=([],[])))
            stack.enter_context(patch.object(v2.shared.metal_engine,'digest',side_effect=digest))
            stack.enter_context(patch.object(v2.shared.controlled.metal_prefill,'observation',side_effect=lambda a:'17179869184' if a[-1]=='hw.memsize' else 'fixture'))
            worker=stack.enter_context(patch.object(v2.shared,'child',side_effect=child))
            try:result=v2.run(path,True)
            except ValueError:
                if replace_initial:worker.assert_not_called()
                raise
            return result,worker

    def test_consumed_manifest_and_event_parser_are_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            result,worker=self.invoke(Path(directory)/'out.json')
            for module in (v2.shared.admission,v2.shared.admission.evidence):self.assertIn(str(Path(module.__file__).resolve()),result['identities'])
            worker.assert_called_once();self.assertIs(worker.call_args.kwargs['admission_check'],v2.producer.ready)

    def test_parser_endpoint_mutations_retain_evidence_and_reject(self):
        for module in (v2.shared.admission,v2.shared.admission.evidence):
            with tempfile.TemporaryDirectory() as directory:
                p=Path(directory)/'out.json'
                with self.subTest(module=module.__name__),self.assertRaisesRegex(ValueError,'evidence retained'):self.invoke(p,replace_endpoint=Path(module.__file__).resolve())
                result=json.loads(p.read_text());self.assertFalse(result['passed']);self.assertFalse(result['identities_unchanged']);self.assertTrue(result['run']['passed'])

    def test_compiled_binary_source_quality_receipt_and_artifact_pins_reasserted_before_launch(self):
        for p in (v2.ROOT/v2.producer.BINARY_REL,v2.ROOT/'src/metal/mod.rs',v2.ROOT/v2.producer.QUAL_REL,
                  v2.ROOT/v2.QUALITY_REL,v2.ROOT/'artifacts/m5-larger-q4-v2/manifest.json'):
            with tempfile.TemporaryDirectory() as directory,self.subTest(path=p),self.assertRaisesRegex(ValueError,'Initial frozen'):self.invoke(Path(directory)/'out.json',replace_initial=p)

    def test_exclusive_destination_rejects_before_any_qualification(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'out.json';p.write_text('immutable')
            with patch.object(v2,'quality_prerequisite') as q,patch.object(v2.shared,'child') as worker,self.assertRaisesRegex(ValueError,'Exclusive'):v2.run(p)
            q.assert_not_called();worker.assert_not_called()

if __name__=='__main__':unittest.main()
