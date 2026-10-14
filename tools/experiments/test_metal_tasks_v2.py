"""Larger v2 admission, frozen caps, provenance and cleanup; no GPU/model."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock,patch

import metal_tasks_v2 as v2
import test_metal_tasks as fixtures
from test_task_cases import inventory


def admission():
    event=fixtures.MetalTasks().admission('larger')
    event['execution']=v2.cli.execution(128)
    event['logical_memory']['weights_bytes']=v2.loss.SPECS['larger']['weights']
    event['memory_admitted_bytes']=sum(event['logical_memory'].values())
    return event


class V2Tasks(unittest.TestCase):
    def test_only_loss_qualified_larger_v2_modes_and_residency_are_admitted(self):
        event=admission();v2.ready(event,'larger')
        with self.assertRaisesRegex(ValueError,'preselected'):v2.ready(event,'small')
        for key,value in [('weight_dtype','FP16'),('projection_mode','HalfCompensated'),('normalization_mode','Original'),
                          ('packed_format_version',1),('quantization_group_size',64),('prefill_chunk_size',True)]:
            bad=copy.deepcopy(event);bad['execution'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):v2.ready(bad,'larger')
        bad=copy.deepcopy(event);bad['logical_memory']['weights_bytes']=3087433340;bad['memory_admitted_bytes']=sum(bad['logical_memory'].values())
        with self.assertRaisesRegex(ValueError,'residency'):v2.ready(bad,'larger')

    def test_full_cases_keep_original_caps_and_pending_stop_state(self):
        for eos in (False,True):
            case,events=fixtures.MetalTasks().fixture(eos=eos)
            for e in events:
                if e['event']=='done':e['statistics'].update({k:v for k,v in admission().items() if k not in ('event','context_capacity')})
            out=v2.shared.collect_response(case,events,admission())
            self.assertEqual(out['statistics']['execution'],v2.cli.execution(128))
        case,events=fixtures.MetalTasks().fixture(cap=8)
        for e in events:
            if e['event']=='done':e['statistics'].update({k:v for k,v in admission().items() if k not in ('event','context_capacity')})
        v2.shared.collect_response(case,events,admission(),8)
        with self.assertRaises(ValueError):v2.shared.collect_response(case,events,admission())

    def test_cli_qualification_requires_review_exact_binary_and_snapshot(self):
        value={'format':'toklet-q4-v2-cli-qualification-v1','passed':True,'review':{'accepted':True},'native_runs':8,
               'raw_event_replay_count':109,'bindings':{v2.BINARY_REL:{'sha256':'sig'},'artifacts/m5-q4-v2-cli-source-v1/src/cpu.rs':{'sha256':'sig'}}}
        with patch.object(v2.loss.shared,'frozen_bytes',return_value=json.dumps(value).encode()),patch.object(v2.shared.metal_engine,'digest',return_value='sig'):
            self.assertEqual(v2.qualified_cli()[0],v2.ROOT/v2.BINARY_REL)
        for key,bad in [('passed',False),('review',{'accepted':False}),('native_runs',7),('raw_event_replay_count',108)]:
            changed=copy.deepcopy(value);changed[key]=bad
            with self.subTest(key=key),patch.object(v2.loss.shared,'frozen_bytes',return_value=json.dumps(changed).encode()),patch.object(v2.shared.metal_engine,'digest',return_value='sig'),self.assertRaises(ValueError):v2.qualified_cli()
        def digest(path):return 'different' if str(path).endswith('/src/cpu.rs') else 'sig'
        with patch.object(v2.loss.shared,'frozen_bytes',return_value=json.dumps(value).encode()),patch.object(v2.shared.metal_engine,'digest',side_effect=digest),self.assertRaisesRegex(ValueError,'snapshot'):v2.qualified_cli()

    def test_full_loss_pass_is_required_before_task_collection(self):
        value={'passed':True,'review':{'accepted':True},'models':[{'model':'larger','loss_gate_passed':True,'scored_targets':10000}]}
        with patch.object(v2.loss.shared,'frozen_bytes',return_value=json.dumps(value).encode()):v2.loss_prerequisite()
        for change in ({'loss_gate_passed':False},{'scored_targets':9999},{'model':'small'}):
            bad=copy.deepcopy(value);bad['models'][0].update(change)
            with self.subTest(change=change),patch.object(v2.loss.shared,'frozen_bytes',return_value=json.dumps(bad).encode()),self.assertRaises(ValueError):v2.loss_prerequisite()

    def test_binary_or_source_replaced_after_admission_never_launches(self):
        binary=v2.ROOT/v2.BINARY_REL;source=v2.ROOT/'models'/v2.shared.controlled.MODELS['larger'][0]
        rust=v2.ROOT/'src/metal/mod.rs';packed=v2.ROOT/'artifacts/m5-larger-q4-v2/manifest.json'
        q={'bindings':{v2.BINARY_REL:{'sha256':'admitted-binary'},
                       'artifacts/m5-q4-v2-cli-source-v1/src/metal/mod.rs':{'sha256':'admitted-rust'},
                       'artifacts/m5-larger-q4-v2/manifest.json':{'sha256':v2.loss.SPECS['larger']['packed']}}}
        original=v2.shared.metal_engine.digest
        for replaced in (binary,rust):
            def digest(path):
                if path==replaced:return 'replacement'
                if path==binary:return 'admitted-binary'
                if path==rust:return 'admitted-rust'
                if path==packed:return v2.loss.SPECS['larger']['packed']
                if path==v2.ROOT/v2.QUAL_REL:return v2.QUAL_SHA
                if path==v2.ROOT/v2.LOSS_REL:return v2.LOSS_SHA
                if path==source/'model.safetensors':return v2.loss.SPECS['larger']['source']
                if path==source/'config.json':return v2.loss.SPECS['larger']['config']
                if path.parent==source:return v2.cli.shared.ASSET_PINS[path.name]
                return original(path)
            with tempfile.TemporaryDirectory() as directory,patch.object(v2,'qualified_cli',return_value=(binary,q)),patch.object(v2,'loss_prerequisite'),patch.object(v2.shared.metal_engine,'digest',side_effect=digest),patch.object(v2.shared,'child') as child:
                with self.subTest(replaced=replaced),self.assertRaisesRegex(ValueError,'Initial frozen'):
                    v2.run(Path(directory)/'report.json',True)
                child.assert_not_called()

    def test_custom_v2_admission_failure_never_decodes_live_owned_model(self):
        real=subprocess.Popen;children=[];commands=[]
        def launch(command,**kwargs):
            commands.append(command);child=real([sys.executable,'-c','print(\'{"event":"wrong"}\',flush=True);import time;time.sleep(60)'],**kwargs)
            child.wait=Mock(wraps=child.wait);children.append(child);return child
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)
            with (p/'raw').open('x') as raw,(p/'stderr').open('xb') as stderr,patch.object(v2.shared.subprocess,'Popen',launch),patch.object(v2.shared,'finalized') as decode:
                out=v2.shared.child(p/'binary',p,'larger',inventory()[0],raw,stderr,True,admission_check=v2.ready,
                                    projection_argument=v2.cli.ARG,extra_args=('--q4-weights','/packed'))
                decode.assert_not_called()
        self.assertFalse(out['passed']);self.assertFalse(out['model_exited']);self.assertEqual(out['cleanup_errors'],[])
        self.assertIsNotNone(children[0].returncode);children[0].wait.assert_called_once_with(timeout=5)
        self.assertEqual(commands[0][commands[0].index('--metal-projection')+1],v2.cli.ARG)
        self.assertEqual(commands[0][-2:],['--q4-weights','/packed'])

    def test_existing_destination_rejects_before_qualification_or_workers(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'out.json';p.write_text('immutable')
            with patch.object(v2,'qualified_cli') as cli,patch.object(v2.shared,'child') as child:
                with self.assertRaisesRegex(ValueError,'evidence exists'):v2.run(p)
                cli.assert_not_called();child.assert_not_called()
            self.assertEqual(p.read_text(),'immutable')


if __name__=='__main__':unittest.main()
