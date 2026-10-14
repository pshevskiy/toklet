"""Frozen v2 paired metadata/pin/admission hooks without a GPU."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock,patch
import metal_prefill_v2 as v2
import test_metal_prefill as fixtures


def fixture(shape=512):
    e,events=fixtures.fixture();e.update(projection_modes=['Q4Compensated','Q4CompensatedTiled'],normalization_mode='Compensated',requires_logit_bits=True,hidden_size=1536,num_layers=28,
        weight_dtype='Q4 v2 + FP16 exclusions',packed_format_version=2,group_size=8,quantizer='toklet-q4-affine-f32-source-half-metadata-rne-g8-v2')
    for key in ('original_memory','tiled_memory'):
        e[key]['weights_bytes']=v2.loss.SPECS['larger']['weights'];e[key]['admitted_bytes']=sum(v for k,v in e[key].items() if k!='admitted_bytes')
    e['prompt_tokens']=shape;e['cached_prefix_tokens']=512 if shape==64 else 0
    e['combined_admitted_bytes']=e['original_memory']['admitted_bytes']+e['tiled_memory']['admitted_bytes']-e['original_memory']['weights_bytes']+e['application_bytes']
    for event in events:
        event.update(original_memory=e['original_memory'],tiled_memory=e['tiled_memory'],logit_bits_equal=True)
        for mode in ('original','tiled'):
            event[mode+'_report'].update(reused_tokens=e['cached_prefix_tokens'],prefill_tokens=shape,history_tokens=e['cached_prefix_tokens']+shape+1,evaluated_tokens=e['cached_prefix_tokens']+shape)
    return e,events


class V2Prefill(unittest.TestCase):
    def test_exact_v2_metadata_residency_pair_bits_and_13samples(self):
        e,events=fixture();v2.ready(e,True,True)
        for i,event in enumerate(events):v2.shared.sample(event,i,e)
        self.assertEqual(len(v2.shared.analyze(events)['tiled']['raw_seconds']),10)
        for key,value in [('weight_dtype','Q4 v1 + FP16 exclusions'),('packed_format_version',2.0),('group_size',64),('quantizer','v1'),('q4_execution_widths',[32,64])]:
            changed=copy.deepcopy(e);changed[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):v2.ready(changed,True,True)
        changed=copy.deepcopy(events[0]);changed['logit_bits_equal']=False
        with self.assertRaisesRegex(ValueError,'logit bits'):v2.shared.sample(changed,0,e)

    def test_all_shapes_keep_exact_reused_prefill_history_and_pending_semantics(self):
        for shape in (64,128,512):
            e,events=fixture(shape);v2.ready(e,True,True,shape=shape)
            for i,event in enumerate(events):v2.shared.sample(event,i,e)
            for key,value in [('cached_prefix_tokens',0 if shape==64 else 512),('prompt_tokens',17)]:
                bad=copy.deepcopy(e);bad[key]=value
                with self.subTest(shape=shape,key=key),self.assertRaises(ValueError):v2.ready(bad,True,True,shape=shape)
            bad=copy.deepcopy(events[0]);bad['original_report']['reused_tokens']=0 if shape==64 else 512
            with self.assertRaisesRegex(ValueError,'semantics'):v2.shared.sample(bad,0,e)

    def test_shape_wrappers_pin_every_consumed_array_and_exact_arguments(self):
        geometry={'passed':True,'review':{'accepted':True},'bindings':{'artifacts/m5-larger-q4-v2/manifest.json':{'sha256':v2.loss.SPECS['larger']['packed']}}}
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)
            for shape,file in ((64,'append_64.u32le'),(128,'prompt_128.u32le'),(512,'prompt_512.u32le')):
                with patch.object(v2.loss.shared,'frozen_bytes',return_value=json.dumps(geometry).encode()),patch.object(v2.shared,'run',return_value='authored') as run:
                    self.assertEqual(v2.run(p/'binary',p/'out.json',shape),'authored')
                kwargs=run.call_args.kwargs;kwargs['admission_check'](fixture(shape)[0],True,True)
                self.assertIn(v2.ROOT/'benchmarks/contract-v1'/file,kwargs['artifact_pins'])
                if shape==64:self.assertIn(v2.ROOT/'benchmarks/contract-v1/prompt_512.u32le',kwargs['artifact_pins'])
                self.assertEqual(kwargs['extra_args'],() if shape==512 else (f'shape{shape}',))
            for invalid in (True,0,256,1024):
                with patch.object(v2.shared,'run') as run,self.assertRaisesRegex(ValueError,'Frozen prefill shape'):v2.run(p/'binary',p/'out.json',invalid)
                run.assert_not_called()

    def test_wrapper_requires_review_and_forwards_complete_explicit_pins(self):
        geometry={'passed':True,'review':{'accepted':True},'bindings':{'artifacts/m5-larger-q4-v2/manifest.json':{'sha256':v2.loss.SPECS['larger']['packed']}}}
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)
            with patch.object(v2.loss.shared,'frozen_bytes',return_value=json.dumps(geometry).encode()),patch.object(v2.shared,'run',return_value='authored') as run:
                self.assertEqual(v2.run(p/'binary',p/'out.json'),'authored')
            kwargs=run.call_args.kwargs;kwargs['admission_check'](fixture()[0],True,True)
            self.assertEqual(kwargs['artifact_pins'][v2.ROOT/v2.loss.GEOMETRY_REL],v2.loss.GEOMETRY_SHA)
            self.assertIn(Path(v2.__file__).resolve(),kwargs['extra_identity_paths'])
            geometry['review']['accepted']=False
            with patch.object(v2.loss.shared,'frozen_bytes',return_value=json.dumps(geometry).encode()),patch.object(v2.shared,'run') as run,self.assertRaisesRegex(ValueError,'Reviewed'):v2.run(p/'binary',p/'out.json')
            run.assert_not_called()

    def test_explicit_pin_replacement_rejects_before_child(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);packed=root/'packed';(root/'src').mkdir();geometry=root/'geometry.json';binary=root/'binary'
            pins={packed/'manifest.json':'packed',geometry:'geometry'}
            for replaced in pins:
                def digest(p):
                    if p==replaced:return 'replacement'
                    if p in pins:return pins[p]
                    if p.name=='model.safetensors':return v2.shared.LARGER_SOURCE_SHA
                    if p.name=='config.json':return v2.shared.LARGER_CONFIG_SHA
                    return 'fixture'
                with patch.object(v2.shared,'ROOT',root),patch.object(v2.shared,'digest',side_effect=digest),patch.object(v2.shared.subprocess,'Popen') as launch:
                    with self.subTest(replaced=replaced),self.assertRaisesRegex(ValueError,'Frozen explicit'):v2.shared.run(binary,root/'out.json',True,True,artifact_path=packed,artifact_pins=pins,admission_check=v2.ready,extra_identity_paths=(geometry,))
                launch.assert_not_called()

    def test_explicit_artifact_without_pins_rejects_before_child(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(v2.shared.subprocess,'Popen') as launch:
            with self.assertRaisesRegex(ValueError,'needs frozen pins'):v2.shared.run(Path('binary'),Path(directory)/'out.json',True,True,artifact_path=Path('packed'))
        launch.assert_not_called()

    def test_custom_v2_ready_failure_preserves_error_and_reaps_owned_worker(self):
        real=subprocess.Popen;children=[]
        def launch(command,**kwargs):
            child=real([sys.executable,'-c','print(\'{"event":"bad"}\',flush=True);import time;time.sleep(60)'],**kwargs);child.wait=Mock(wraps=child.wait);children.append(child);return child
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'src').mkdir();packed=root/'packed';pins={packed/'manifest.json':'packed'}
            def digest(p):
                if p in pins:return pins[p]
                if p.name=='model.safetensors':return v2.shared.LARGER_SOURCE_SHA
                if p.name=='config.json':return v2.shared.LARGER_CONFIG_SHA
                return 'fixture'
            with patch.object(v2.shared,'ROOT',root),patch.object(v2.shared,'digest',side_effect=digest),patch.object(v2.shared,'observation',return_value='fixture'),patch.object(v2.shared,'endpoints',return_value={}),patch.object(v2.shared.subprocess,'check_output',return_value='fixture'),patch.object(v2.shared.subprocess,'Popen',launch):
                with self.assertRaisesRegex(ValueError,'admission/mode'):v2.shared.run(root/'binary',root/'out.json',True,True,artifact_path=packed,artifact_pins=pins,admission_check=v2.ready)
            record=json.loads((root/'out.json').read_text());self.assertFalse(record['passed']);self.assertEqual(record['cleanup_errors'],[])
            self.assertIn('prefill/append',record['scope']);self.assertNotIn('512-input',record['scope'])
        children[0].wait.assert_called_once_with(timeout=5);self.assertIsNotNone(children[0].returncode)

if __name__=='__main__':unittest.main()
