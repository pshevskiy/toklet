"""V2 controlled cold hooks, representation and owned failure cleanup; no GPU."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock,patch
import metal_controlled_v2 as v2
import test_metal_controlled as fixtures


def admission():
    e=fixtures.Controlled().admission('larger');e.update(projection_mode=v2.MODE,normalization_mode='Compensated',
        weight_dtype='Q4 v2 + FP16 exclusions',packed_format_version=2,group_size=8,
        quantizer='toklet-q4-affine-f32-source-half-metadata-rne-g8-v2',q4_compensated_execution_widths=[32,32])
    e['memory']['weights_bytes']=v2.loss.SPECS['larger']['weights'];e['memory']['admitted_bytes']=sum(v for k,v in e['memory'].items() if k!='admitted_bytes');e['memory_admitted_bytes']=e['memory']['admitted_bytes']+e['application_bytes'];return e


class V2Controlled(unittest.TestCase):
    def test_exact_representation_geometry_budget_and_widths(self):
        e=admission();v2.ready(e,'larger')
        with self.assertRaises(ValueError):v2.ready(e,'small')
        for key,value in [('projection_mode','HalfCompensated'),('weight_dtype','FP16'),('normalization_mode','Original'),('packed_format_version',2.0),('group_size',64),('quantizer','v1'),('q4_compensated_execution_widths',[32,64]),('memory_admitted_bytes',v2.shared.LIMIT+1)]:
            changed=copy.deepcopy(e);changed[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):v2.ready(changed,'larger')

    def test_result_records_v2_without_relabeling(self):
        legacy=fixtures.Controlled();legacy.setUpClass();response=legacy.fixture('cached');response['memory']=admission()['memory']
        setup,req,cache=v2.shared.requests('cached',legacy.arrays,legacy.manifest)
        run={'workload':'cached','ready':admission(),'samples':[{'response':response,'process_peak_footprint_bytes':1}]*13,'idle_usage':[{'worker':{'lifetime_max_phys_footprint':1}}],'cache_state':cache}
        obs={'compiler':'rustc','device':{'name':'MacBookPro','chip':'M1 Pro','physical_memory_bytes':16*1024**3,'os_version':'26','os_build':'test','power_mode':'AC','thermal_conditions':None}}
        representation={'weight_dtype':'Q4','kv_dtype':'FP16','packed_format_version':2,'group_size':8}
        result=v2.shared.result(run,'larger',legacy.manifest,obs,'0'*40,True,representation=representation,runtime='Toklet Q4 v2',execution_note='V2 actual execution')
        self.assertEqual(result['representation'],representation);self.assertEqual(result['runtime'],'Toklet Q4 v2')
        self.assertEqual(len(result['samples']),10)

    def test_custom_admission_failure_keeps_exact_command_and_reaps_owned_worker(self):
        real=subprocess.Popen;owned=[];commands=[];legacy=fixtures.Controlled();legacy.setUpClass()
        def launch(command,**kw):
            commands.append(command);p=real([sys.executable,'-c','print(\'{"event":"wrong"}\',flush=True);import time;time.sleep(60)'],**kw);p.wait=Mock(wraps=p.wait);owned.append(p);return p
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with (root/'raw').open('x') as raw,(root/'stderr').open('xb') as stderr,patch.object(v2.shared.subprocess,'Popen',launch):
                result=v2.shared.worker(root/'binary',root,'larger','short',legacy.manifest,legacy.arrays,raw,stderr,True,admission_check=v2.ready,extra_args=('/packed','q4-v2-tiled'))
        self.assertFalse(result['passed']);self.assertIn('v2 controlled mode',result['error']);self.assertEqual(result['cleanup_errors'],[])
        self.assertEqual(commands[0][-2:],['/packed','q4-v2-tiled']);owned[0].wait.assert_called_once_with(timeout=5);self.assertIsNotNone(owned[0].returncode)

    def test_exclusive_destination_rejects_before_qualification_or_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'out.json';p.write_text('immutable')
            with patch.object(v2.shared,'contract') as contract,patch.object(v2.shared,'worker') as worker,self.assertRaisesRegex(ValueError,'Exclusive'):v2.run(p/'binary',p,True)
            contract.assert_not_called();worker.assert_not_called();self.assertEqual(p.read_text(),'immutable')

    def test_saved_artifact_or_geometry_pin_replacement_rejects_before_worker(self):
        legacy=fixtures.Controlled();legacy.setUpClass();source=v2.ROOT/'models'/v2.shared.MODELS['larger'][0]
        packed=v2.ROOT/'artifacts/m5-larger-q4-v2/manifest.json';geometry_path=v2.ROOT/v2.loss.GEOMETRY_REL
        geometry={'passed':True,'review':{'accepted':True},'bindings':{'artifacts/m5-larger-q4-v2/manifest.json':{'sha256':v2.loss.SPECS['larger']['packed']}}}
        original=v2.shared.metal_engine.digest
        for replaced in (packed,geometry_path):
            def digest(p):
                if p==replaced:return 'replacement'
                if p==packed:return v2.loss.SPECS['larger']['packed']
                if p==geometry_path:return v2.loss.GEOMETRY_SHA
                if p==source/'model.safetensors':return v2.shared.MODELS['larger'][1]
                if p==source/'config.json':return v2.shared.MODELS['larger'][2]
                if p==source/'tokenizer.json':return 'c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539'
                return original(p)
            with tempfile.TemporaryDirectory() as directory:
                binary=Path(directory)/'binary';binary.write_text('fixture')
                with patch.object(v2.loss.shared,'frozen_bytes',return_value=json.dumps(geometry).encode()),patch.object(v2.shared.metal_engine,'digest',side_effect=digest),patch.object(v2.shared,'worker') as worker:
                    with self.subTest(replaced=replaced),self.assertRaisesRegex(ValueError,'Initial v2'):v2.run(binary,Path(directory)/'out.json',True)
                worker.assert_not_called()

class SelectedWorkloads(unittest.TestCase):
    def test_invalid_empty_duplicate_or_unknown_selection_rejects_before_inputs(self):
        for workloads in ([],['short','short'],['unknown'],['short','unknown']):
            with patch.object(v2.shared,'contract') as contract,patch.object(v2.shared,'worker') as worker:
                with self.subTest(workloads=workloads),self.assertRaisesRegex(ValueError,'Distinct frozen'):v2.run(Path('binary'),Path('output'),workloads=workloads)
                contract.assert_not_called();worker.assert_not_called()

    def invoke(self,workloads):
        legacy=fixtures.Controlled();legacy.setUpClass();spec=v2.shared.MODELS['larger'];source=v2.ROOT/'models'/spec[0]
        geometry={'passed':True,'review':{'accepted':True},'bindings':{'artifacts/m5-larger-q4-v2/manifest.json':{'sha256':v2.loss.SPECS['larger']['packed']}}}
        original=v2.shared.metal_engine.digest
        def digest(p):
            if p==source/'model.safetensors':return spec[1]
            if p==source/'config.json':return spec[2]
            if p==source/'tokenizer.json':return 'c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539'
            if p==v2.ROOT/v2.loss.GEOMETRY_REL:return v2.loss.GEOMETRY_SHA
            if p==v2.ROOT/'artifacts/m5-larger-q4-v2/manifest.json':return v2.loss.SPECS['larger']['packed']
            return original(p)
        def worker(binary,source,model,name,manifest,arrays,raw,stderr,check,**kwargs):
            response=legacy.fixture(name);response['memory']=admission()['memory'];setup,request,cache=v2.shared.requests(name,arrays,manifest)
            return {'passed':True,'workload':name,'ready':admission(),'samples':[{'response':response,'process_peak_footprint_bytes':1}]*13,'idle_usage':[{'worker':{'lifetime_max_phys_footprint':1}}],'cache_state':cache}
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);binary=p/'binary';binary.write_text('fixture')
            with patch.object(v2.loss.shared,'frozen_bytes',return_value=json.dumps(geometry).encode()),patch.object(v2.shared.metal_engine,'digest',side_effect=digest),patch.object(v2.shared.metal_prefill,'observation',side_effect=lambda a:'17179869184' if a[-1]=='hw.memsize' else 'fixture'),patch.object(v2.shared.metal_prefill,'endpoints',return_value={}),patch.object(v2.shared,'worker',side_effect=worker) as call:
                result=v2.run(binary,p/'out.json',False,workloads=workloads)
            return result,[c.args[3] for c in call.call_args_list]

    def test_scoped_confirmation_cannot_claim_full_matrix(self):
        result,names=self.invoke(['short']);self.assertEqual(names,['short']);self.assertEqual(result['selected_workloads'],['short']);self.assertEqual(len(result['results']),1)
        self.assertTrue(result['measurements_completed']);self.assertFalse(result['full_matrix_completed'])

    def test_default_still_runs_canonical_all_eight_and_only_complete_scope_marks_full(self):
        result,names=self.invoke(None);self.assertEqual(names,list(v2.shared.frozen.WORKLOADS));self.assertEqual(len(result['results']),8);self.assertTrue(result['full_matrix_completed'])
        result,names=self.invoke(['cached','short']);self.assertEqual(names,['cached','short']);self.assertFalse(result['full_matrix_completed'])

if __name__=='__main__':unittest.main()
