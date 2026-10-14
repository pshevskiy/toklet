"""Offline exact-storage oracle, inventory rejection and owned-capture cleanup."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock,patch
import half_consistency as capture

class HalfConsistency(unittest.TestCase):
    def test_exact_dyadic_half_ties_sign_subnormal_and_overflow(self):
        for numerator,power,expected in [(0,0,0),(1,24,1),(-1,24,0x8001),(1,25,0),(3,25,2),(2049,11,0x3c00),(2051,11,0x3c02),(-2051,11,0xbc02),(65504,0,0x7bff),(65519,0,0x7bff),(65520,0,0x7c00),(-65520,0,0xfc00)]:
            with self.subTest(numerator=numerator,power=power):self.assertEqual(capture.exact_half(numerator,power),expected)
        bits,n,power=capture.dyadic_dot([1.,1.0009765625],[1.,-1.],0.)
        self.assertEqual(bits,0x9400);self.assertEqual(n/(1<<power),-1/1024)
        with self.assertRaises(ValueError):capture.half_value(0x7c00)

    def test_complete_stage_and_cache_identity_shape_order_are_required(self):
        record={'stages':[{'layer':l,'stage':s,'shape':[4,w],'half_storage':s!='logits','dtype':'f32le'} for l,s,w in capture.inventory()],
                'cache':[{'layer':l,'kind':k,'shape':[2,4,128],'dtype':'f16le'} for l in range(28) for k in ('k','v')],'admitted_bytes_including_extra':4*1024**3}
        self.assertEqual(len(record['stages']),451);capture.validate_inventory(record,True)
        mutations=[]
        v=copy.deepcopy(record);v['stages'].pop();mutations.append(v)
        for key,value in [('layer',27),('stage','logits'),('shape',[1,1536]),('dtype','f16le'),('half_storage',False)]:
            v=copy.deepcopy(record);v['stages'][1][key]=value;mutations.append(v)
        v=copy.deepcopy(record);v['cache'].pop();mutations.append(v)
        v=copy.deepcopy(record);v['cache'][0]['shape']=[4,2,128];mutations.append(v)
        v=copy.deepcopy(record);v['admitted_bytes_including_extra']=capture.LIMIT+1;mutations.append(v)
        for v in mutations:
            with self.assertRaises(ValueError):capture.validate_inventory(v,True)

    def test_owned_bad_charge_reaps_child_and_retains_failure(self):
        children=[];real=subprocess.Popen
        def launch(command,**kwargs):
            self.assertTrue(kwargs['start_new_session']);p=real([sys.executable,'-c','print("\\nTOKLET_HALF_CAPTURE_DONE",flush=True);import time;time.sleep(60)'],**kwargs);p.wait=Mock(wraps=p.wait);children.append(p);return p
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'src').mkdir();output=root/'result.json'
            def digest(path):return {'2026-10-14-larger-half-held-out-loss-v1.json':capture.BASELINE_SHA,'model.safetensors':capture.MODEL_SHA,'config.json':capture.CONFIG_SHA}.get(path.name,'test')
            with patch.object(capture,'ROOT',root),patch.object(capture,'digest',digest),patch.object(capture.subprocess,'Popen',launch),patch.object(capture.native_usage,'usage',return_value={'lifetime_max_phys_footprint':0}):
                with self.assertRaisesRegex(ValueError,'localization failed'):capture.collect(root/'binary',root/'capture',output)
            result=json.loads(output.read_text());self.assertFalse(result['passed']);self.assertIn('native charge',result['error']);self.assertEqual(result['cleanup_errors'],[]);self.assertIsNotNone(children[0].returncode);children[0].wait.assert_called_once_with(timeout=5)
            with patch.object(capture.subprocess,'Popen') as launch:
                with self.assertRaisesRegex(ValueError,'destination exists'):capture.collect(root/'binary',root/'capture',output)
                launch.assert_not_called()

    def test_first_projection_geometry_and_baseline_bytes_are_bound(self):
        self.assertEqual(capture.projection_geometry({'layer':0,'stage':'q'}),('input_norm',1536,1536,True))
        self.assertEqual(capture.projection_geometry({'layer':27,'stage':'down'}),('gated',1536,8960,False))
        self.assertEqual(capture.projection_geometry({'layer':None,'stage':'logits'}),('final_norm',151936,1536,False))
        for first in ({'layer':None,'stage':'q'},{'layer':0,'stage':'logits'},{'layer':0,'stage':'input_norm'}):
            with self.assertRaises(ValueError):capture.projection_geometry(first)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);folder=root/'benchmarks/results';folder.mkdir(parents=True)
            path=folder/'2026-10-14-larger-half-held-out-loss-v1.json';path.write_bytes(b'{}')
            with patch.object(capture,'ROOT',root):
                with self.assertRaisesRegex(ValueError,'baseline bytes differ'):capture.baseline_record()
                path.write_bytes(b' '*(2*1024**2+1))
                with self.assertRaisesRegex(ValueError,'baseline bytes differ'):capture.baseline_record()

if __name__=='__main__':unittest.main()
