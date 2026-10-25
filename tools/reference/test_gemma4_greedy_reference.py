"""Offline selection/capture/failure checks; no downloaded model or GPU required."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import gemma4_greedy_reference as driver

ROOT=Path(__file__).resolve().parents[2]


class GreedyReference(unittest.TestCase):
    def test_closed_workloads_and_profile_capture_bounds(self):
        raw=driver.workload('raw');self.assertEqual(raw['prompt'],[2,100,257,100]);self.assertEqual(raw['steps'],4)
        basic=driver.workload('basic-hello');self.assertEqual(len(basic['prompt']),10)
        self.assertEqual((basic['steps'],basic['largest_prefix'],basic['planned_bytes']),(16,25,100958208))
        self.assertEqual(basic['stops'],[1,106,50])
        self.assertLessEqual(2*25*262144*8,128*1024**2)
        with self.assertRaises(ValueError):driver.workload('arbitrary')
        binary=io.BytesIO();arrays=[]
        with self.assertRaisesRegex(ValueError,'fixed capture profiles'):
            driver.capture(binary,arrays,0,'staged_fp32','final_norm',np.zeros(1536,dtype=np.float32),
                           steps=17,planned_bytes=200000000,maximum=1024**3)
        self.assertFalse(arrays);self.assertEqual(binary.tell(),0)
    def test_first_maximum_ties_margin_and_finite_vocabulary(self):
        for dtype in (np.float32,np.float64):
            self.assertEqual(driver.sample(np.array([1.,5.,5.,-1.],dtype=dtype)),dict(token=1,margin=0.,maximum_ties=2))
            self.assertEqual(driver.sample(np.array([1.,5.,2.,-1.],dtype=dtype)),dict(token=1,margin=3.,maximum_ties=1))
            self.assertEqual(driver.sample(np.array([100000000.,1.,0.],dtype=dtype))['margin'],99999999.)
            for invalid in ([1.],[1.,np.nan],[1.,np.inf],[[1.,2.]]):
                with self.assertRaises(ValueError):driver.sample(np.array(invalid,dtype=dtype))

    def test_complete_typed_capture_retains_bytes_before_nonfinite_or_write_error(self):
        for contract,dtype in [('math_fp64',np.float64),('staged_fp32',np.float32)]:
            value=np.arange(1536,dtype=dtype);arrays=[];binary=io.BytesIO()
            item=driver.capture(binary,arrays,0,contract,'final_norm',value)
            self.assertTrue(item['completed']);self.assertEqual(binary.getvalue(),value.tobytes())
            self.assertEqual(item['sha256'],hashlib.sha256(binary.getvalue()).hexdigest())
            arrays=[];binary=io.BytesIO();value[-1]=np.inf
            with self.assertRaises(ValueError):driver.capture(binary,arrays,0,contract,'final_norm',value)
            self.assertTrue(arrays[0]['completed']);self.assertEqual(binary.getvalue(),value.tobytes())
            class Broken(io.BytesIO):
                def write(self,_):raise OSError('injected write failure')
            arrays=[]
            with self.assertRaisesRegex(OSError,'injected write failure'):driver.capture(Broken(),arrays,0,contract,'final_norm',value)
            self.assertEqual(len(arrays),1);self.assertFalse(arrays[0]['completed'])
        for name,shape,dtype in [('unknown',1536,np.float32),('final_norm',1535,np.float32),('final_norm',1536,np.float64)]:
            with self.assertRaises(ValueError):driver.capture(io.BytesIO(),[],0,'staged_fp32',name,np.zeros(shape,dtype=dtype))
        self.assertEqual(driver.PLANNED_BYTES,25239552)
        self.assertLess(driver.PLANNED_BYTES,driver.MAX_BINARY)

    def test_actual_driver_retains_nonfinite_selected_row_before_rejection(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as directory:
            root=Path(directory);model=root/'model';model.mkdir();raw=b'synthetic model bytes'
            (model/'model.safetensors').write_bytes(raw);pins={'model.safetensors':hashlib.sha256(raw).hexdigest()}
            def evaluate(_,ids,staged,*,sink):
                values=np.zeros((1,len(ids),1536),dtype=np.float32 if staged else np.float64)
                values[0,-1,-1]=np.inf;sink('final_norm',values)
            output=root/'observation'
            with mock.patch.multiple(driver.seed.source,MODEL=model,PINS=pins), \
                 mock.patch.object(driver.seed,'weights',return_value=object()), \
                 mock.patch.object(driver.seed.graph,'evaluate',side_effect=evaluate), \
                 mock.patch.object(driver.seed.source,'descriptor_sha',return_value=pins['model.safetensors']):
                self.assertEqual(driver.run(output,{'synthetic_test':True},lambda:{}),1)
            report=json.loads((output/'observation.json').read_bytes())
            self.assertFalse(report['passed']);self.assertIn('Nonfinite',report['primary_error'])
            self.assertEqual(report['sampled_ids'],[]);self.assertFalse(report['steps'][0]['completed'])
            self.assertEqual(len(report['arrays']),1);item=report['arrays'][0];self.assertTrue(item['completed'])
            body=(output/'arrays.bin').read_bytes()
            self.assertEqual(len(body),1536*8);self.assertEqual(hashlib.sha256(body).hexdigest(),item['sha256'])
            self.assertTrue(np.isinf(np.frombuffer(body,dtype='<f8')[-1]))

    def test_actual_driver_keeps_incomplete_step_without_selecting_after_failure(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as directory:
            root=Path(directory);model=root/'model';model.mkdir();raw=b'synthetic model bytes'
            (model/'model.safetensors').write_bytes(raw);pins={'model.safetensors':hashlib.sha256(raw).hexdigest()}
            for label,error in [('error',OSError('injected graph failure')),('interrupt',KeyboardInterrupt('injected interruption')),('endpoint',ValueError('original primary'))]:
                def evaluate(_,ids,staged,*,sink):
                    sink('final_norm',np.zeros((1,len(ids),1536),dtype=np.float32 if staged else np.float64));raise error
                output=root/label
                with mock.patch.multiple(driver.seed.source,MODEL=model,PINS=pins), \
                     mock.patch.object(driver.seed,'weights',return_value=object()), \
                     mock.patch.object(driver.seed.graph,'evaluate',side_effect=evaluate), \
                     mock.patch.object(driver.seed.source,'descriptor_sha',return_value='bad' if label=='endpoint' else pins['model.safetensors']):
                    self.assertEqual(driver.run(output,{'synthetic_test':True},lambda:{}),1)
                report=json.loads((output/'observation.json').read_bytes())
                self.assertFalse(report['passed']);self.assertIn(str(error),report['primary_error'])
                self.assertEqual(report['sampled_ids'],[]);self.assertEqual(len(report['steps']),1)
                self.assertFalse(report['steps'][0]['completed']);self.assertEqual(len(report['arrays']),1)
                self.assertTrue(report['arrays'][0]['completed'])
                self.assertEqual(report['binary']['sha256'],hashlib.sha256((output/'arrays.bin').read_bytes()).hexdigest())
                if label=='endpoint':self.assertTrue(report['endpoint_errors'])

    def test_formatted_first_final_stop_output_limit_and_between_contract_failure(self):
        spec=driver.workload('basic-hello')
        shapes={f'dummy.{i}':(1,) for i in range(710)}
        shapes.update({name:(1,1,width) for name,width in driver.CAPTURE.items()})
        self.assertEqual(len(shapes),713)
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as directory:
            root=Path(directory);model=root/'model';model.mkdir();raw=b'synthetic model bytes'
            (model/'model.safetensors').write_bytes(raw);pins={'model.safetensors':hashlib.sha256(raw).hexdigest()}
            for label,sequence,failure in [('first',[106],False),('last',[7]*15+[1],False),
                                           ('limit',[7]*16,False),('between',[7],True)]:
                def evaluate(_,ids,staged,*,sink):
                    if failure and staged:raise OSError('injected between contracts')
                    token=sequence[len(ids)-len(spec['prompt'])]
                    for name,shape in shapes.items():
                        value=np.zeros(shape,dtype=np.float32 if staged else np.float64)
                        if name=='logits':value[0,0,token]=5.
                        sink(name,value)
                output=root/label
                with mock.patch.multiple(driver.seed.source,MODEL=model,PINS=pins),\
                     mock.patch.object(driver.seed,'weights',return_value=object()),\
                     mock.patch.object(driver.seed.graph,'stage_shapes',return_value=shapes),\
                     mock.patch.object(driver.seed.graph,'evaluate',side_effect=evaluate),\
                     mock.patch.object(driver.seed.source,'descriptor_sha',return_value=pins['model.safetensors']):
                    self.assertEqual(driver.run(output,{'synthetic_test':True},lambda:{},profile='basic-hello'),1 if failure else 0)
                report=json.loads((output/'observation.json').read_bytes())
                if failure:
                    self.assertFalse(report['passed']);self.assertEqual(report['sampled_ids'],[])
                    self.assertEqual(report['visible_ids'],[]);self.assertIsNone(report['finish_reason'])
                    self.assertEqual(len(report['arrays']),3);self.assertFalse(report['steps'][0]['completed'])
                    self.assertIn('between contracts',report['primary_error']);continue
                self.assertTrue(report['passed']);self.assertEqual(report['sampled_ids'],sequence)
                self.assertEqual(report['pending_id'],sequence[-1]);self.assertEqual(report['committed_ids'],spec['prompt']+sequence[:-1])
                self.assertEqual(len(report['arrays']),6*len(sequence))
                self.assertEqual(report['binary']['bytes'],len(sequence)*(1536+2*262144)*12)
                if label=='limit':
                    self.assertEqual(report['visible_ids'],sequence);self.assertEqual(report['finish_reason'],dict(kind='output_limit'))
                else:
                    self.assertEqual(report['visible_ids'],sequence[:-1]);self.assertEqual(report['finish_reason'],dict(kind='stop_token',token_id=sequence[-1]))
                self.assertTrue(report['policy_diagnostics_passed']);self.assertTrue(report['all_cross_precision_gates_passed'])


if __name__=='__main__':unittest.main()
