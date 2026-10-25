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


if __name__=='__main__':unittest.main()
