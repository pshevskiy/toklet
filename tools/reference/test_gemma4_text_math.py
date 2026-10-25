"""Offline complete-graph checks against independently frozen packed witnesses."""
from dataclasses import replace
import hashlib
import io
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
import tempfile
from unittest import mock

import numpy as np
import affine4_tiles as tiles
import gemma4_text_math as graph

ROOT=Path(__file__).resolve().parents[2]


def fixture():
    receipt=ROOT/'benchmarks/results/2026-10-25-gemma-packed-graph-reference-v1.json'
    assert hashlib.sha256(receipt.read_bytes()).hexdigest()=='985e3d06115533396321be974a34fe4eecad5e5495efcc1fb2c327577e96c819'
    accepted=json.loads(receipt.read_bytes())
    folder=ROOT/'tests/fixtures/gemma-packed-graph-v1';raw_manifest=(folder/'math.json').read_bytes()
    assert hashlib.sha256(raw_manifest).hexdigest()==accepted['fixture_sha256']
    manifest=json.loads(raw_manifest);raw=(folder/'math.bin').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==manifest['binary']['sha256']
    stream=io.BytesIO(raw);matrices={};vectors={}
    for name,shape in graph.shapes(graph.WITNESS).items():
        original='model.'+name
        if len(shape)==2:
            item=manifest['packed_original_bf16'][original]
            matrices[name]=tiles.MatrixRows(stream,len(raw),*shape,*[item[k]['offset'] for k in ('weight','scales','biases')])
        else:
            item=manifest['weights'][original];body=raw[item['offset']:item['offset']+item['bytes']]
            assert hashlib.sha256(body).hexdigest()==item['sha256']
            vectors[name]=np.frombuffer(body,dtype='<f4').copy()
    return graph.Weights(graph.WITNESS,matrices,vectors),manifest,raw


def gate(actual,expected):
    assert actual.shape==expected.shape
    difference=np.abs(actual.astype(np.float64)-expected.astype(np.float64))
    bound=1e-5+1e-4*np.abs(expected.astype(np.float64))
    assert np.isfinite(actual).all() and np.all(difference<=bound),float(np.max(difference/bound))


def prefix(value,name,count):
    return value[:,:,:count] if name.endswith(('rotary','context')) else value[:,:count]


class CompleteGraph(unittest.TestCase):
    def test_every_frozen_state_in_both_precision_contracts(self):
        weights,manifest,raw=fixture()
        for staged,contract in [(False,'math'),(True,'staged_fp32')]:
            actual=graph.evaluate(weights,manifest['token_ids'],staged)
            self.assertEqual(set(actual),set(manifest[contract]))
            self.assertEqual(len(actual),90)
            self.assertEqual({n:tuple(x.shape) for n,x in actual.items()},graph.stage_shapes(graph.WITNESS,32))
            for name,item in manifest[contract].items():
                body=raw[item['offset']:item['offset']+item['bytes']]
                self.assertEqual(hashlib.sha256(body).hexdigest(),item['sha256'])
                expected=np.frombuffer(body,dtype=item['dtype']).reshape(item['shape'])
                gate(actual[name],expected)

    def test_bounded_softcap_tiles_and_tail_match_scalar_rounding_and_signed_zero(self):
        source=np.array([-10000.,-30.,-1.,-0.,0.,1.,30.,10000.]*257).reshape(1,2056)
        for staged,dtype in [(False,np.float64),(True,np.float32)]:
            expected=[]
            for v in source.flat:
                divided=float(dtype(float(v)/30.))
                capped=float(dtype(math.tanh(divided)))
                expected.append(dtype(capped*30.))
            expected=np.array(expected,dtype=dtype).reshape(source.shape)
            self.assertEqual(graph.softcap(source,staged).tobytes(),expected.tobytes())

    def test_prefix_recomputation_within_frozen_gates_and_future_token_isolation(self):
        weights,manifest,_=fixture();ids=manifest['token_ids']
        for staged in (False,True):
            full=graph.evaluate(weights,ids,staged)
            changed=graph.evaluate(weights,ids[:-1]+[(ids[-1]+1)%17],staged)
            for name,value in full.items():gate(prefix(changed[name],name,31),prefix(value,name,31))
            for count in (1,3,8):
                shorter=graph.evaluate(weights,ids[:count],staged)
                for name,value in shorter.items():gate(value,prefix(full[name],name,count))

    def test_real_geometry_inventory_and_double_width_boundary(self):
        shapes=graph.shapes(graph.E2B)
        self.assertEqual(len(shapes),540)
        self.assertEqual(sum(len(s)==2 for s in shapes.values()),278)
        self.assertEqual(sum(len(s)==1 for s in shapes.values()),262)
        self.assertEqual(shapes['embed_tokens_per_layer.weight'],(262144,8960))
        self.assertEqual(shapes['layers.14.mlp.down_proj.weight'],(1536,6144))
        self.assertEqual(shapes['layers.15.mlp.down_proj.weight'],(1536,12288))
        self.assertEqual(shapes['layers.14.self_attn.k_proj.weight'],(512,1536))
        self.assertNotIn('layers.15.self_attn.k_proj.weight',shapes)
        self.assertEqual(graph.E2B.dimension(13),256)
        self.assertEqual(graph.E2B.dimension(14),512)

    def test_graph_and_weight_admission_rejects_unknown_or_incomplete_inputs(self):
        weights,manifest,_=fixture()
        for ids,staged in [([],False),([True],False),([-1],False),([17],False),([0]*33,False),([0],1)]:
            with self.assertRaises(ValueError):graph.evaluate(weights,ids,staged)
        with self.assertRaises(ValueError):graph.shapes(replace(graph.WITNESS,layers=5))
        with self.assertRaises(ValueError):graph.evaluate(SimpleNamespace(profile=graph.E2B),[2],True)
        with self.assertRaises(ValueError):graph.evaluate(weights,[1],True,sink='invalid')
        for change in ('missing','extra','shape','normshape','nonfinite'):
            matrices=dict(weights.matrices);vectors=dict(weights.vectors)
            if change=='missing':matrices.pop('embed_tokens.weight')
            elif change=='extra':vectors['unknown']=np.ones(1,dtype=np.float32)
            elif change=='shape':matrices['embed_tokens.weight']=SimpleNamespace(rows=18,columns=64)
            elif change=='normshape':vectors['norm.weight']=np.ones(63,dtype=np.float32)
            else:vectors['norm.weight']=np.full(64,np.inf,dtype=np.float32)
            with self.assertRaises(ValueError):graph.Weights(graph.WITNESS,matrices,vectors)

    def test_sink_matches_all_states_and_propagates_failure_without_mutating_weights(self):
        weights,manifest,_=fixture();ids=manifest['token_ids'][:3]
        control=graph.evaluate(weights,ids,True);captured={}
        self.assertEqual(graph.evaluate(weights,ids,True,sink=lambda n,x:captured.setdefault(n,x.copy())),{})
        self.assertEqual(set(captured),set(control))
        for name,value in captured.items():np.testing.assert_array_equal(value,control[name])
        stream=weights.matrices['embed_tokens.weight'].stream;before=stream.getvalue();partial=[]
        def sink(name,_):
            partial.append(name)
            if name=='layer.2.output':raise OSError('injected sink failure')
        with self.assertRaisesRegex(OSError,'injected sink failure'):graph.evaluate(weights,ids,True,sink=sink)
        self.assertEqual(partial[-1],'layer.2.output');self.assertNotIn('logits',partial)
        self.assertEqual(stream.getvalue(),before)

    def test_future_prefix_exact_hashes_and_nonexact_values_survive_failed_gate(self):
        import gemma4_text_reference as driver
        for dtype in (np.float32,np.float64):
            reference=np.zeros((1,4,8),dtype=dtype)
            for label in ('exact','signedzero','within','reject','nonfinite'):
                actual=reference.copy();actual[:,3]=99 # excluded future position
                if label=='signedzero':actual[0,0,0]=-0.
                if label=='within':actual[0,0,0]=1e-6
                if label=='reject':actual[0,0,0]=1.
                if label=='nonfinite':actual[0,0,0]=np.inf
                record={'future_prefix':[]};binary=io.BytesIO()
                if label in ('reject','nonfinite'):
                    with self.assertRaises(ValueError):driver.capture_future_prefix('test','embedding',actual,reference,record,binary)
                else:driver.capture_future_prefix('test','embedding',actual,reference,record,binary)
                self.assertEqual(len(record['future_prefix']),1);item=record['future_prefix'][0]
                self.assertTrue(item['capture_completed']);self.assertEqual(item['shape'],[1,3,8])
                self.assertEqual(item['actual_sha256'],hashlib.sha256(actual[:,:3].tobytes()).hexdigest())
                self.assertEqual(item['reference_sha256'],hashlib.sha256(reference[:,:3].tobytes()).hexdigest())
                self.assertEqual(item['exact_prefix'],label=='exact')
                if label=='exact':self.assertIsNone(item['raw']);self.assertEqual(binary.getvalue(),b'')
                else:
                    self.assertTrue(item['raw']['completed'])
                    self.assertEqual(binary.getvalue(),actual[:,:3].tobytes())
                self.assertEqual(item['passed'],label not in ('reject','nonfinite'))
                self.assertEqual(item['comparison_completed'],label!='nonfinite')
        # Rotary/context time is axis2, independently of the number of heads.
        reference=np.zeros((1,8,4,2));actual=reference.copy();actual[:,:,3]=1.
        record={'future_prefix':[]}
        driver.capture_future_prefix('test','layer.0.context',actual,reference,record,io.BytesIO())
        self.assertEqual(record['future_prefix'][0]['shape'],[1,8,3,2])
        class Broken(io.BytesIO):
            def write(self,_):raise OSError('injected future capture failure')
        record={'future_prefix':[]};actual[:,:,0]=1.
        with self.assertRaisesRegex(OSError,'injected future capture failure'):
            driver.capture_future_prefix('test','layer.0.context',actual,reference,record,Broken())
        item=record['future_prefix'][0]
        self.assertFalse(item['capture_completed']);self.assertFalse(item['raw']['completed'])
        self.assertFalse(item['comparison_completed']);self.assertFalse(item['passed'])

    def test_real_export_failure_interrupt_and_endpoint_keep_completed_arrays_and_probes(self):
        import gemma4_text_reference as driver
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as directory:
            root=Path(directory);model=root/'model';model.mkdir();raw=b'synthetic model bytes'
            (model/'model.safetensors').write_bytes(raw)
            pins={'model.safetensors':hashlib.sha256(raw).hexdigest()}
            for label,error in [('error',OSError('injected export failure')),
                                ('interrupt',KeyboardInterrupt('injected interruption')),
                                ('endpoint',ValueError('original primary'))]:
                def evaluate(_,ids,staged,*,sink):
                    value=np.zeros((1,4,1536),dtype=np.float32 if staged else np.float64)
                    sink('embedding',value);sink('scaled_embedding',value);raise error
                output=root/label
                with mock.patch.multiple(driver.source,MODEL=model,PINS=pins), \
                     mock.patch.object(driver,'weights',return_value=object()), \
                     mock.patch.object(driver.graph,'evaluate',side_effect=evaluate), \
                     mock.patch.object(driver.source,'descriptor_sha',return_value='bad' if label=='endpoint' else pins['model.safetensors']):
                    self.assertEqual(driver.run(output,{'synthetic_test':True},lambda:{}),1)
                report=json.loads((output/'observation.json').read_bytes())
                self.assertFalse(report['passed']);self.assertIn(str(error),report['primary_error'])
                self.assertEqual(len(report['arrays']),2)
                for name in ('arrays.bin','probes.bin'):
                    body=(output/name).read_bytes()
                    self.assertEqual(report[name]['sha256'],hashlib.sha256(body).hexdigest())
                for item in report['arrays']:
                    self.assertTrue(item['completed']);self.assertTrue(item['probe']['completed'])
                if label=='endpoint':self.assertTrue(report['endpoint_errors'])



if __name__=='__main__':unittest.main()
