"""Portable deeper-workload bounds and complete/failed streaming controls."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import qwen35_9b_context256_reference as producer


class Source:
    def __init__(self,*args,**kwargs):self.closed=False
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
    def close(self):self.closed=True
    def audit(self):return dict(closed=self.closed,poisoned=False)


class Controls(unittest.TestCase):
    def test_frozen_workload_identity_extents_and_resource_admission(self):
        ids,offset,values=producer.plan()
        self.assertEqual(len(ids),256);self.assertEqual(ids[:17],list(producer.prior.graph.TOKENS))
        self.assertEqual(offset,37);self.assertEqual(values,{128:18773504,256:20903424})
        self.assertEqual(sum(values.values())*12,476123136)
        self.assertLess(sum(values.values())*12,producer.MAX_CAPTURE)
        self.assertLess(sum(producer.LEDGER.values()),6*1024**3)
        self.assertEqual(producer.DEADLINE_SECONDS,28800)
        self.assertEqual(producer.prior.graph.GATE,dict(atol=1e-5,rtol=1e-4))
        for n in producer.CHECKPOINTS:
            self.assertEqual(sum(producer.prior.weights.expected_native(n).values()),values[n])
        ids.clear();values.clear();self.assertEqual(len(producer.plan()[0]),256)
        self.assertEqual(producer.prior.plan('context17'),(list(producer.prior.graph.TOKENS),37,16926464))

    def test_checkpoint_selection_is_typed_bounded_and_exact(self):
        selected=[producer.checkpoint(t) for t in range(256)]
        self.assertEqual([v for v in selected if v is not None],[128,256])
        for bad in [True,False,-1,256,127.,None,'127']:
            with self.assertRaises(ValueError):producer.checkpoint(bad)

    def test_original_arithmetic_and_complete_cold_source_inventory(self):
        extras=producer.extras();names={p.name for p in extras}
        self.assertEqual(len(extras),12)
        self.assertTrue({'qwen35_cpu_graph.py','qwen35_vector_graph.py','qwen35_recurrent_vector.py',
            'build_qwen35_vector_graph.py','qwen35_9b_context256_reference.py','qwen35_9b_vector_reference.py'}<=names)
        code="import sys;sys.path.insert(0,sys.argv[1]);import qwen35_9b_context256_reference as p;p.extras();assert 'numpy' not in sys.modules"
        subprocess.run([sys.executable,'-s','-c',code,str(Path(__file__).resolve().parent)],check=True,capture_output=True)

    def collect(self,folder,forward,endpoint=lambda:{}):
        ids,offset,_=producer.plan()
        with patch.object(producer,'plan',return_value=(ids,offset,{128:4,256:4})),\
             patch.object(producer.prior,'ARRAYS',2),\
             patch.object(producer.prior.bf16_source,'Source',Source),\
             patch.object(producer.prior.runtime_owner.runtime,'captured',return_value=b'config'),\
             patch.object(producer.prior.weights,'weights',return_value=({},{})),\
             patch.object(producer.prior.weights,'math_type',return_value=object),\
             patch.object(producer.prior.graph,'forward',side_effect=forward),\
             patch.object(producer.prior,'source_endpoint',return_value=producer.prior.pin.FILE_SHA):
            return producer.collect(folder,folder.parent,{},endpoint)

    @staticmethod
    def forward(matrices,vectors,tokens,offset,staged,**kwargs):
        if len(tokens)!=256 or offset!=37:raise ValueError('Wrong actual causal pass')
        sink=kwargs['sink']
        # Noncheckpoint observations must be discarded without entering the body.
        sink(dict(token=0,layer=None,stage='embedding',values=[999.]))
        for token in [127,255]:
            for stage,values in [('embedding',[.125,-0.]),('logits',[.5,-1.])]:
                sink(dict(token=token,layer=None,stage=stage,values=values))
        return []

    def test_streaming_both_profiles_and_checkpoints_close_and_seal(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'out';record=self.collect(folder,self.forward)
            self.assertTrue(record['completed']);self.assertIsNone(record['primary_error'])
            self.assertEqual(record['endpoint_errors'],[]);self.assertTrue(record['source']['closed'])
            self.assertEqual(len(record['arrays']),8);self.assertEqual((folder/'rows.bin').stat().st_size,96)
            self.assertEqual(record['rows_sha256'],hashlib.sha256((folder/'rows.bin').read_bytes()).hexdigest())
            self.assertEqual(list(record['profile_ns']),['ideal_fp64','staged_fp32'])
            self.assertEqual([(r['profile'],r['checkpoint']) for r in record['arrays']],
                [(p,n) for p in ['ideal_fp64','staged_fp32'] for n in [128,256] for _ in range(2)])
            self.assertEqual(json.loads((folder/'result.json').read_text()),record)
            with self.assertRaises(FileExistsError):self.collect(folder,self.forward)

    def test_incomplete_duplicate_and_nonfinite_checkpoints_never_complete(self):
        for mode in ['missing','duplicate','nonfinite']:
            def broken(*args,**kwargs):
                if mode=='missing':return []
                row=dict(token=127,layer=None,stage='embedding',values=[float('nan')] if mode=='nonfinite' else [.125,-0.])
                kwargs['sink'](dict(row));kwargs['sink'](dict(row));return []
            with tempfile.TemporaryDirectory() as tmp:
                folder=Path(tmp)/'out';record=self.collect(folder,broken)
                self.assertFalse(record['completed']);self.assertEqual(record['primary_error']['type'],'ValueError')
                self.assertTrue(record['source']['closed']);self.assertTrue((folder/'result.json').is_file())

    def test_primary_interruption_is_preserved_with_endpoint_failure(self):
        def interrupted(*args,**kwargs):raise KeyboardInterrupt('original interruption')
        def endpoint():raise OSError('endpoint problem')
        with tempfile.TemporaryDirectory() as tmp:
            record=self.collect(Path(tmp)/'out',interrupted,endpoint)
            self.assertFalse(record['completed']);self.assertEqual(record['primary_error'],dict(type='KeyboardInterrupt',message='original interruption'))
            self.assertEqual(record['endpoint_errors'],['endpoint problem']);self.assertTrue(record['source']['closed'])


if __name__=='__main__':unittest.main()
