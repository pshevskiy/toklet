"""Independent stable distribution math, full-input admission and memory controls."""
from pathlib import Path
import copy
import hashlib
import json
import math
import io
import tempfile
import unittest
from unittest.mock import patch

import qwen3_quality_source as subject


class QualitySource(unittest.TestCase):
    def test_short_return_and_missing_rows_cannot_emit_success(self):
        import numpy as np
        with self.assertRaises(ValueError):subject.require_output([1,2,151936],'mlx.core.bfloat16',3)
        subject.require_output([1,3,151936],'mlx.core.bfloat16',3)
        events=[]
        with self.assertRaises(ValueError):
            subject.score_block([(0,bytes(151936*2),np.zeros((1,151936),dtype='<f4'))],0,0,2,[0,1,0],events.append)
        self.assertEqual([v['event']for v in events],['loss'])

    def test_failed_capture_retains_descriptors_and_short_write_prefix(self):
        class Short(io.BytesIO):
            def write(self,value):return super().write(value[:7])
            def close(self):pass
        stream=Short();report={}
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)/'failed'
            with patch.object(Path,'open',return_value=stream):
                with self.assertRaises(OSError):subject.retain_failed_output(folder,1,lambda _:iter([b'abcdefghij']),report)
            record=report['failed_outputs'][str(folder/'source-bits.bin')]
            self.assertEqual(record['bytes'],7);self.assertEqual(record['sha256'],hashlib.sha256(b'abcdefg').hexdigest())
            self.assertEqual(report['failed_outputs'][str(folder/'widened.f32le.bin')]['bytes'],0)

    def test_early_eos_is_hidden_pending_and_never_committed(self):
        for sequence,expected in [([151645],[]),([42,151645],[42]),([42,43,44,45],[42,43,44,45])]:
            values=iter(sequence);commits=[];events=[]
            report,visible=subject.generate_task(dict(id='authored',prompt_tokens=[11,12],max_output_tokens=4),
                lambda:dict(greedy=next(values)),commits.append,events.append)
            self.assertEqual(visible,expected)
            self.assertEqual(report['sampled_tokens'],len(sequence))
            self.assertEqual(report['evaluated_tokens'],2+len(sequence)-1)
            self.assertEqual(report['pending_tokens'],1)
            self.assertEqual(commits,sequence[:-1])
            if sequence[-1]==151645:
                self.assertEqual(report['sampled_tokens'],len(visible)+1)
                self.assertEqual(events[-1]['visible'],False)

    def test_raw_logit_ties_and_manual_loss(self):
        import numpy as np
        row = np.array([0,0],dtype='<f4');value = subject.row_statistics(row,1)
        self.assertEqual((value['greedy'],value['second'],value['margin']),(0,1,0))
        self.assertAlmostEqual(value['negative_log_likelihood'],math.log(2),places=15)
        row = np.array([1000,-1000],dtype='<f4')
        self.assertEqual(subject.row_statistics(row,1)['negative_log_likelihood'],2000)
        a=subject.row_statistics(np.array([1,2,3],dtype='<f4'),0)
        b=subject.row_statistics(np.array([101,102,103],dtype='<f4'),0)
        self.assertEqual(a['negative_log_likelihood'],b['negative_log_likelihood'])

    def test_nonfinite_shape_dtype_and_target_reject(self):
        import numpy as np
        for row,target in [(np.array([1,np.nan],dtype='<f4'),None),(np.array([1,np.inf],dtype='<f4'),0),
            (np.array([1],dtype='<f4'),0),(np.ones((2,2),dtype='<f4'),0),
            (np.ones(2,dtype='<f8'),0),(np.ones(2,dtype='<f4'),2),(np.ones(2,dtype='<f4'),True)]:
            with self.assertRaises(ValueError):subject.row_statistics(row,target)

    def test_phase_memory_is_bounded_and_growth_is_charged(self):
        value=subject.memory_plan()
        self.assertEqual(value['load']['fp32_constructor'],2384199680)
        self.assertEqual(value['inference']['cache_and_growth_overlap'],2*2*28*8*1280*128*2)
        self.assertEqual(value['admitted'],max(sum(value['load'].values()),sum(value['inference'].values())))
        self.assertLessEqual(value['admitted'],6*1024**3)

    def test_probes_and_full_loss_are_explicit_and_independent_of_downloads(self):
        folder=subject.ROOT/'benchmarks/qwen3-quality-v1'
        _,ids,_=subject.load_inputs(folder,'probe');self.assertEqual(ids,[9707,11,1879,0])
        _,cases,_=subject.load_inputs(folder,'task_probe');self.assertEqual(len(cases),1);self.assertEqual(cases[0]['max_output_tokens'],4)
        _,ids,_=subject.load_inputs(folder,'loss');self.assertEqual(len(ids),10001)
        with self.assertRaises(ValueError):subject.load_inputs(folder,'unknown')
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);manifest=json.loads((folder/'manifest.json').read_text())
            body=(folder/'held_out_loss.u32le').read_bytes()[:1025*4]
            manifest['files']['held_out_loss.u32le']=dict(bytes=len(body),sha256=hashlib.sha256(body).hexdigest())
            (root/'held_out_loss.u32le').write_bytes(body)
            with patch.object(subject.source.calibration.reference,'load',return_value=manifest):
                with self.assertRaisesRegex(ValueError,'All10000'):subject.load_inputs(root,'loss')

    def test_full_task_caps_and_order_reject_even_with_consistent_hash(self):
        manifest=json.loads((subject.ROOT/'benchmarks/qwen3-quality-v1/manifest.json').read_text())
        ids=[f'HumanEval/{i}'for i in range(40)]+[f'truthfulqa/row-{i}'for i in range(30)]+[f'lambada/row-{i}'for i in range(30)]
        cases=[dict(id=label,category='code'if i<40 else 'prose'if i<70 else 'continuation',
            prompt_tokens=[9707],max_output_tokens=256 if i<40 else 32 if i<70 else 16)for i,label in enumerate(ids)]
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            for label in ('valid','missing','cap','order','budget'):
                modified=copy.deepcopy(cases)
                if label=='missing':modified.pop()
                elif label=='cap':modified[0]['max_output_tokens']=255
                elif label=='order':modified[0],modified[1]=modified[1],modified[0]
                elif label=='budget':modified[0]['prompt_tokens']=[11]*770
                body=b'\n'.join(json.dumps(row).encode()for row in modified)+b'\n'
                (root/'execution.jsonl').write_bytes(body)
                manifest['files']['execution.jsonl']=dict(bytes=len(body),sha256=hashlib.sha256(body).hexdigest())
                with patch.object(subject.source.calibration.reference,'load',return_value=manifest):
                    if label=='valid':self.assertEqual(len(subject.load_inputs(root,'tasks')[1]),100)
                    else:
                        with self.assertRaises(ValueError):subject.load_inputs(root,'tasks')


if __name__=='__main__':unittest.main()
