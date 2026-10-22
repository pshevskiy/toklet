import math
import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import mlx_qwen_upstream_loss as q


class UpstreamLossChecks(unittest.TestCase):
    def test_stable_fp64_loss_and_first_greedy_tie(self):
        for values,target in [([0,0],1),([10000,9999,-10000],1),([-10000,-10001],0),([1,-1,1],2)]:
            logits=np.array(values,dtype=np.float32);out=q.loss(logits,target);maximum=max(values)
            expected=maximum+math.log(math.fsum(math.exp(v-maximum) for v in values))-values[target]
            self.assertEqual(out['negative_log_likelihood'],expected);self.assertEqual(out['greedy'],values.index(maximum))
            self.assertEqual(out['target_logit'],float(logits[target]))
    def test_nonfinite_shape_dtype_and_target_reject(self):
        for array,target in [(np.array([0.],dtype=np.float64),0),(np.array([],dtype=np.float32),0),
                             (np.array([[0]],dtype=np.float32),0),(np.array([float('nan')],dtype=np.float32),0),
                             (np.array([float('inf')],dtype=np.float32),0),(np.zeros(2,dtype=np.float32),True),
                             (np.zeros(2,dtype=np.float32),-1),(np.zeros(2,dtype=np.float32),2)]:
            with self.subTest(array=array,target=target),self.assertRaises(ValueError):q.loss(array,target)
    def test_frozen_block_boundaries_score_every_target_once(self):
        blocks=q.blocks();self.assertEqual(len(blocks),10);self.assertEqual(blocks[-1],(9216,10000))
        self.assertEqual([i for a,b in blocks for i in range(a,b)],list(range(10000)))
        for targets,size in [(9999,1024),(10000,1023),(10000.0,1024),(10000,True)]:
            with self.assertRaises(ValueError):q.blocks(targets,size)
    def test_existing_file_directory_and_dangling_output_precede_producer_work(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);file=base/'file';file.write_text('keep');directory=base/'directory';directory.mkdir();link=base/'link';link.symlink_to(base/'missing')
            with patch.object(q,'producers') as producers:
                for path in [file,directory,link]:
                    with self.assertRaises(FileExistsError):q.run(path)
                producers.assert_not_called()
            self.assertEqual(file.read_text(),'keep')
    def test_extra_or_symlink_shard_blocks_before_load_and_versions(self):
        with tempfile.TemporaryDirectory(dir=q.ROOT/'artifacts') as temp:
            base=Path(temp);model=base/'model';model.mkdir();weight=model/'model.safetensors';weight.write_bytes(b'pinned')
            q.model_inventory(model)
            for index,kind in enumerate(['extra','symlink']):
                if kind=='extra':extra=model/'model-extra.safetensors';extra.write_bytes(b'override')
                else:weight.unlink();weight.symlink_to(base/'missing')
                output=base/('run-'+str(index))
                with patch.object(q.reference,'MODEL',model),patch.object(q,'producers',return_value={}),patch.object(q.importlib.metadata,'version') as version:
                    with self.assertRaises(ValueError):q.run(output)
                    version.assert_not_called()
                result=json.loads((output/'result.json').read_text());self.assertFalse(result['passed']);self.assertEqual(result['completed_targets'],0)
                self.assertIn('inventory',result['primary_error'])
                if kind=='extra':extra.unlink()
    def test_actual_partial_journal_is_bound_after_synthetic_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'targets.jsonl';data=b'{"target_index":0}\n'
            with self.assertRaisesRegex(RuntimeError,'original failure'):
                with path.open('xb') as stream:
                    stream.write(data);raise RuntimeError('original failure')
            actual=q.retained_journal(path)
            self.assertEqual(actual,dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),complete_lines=1,trailing_bytes=False))
            with path.open('ab') as stream:stream.write(b'{partial')
            actual=q.retained_journal(path);self.assertEqual(actual['complete_lines'],1);self.assertTrue(actual['trailing_bytes'])
            self.assertEqual(actual['sha256'],hashlib.sha256(data+b'{partial').hexdigest())
            absent=Path(temp)/'absent';self.assertIsNone(q.retained_journal(absent))
            link=Path(temp)/'link';link.symlink_to(path)
            with self.assertRaises(ValueError):q.retained_journal(link)
    def test_separate_metal_library_and_mlx_headers_are_producer_inputs(self):
        with tempfile.TemporaryDirectory(dir=q.ROOT/'artifacts') as temp:
            site=Path(temp)
            for name,data in [('mlx/lib/mlx.metallib',b'kernel'),('mlx/include/runtime.h',b'header'),('mlx_lm/model.py',b'graph'),('numpy/array.so',b'binary')]:
                path=site/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
            library=site/'mlx/lib/mlx.metallib';name=str(library.relative_to(q.ROOT))
            with patch.object(q.sysconfig,'get_path',return_value=str(site)):
                before=q.producers();self.assertEqual(before[name],hashlib.sha256(b'kernel').hexdigest())
                self.assertIn(str((site/'mlx/include/runtime.h').relative_to(q.ROOT)),before)
                library.write_bytes(b'changed kernel');self.assertNotEqual(q.producers()[name],before[name])


if __name__=='__main__':unittest.main()
