"""Small BF16 replay controls, including a partial final position batch."""
import hashlib
import copy
import math
from pathlib import Path
import tempfile
import unittest
import gemma4_loss_replay as target


class Controls(unittest.TestCase):
    def test_uniform_tail_and_coherent_wrong_producer_rejection(self):
        ids=[0]*34;raw=b'\0'*(33*17*2)
        batches=[]
        for start,end in [(0,32),(32,33)]:
            body=raw[start*34:end*34]
            batches.append(dict(start=start,end=end,shape=[1,end-start,17],offset=start*34,
                bytes=len(body),sha256=hashlib.sha256(body).hexdigest()))
        producer=dict(targets=33,nlls=[math.log(17)]*33,greedy=[0]*33)
        with tempfile.TemporaryDirectory()as folder:
            path=Path(folder)/'rows';path.write_bytes(raw)
            result=target.replay(path,batches,ids,producer,block=9,global_start=9216,vocab=17)
            self.assertEqual(result['targets'],33)
            self.assertEqual(result['records'][-1]['global_target_index'],9249)
            self.assertEqual(result['mean_nll'],math.log(17))
            for wrong in [dict(producer,nlls=[0.]*33),dict(producer,greedy=[1]*33),dict(producer,targets=32)]:
                with self.assertRaises(ValueError):target.replay(path,batches,ids,wrong,block=9,global_start=9216,vocab=17)
            corrupt=b'\xc0\x7f'+raw[2:]
            path.write_bytes(corrupt)
            changed=copy.deepcopy(batches)
            changed[0]['sha256']=hashlib.sha256(corrupt[:32*34]).hexdigest()
            with self.assertRaisesRegex(ValueError,'Finite'):
                target.replay(path,changed,ids,producer,block=9,global_start=9216,vocab=17)
            path.write_bytes(raw[:-1])
            with self.assertRaises(ValueError):target.replay(path,batches,ids,producer,block=9,global_start=9216,vocab=17)


if __name__=='__main__':unittest.main()
