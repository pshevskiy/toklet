"""Independent uniform-row journal, alignment and coherent-corruption controls."""
import copy
import hashlib
import json
import math
import unittest
import check_gemma_loss_block as target


def fixture():
    row=b'\0'*(4*target.VOCAB);sha=hashlib.sha256(row).hexdigest()
    ids=[0]*1025;records=[]
    for position in range(1024):
        records.append(dict(block=0,local_row=position,global_target_index=position+1,target=0,
            target_fp32_bits=0,vocab=target.VOCAB,greedy_lowest_id=0,maximum=0.,exp_mass=target.VOCAB,
            nll=math.log(target.VOCAB),row_sha256=sha))
    arrays=[dict(record=i,name='loss.anchor',offset=i*len(row),bytes=len(row),before=p,count=1,
                 values=target.VOCAB,sha256=sha)for i,p in enumerate(target.POSITIONS)]
    body=row*8
    summary=dict(targets=1024,capacity=1025,chunk_capacity=32,session_allocated_bytes=271117548,
                 quality_qualified=False,mean_nll=math.log(target.VOCAB))
    footer=dict(bytes=len(body),records=8,sha256=hashlib.sha256(body).hexdigest())
    def encode(records=records,arrays=arrays,summary=summary):
        lines=[b'test result: ok.']
        for prefix,items in [(b'GEMMA_HYBRID_REAL_LOSS_TARGET ',records),
            (b'GEMMA_HYBRID_REAL_LOSS_UNSCORED ',[dict(position=1024,row_sha256=sha)]),
            (b'GEMMA_HYBRID_REAL_LOSS_SUMMARY ',[summary]),
            (b'GEMMA_HYBRID_ARRAY ',arrays),(b'GEMMA_HYBRID_CAPTURE ',[footer])]:
            lines.extend(prefix+json.dumps(item).encode()for item in items)
        return b'\n'.join(lines)
    return encode,records,arrays,summary,body,ids,{p:sha for p in target.POSITIONS}


class ReplayControls(unittest.TestCase):
    def test_uniform_complete_rows_and_stable_aggregate(self):
        encode,records,arrays,summary,body,ids,prior=fixture()
        result=target.replay(encode(),body,ids,prior)
        self.assertTrue(result['passed'])
        self.assertEqual(result['targets'],1024)
        self.assertEqual(result['independent_scored_anchors'],target.POSITIONS[:-1])
        self.assertEqual(result['mean_nll'],math.log(target.VOCAB))

    def test_bad_coverage_formula_coherent_statistics_and_rows_reject(self):
        encode,records,arrays,summary,body,ids,prior=fixture()
        for change in [lambda r:r.pop(),lambda r:r.__setitem__(1,r[0]),
            lambda r:r[0].update(target=1),lambda r:r[0].update(local_row=True),
            lambda r:r[0].update(nll=1.),lambda r:r[0].update(maximum=float('nan')),
            lambda r:r[0].update(exp_mass=1.,nll=0.)]:
            changed=copy.deepcopy(records);change(changed)
            with self.assertRaises(ValueError):target.replay(encode(records=changed),body,ids,prior)
        changed=copy.deepcopy(arrays);changed[-1]['before']=1023
        with self.assertRaises(ValueError):target.replay(encode(arrays=changed),body,ids,prior)
        for raw in [body[:-1],b'\1'+body[1:]]:
            with self.assertRaises(ValueError):target.replay(encode(),raw,ids,prior)


if __name__=='__main__':
    unittest.main()
