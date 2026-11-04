"""10,000-target coverage, unequal-block weighting and unchanged gate controls."""
import copy
import math
import json
import unittest
import check_gemma_full_loss as target


def rows(mass):
    return [dict(block=(i-1)//1024,local_row=(i-1)%1024,global_target_index=i,target=0,
        target_fp32_bits=0,vocab=262144,greedy_lowest_id=0,maximum=0.,exp_mass=mass,
        nll=math.log(mass),row_sha256='0'*64)for i in range(1,10001)]


class Controls(unittest.TestCase):
    def test_exact_coverage_weighting_and_gate(self):
        original=rows(1.);native=rows(1.)
        for item in native[9216:]:item.update(exp_mass=math.exp(.6),nll=.6)
        result=target.check(original,native,[0]*10001)
        self.assertTrue(result['loss_gate_passed'])
        self.assertAlmostEqual(result['native_minus_original'],.6*784/10000)
        for item in native[9216:]:item.update(exp_mass=math.exp(.7),nll=.7)
        self.assertFalse(target.check(original,native,[0]*10001)['loss_gate_passed'])

    def test_native_block_reset_metadata_and_short_last_block(self):
        records=rows(1.)[1024:]
        def transcript(wrong_reset=False):
            lines=[b'test result: ok.']
            for number in range(1,10):
                count=min(1024,10000-number*1024)
                value=dict(block=number,input_start=number*1024,targets=count,positions=count+1,
                           history_empty=not(wrong_reset and number==9))
                lines.append(b'GEMMA_HYBRID_REAL_LOSS_BLOCK_START '+json.dumps(value).encode())
                lines.append(b'GEMMA_HYBRID_REAL_LOSS_UNSCORED '+json.dumps(dict(block=number,position=count,row_sha256='0'*64)).encode())
            lines.extend(b'GEMMA_HYBRID_REAL_LOSS_TARGET '+json.dumps(r).encode()for r in records)
            lines.append(b'GEMMA_HYBRID_REAL_LOSS_SUMMARY '+json.dumps(dict(targets=8976,capacity=1025,chunk_capacity=32,session_allocated_bytes=271117548,quality_qualified=False,mean_nll=0.)).encode())
            lines.append(b'GEMMA_HYBRID_CAPTURE '+json.dumps(dict(bytes=0,records=0,sha256='e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855')).encode())
            return b'\n'.join(lines)
        self.assertEqual(len(target.remaining_native(transcript(),[0]*10001)),8976)
        with self.assertRaises(ValueError):target.remaining_native(transcript(True),[0]*10001)

    def test_missing_duplicate_target_reset_and_nonfinite_reject(self):
        original=rows(1.)
        for mutation in [lambda r:r.pop(),lambda r:r.__setitem__(1024,r[1023]),
            lambda r:r[1024].update(local_row=1024),lambda r:r[-1].update(target=1),
            lambda r:r[0].update(nll=float('nan'))]:
            native=copy.deepcopy(original);mutation(native)
            with self.assertRaises(ValueError):target.check(original,native,[0]*10001)


if __name__=='__main__':unittest.main()
