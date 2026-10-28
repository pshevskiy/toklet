"""Proposed synthetic stable-loss controls; no real weights or model execution."""
import math
import unittest
import numpy as np
from llama3_loss1024 import row_statistics


class StableLoss(unittest.TestCase):
    def test_uniform_tie_signed_zero_and_target_bits(self):
        result=row_statistics(np.array([-0.,0.,0.],dtype=np.float32),0)
        self.assertEqual(result['nll'],math.log(3));self.assertEqual(result['exp_mass'],3)
        self.assertEqual(result['greedy_lowest_id'],0);self.assertEqual(result['target_fp32_bits'],0x80000000)

    def test_large_common_offset_does_not_destroy_logmass(self):
        row=np.array([1e30,1e30,-1e30],dtype=np.float32)
        a=row_statistics(row,0);self.assertEqual(a['nll'],math.log(2))
        b=row_statistics(row,2)
        self.assertEqual(b['nll'],float(row[0])-float(row[2])+math.log(2))
        self.assertEqual(a['row_sha256'],b['row_sha256'])

    def test_nonuniform_scalar_reference_and_invalid_rows(self):
        row=np.array([-3.,2.,.5],dtype=np.float32);got=row_statistics(row,2)
        mass=math.fsum([math.exp(-5),1.,math.exp(-1.5)])
        self.assertEqual(got['exp_mass'],mass);self.assertEqual(got['nll'],1.5+math.log(mass))
        for value,target in [(row.astype(np.float64),0),(row.reshape(1,3),0),(row,True),(row,-1),(row,3),
                             (np.array([np.inf],dtype=np.float32),0),(np.array([np.nan],dtype=np.float32),0)]:
            with self.assertRaises(ValueError):row_statistics(value,target)


if __name__=='__main__':unittest.main()
