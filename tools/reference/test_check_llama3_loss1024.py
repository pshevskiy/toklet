"""Synthetic statistics/coverage/primary-gate controls; no model or corpus."""
import copy
import math
import unittest
import check_llama3_loss1024 as check


def record(index,label,native=False):
    block,row=divmod(index-1,256)
    result=dict(block=block,local_row=row,global_target_index=index,target=0,target_fp32_bits=0,
                vocab=check.VOCAB,greedy_lowest_id=0,row_sha256='0'*64,maximum=0.,exp_mass=check.VOCAB,nll=math.log(check.VOCAB))
    if native:result.update(kind='target',chunk=label,numeric_row_sha256='1'*64)
    else:result['profile']=label
    return result


class Checker(unittest.TestCase):
    def test_statistics_require_target_binding_finite_stable_formula_and_integer_indices(self):
        item=record(1,'wide');self.assertEqual(check.statistics(item,0),math.log(check.VOCAB))
        for key,value in [('target',1),('vocab',3),('target_fp32_bits',0x7f800000),('maximum',float('nan')),
                          ('exp_mass',0),('nll',0.),('greedy_lowest_id',1),('block',False),('local_row',0.)]:
            changed=item.copy();changed[key]=value
            with self.assertRaises((ValueError,OverflowError)):check.statistics(changed,0)
    def test_primary_gate_is_exact_and_independent_of_statistics_coherence_allowance(self):
        self.assertEqual(check.paired_delta([.05]*1024,[0.]*1024),.05)
        self.assertTrue(check.paired_delta([.05]*1024,[0.]*1024)<=check.LIMIT)
        self.assertFalse(check.paired_delta([math.nextafter(.05,math.inf)]*1024,[0.]*1024)<=check.LIMIT)
        for bad in [[0.]*1023,[float('nan')]*1024]:
            with self.assertRaises(ValueError):check.paired_delta(bad,[0.]*1024)
    def test_complete_unique_source_coverage_and_native_numeric_hash_controls(self):
        ids=[0]*1025
        source=[record(i,label)for label in ['main','wide']for i in range(1,1025)]
        values=check.collect(source,ids);self.assertEqual(len(values['wide']),1024)
        with self.assertRaises(ValueError):check.collect(source[:-1],ids)
        bad=copy.deepcopy(source);bad[-1]=bad[-2].copy()
        with self.assertRaises(ValueError):check.collect(bad,ids)
        native=[]
        for chunk in [32,128]:
            native.extend(record(i,chunk,True)for i in range(1,1025))
            native.extend(dict(kind='unscored',chunk=chunk,block=b,local_row=256,row_sha256='2'*64,numeric_row_sha256='3'*64)for b in range(4))
        check.collect(native,ids,True)
        # Raw signed-zero-sensitive hashes need not agree; canonical numeric
        # hashes must. Cross-profile source hashes are never equality gates.
        native[1028]['row_sha256']='4'*64;check.collect(native,ids,True)
        native[1028]['numeric_row_sha256']='5'*64
        with self.assertRaisesRegex(ValueError,'numeric row'):check.collect(native,ids,True)
        native[1028]['numeric_row_sha256']='1'*64
        native[-1]['numeric_row_sha256']='6'*64
        with self.assertRaisesRegex(ValueError,'numeric row'):check.collect(native,ids,True)

if __name__=='__main__':unittest.main()
