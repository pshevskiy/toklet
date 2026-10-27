import copy
import unittest
import prepare_qwen3_context4096 as inputs


class Inputs(unittest.TestCase):
    def test_new_suffix_and_future_control_have_exact_boundaries(self):
        ids=list(range(14097));old=ids[:10001]
        arrays=inputs.choose([9707,11,1879,0]*128,ids,old)
        self.assertEqual(arrays['Q.u32le'],list(range(10001,14097)))
        self.assertEqual(arrays['B.u32le'][:-1],arrays['A.u32le'][:-1])
        self.assertEqual(arrays['B.u32le'][-1],1)
        self.assertEqual(len(arrays['Q.u32le'][3072:4096]),1024)

    def test_clipping_prefix_changes_and_noninteger_ids_reject(self):
        original=[9707,11,1879,0]*128;ids=list(range(14097));old=ids[:10001]
        for bad in (ids[:-1],[1,*ids[1:]],ids[:-1]+[True],ids[:-1]+[151936]):
            with self.assertRaises(ValueError):inputs.choose(original,bad,old)

    def test_target_policy_or_scope_cannot_be_coherently_changed(self):
        good=dict(inputs.REQUIRED,files={name:dict(bytes=16384,sha256='a'*64)
            for name in ('A.u32le','B.u32le','Q.u32le')})
        inputs.validate(good)
        for key,value in [('capacity',4095),('quality_global_token_start',10000),
                          ('score_rows',[3072,4096]),('score_target_indices',[3071,4095]),
                          ('targets',1023),('quality_mean_nll_increase_limit',0.051)]:
            changed=copy.deepcopy(good);changed[key]=value
            with self.assertRaises(ValueError):inputs.validate(changed)


if __name__=='__main__':unittest.main()
