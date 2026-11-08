import unittest
import check_qwen35_graph_capture as checker


class Controls(unittest.TestCase):
    def test_actual_owner_and_stage_coverage_is_concrete(self):
        expected=checker.expected();self.assertEqual(len(expected),1051)
        self.assertEqual(sum(stage=='delta_state' for _,stage,_ in expected),24)
        self.assertEqual(sum(stage=='attention_probabilities' for _,stage,_ in expected),128)
        self.assertEqual(expected[0,'delta_state',None],32*128*128)
        self.assertEqual(expected[3,'attention_q_gate',None],8192)
        self.assertEqual(expected[31,'attention_k_cache',3],256)

    def test_strict_typed_coordinates_reject_booleans(self):
        row=dict(token=0,position=0,count=256,offset=0,bytes=1024,layer=3,stage='attention_k_cache',head=0)
        self.assertEqual(checker.key(row),(3,'attention_k_cache',0))
        for field in ['token','position','count','offset','bytes','layer','head']:
            bad=dict(row);bad[field]=False
            with self.assertRaises(ValueError):checker.key(bad)

    def test_wrong_position_and_stage_type_reject(self):
        row=dict(token=0,position=0,count=1,offset=0,bytes=4,layer=None,stage='logits',head=None)
        for field,value in [('position',1),('token',1),('stage',True),('head','0'),('count',-1)]:
            with self.assertRaises(ValueError):checker.key(dict(row,**{field:value}))


if __name__=='__main__':unittest.main()
