"""Context17 plan/inventory controls; original one-input arithmetic stays frozen."""
import copy
import unittest
import qwen35_9b_reference as producer
import qwen35_9b_weights as weights
import qwen35_cpu_graph as graph

class Controls(unittest.TestCase):
    def test_frozen_workload_offset_and_all_capture_extents(self):
        self.assertEqual(producer.plan('one'),([1],0,16660224))
        self.assertEqual(producer.plan('context17'),(list(graph.TOKENS),37,16926464))
        self.assertEqual(len(graph.TOKENS),17)
        self.assertEqual(16926464*4,67705856)
        self.assertEqual(16926464*12,203117568)
        self.assertLess(203117568,producer.MAX_CAPTURE)
        for name in ['context18','context128','context256','',None,17]:
            with self.assertRaises(ValueError):producer.plan(name)

    def test_only_committed_kv_and_attention_extents_grow(self):
        old=weights.expected_native();new=weights.expected_native(17)
        self.assertEqual(len(old),1051);self.assertEqual(set(old),set(new))
        growing={'attention_k_cache','attention_v_cache','attention_scores','attention_probabilities'}
        for key,count in old.items():self.assertEqual(new[key],count*(17 if key[1] in growing else 1))
        self.assertEqual(sum(new.values()),16926464)
        self.assertEqual(sum(new.values())-sum(old.values()),266240)
        self.assertEqual(new[3,'attention_k_cache',0],4352)
        self.assertEqual(new[31,'attention_probabilities',15],17)
        for n in [True,False,0,-1,513,17.,None]:
            with self.assertRaises(ValueError):weights.expected_native(n)

    def test_inventory_is_fresh_and_original_gates_do_not_change(self):
        value=weights.expected_native(17);value.clear()
        self.assertEqual(len(weights.expected_native(17)),1051)
        self.assertEqual(graph.GATE,dict(atol=1e-5,rtol=1e-4))
        self.assertEqual(producer.ARRAYS,1051)
        self.assertEqual(producer.VALUES,16660224)
        self.assertLess(sum(producer.LEDGER.values()),6*1024**3)

if __name__=='__main__':unittest.main()
