import unittest
import check_qwen35_context as context
import check_qwen35_real_reference as prior


class Controls(unittest.TestCase):
    def test_complete_context_geometry_includes_all_retained_heads(self):
        shapes=prior.expected(17)
        self.assertEqual((len(shapes),sum(shapes.values())),(1051,16333568))
        self.assertEqual(shapes[3,'attention_k_cache',3],17*256)
        self.assertEqual(shapes[31,'attention_probabilities',15],17)
        for value in [True,0,-1,513,17.0]:
            with self.assertRaises(ValueError): prior.expected(value)

    def test_final_token_is_separate_from_absolute_rotary_position(self):
        row=dict(token=16,layer=0,stage='delta_state')
        self.assertEqual(prior.key(row,16),(0,'delta_state',None))
        with self.assertRaises(ValueError): prior.key({**row,'token':53},16)

    def test_work_denominators_remain_per_successful_call(self):
        self.assertEqual(context.work(17)['projections_completed'],4233)
        self.assertEqual(context.work(12)['tokens_committed'],12)
        self.assertFalse(prior.exact(context.work(17),context.work(12)))

    def test_boundary_rejection_counters_are_distinct_from_prior_retry(self):
        final=context.work(0,'Error',0)
        self.assertEqual(final['outcome'],'Error')
        self.assertEqual(sum(v for k,v in final.items() if k!='outcome'),0)
        self.assertFalse(prior.exact(final,context.work(12)))

    def test_duplicate_or_nonfinite_log_events_reject(self):
        for raw in ['{"token":16,"token":0}','{"x":NaN}','{"x":1e999}']:
            with self.assertRaises(ValueError): prior.decode(raw)


if __name__=='__main__': unittest.main()
