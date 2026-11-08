import copy
import unittest
import check_qwen35_generation_state as c


def rows(cohort=0):
    return [dict(iteration=i,warmup=i<3,cohort=cohort,first_role='generation' if (i+cohort)%2 else 'numerical',
                 inputs=17,position_offset=37,warmed_rust_allocations=0,all_state_logits_bits_exact=True,
                 numerical_ns=100+i,generation_ns=120+i) for i in range(13)]


class Samples(unittest.TestCase):
    def test_exact_native_footer_survives_test_harness_prefix(self):
        tag=c.TAGS['tiny']; prefix='test '+c.TESTS['tiny']+' ... '
        self.assertEqual(c.labels(prefix+tag+' {"ok":true}\n',tag),[dict(ok=True)])
        self.assertEqual(c.labels('test wrong ... '+tag+' {"ok":true}\n',tag),[])

    def test_complete_cohort_excludes_three_warmups_and_keeps_raw_samples(self):
        result=c.samples(rows(),0)
        self.assertEqual(result['numerical_ns']['samples_ns'],list(range(103,113)))
        self.assertEqual(result['generation_ns']['median_ns'],127.5)
        self.assertEqual(result['generation_ns']['mad_ns'],2.5)

    def test_rejects_missing_extra_or_reordered_pairs(self):
        valid=rows()
        for altered in [valid[:-1],valid+[valid[-1]],list(reversed(valid))]:
            with self.assertRaises(ValueError):c.samples(altered,0)

    def test_typed_work_and_completed_durations_are_not_booleans(self):
        for field,value in [('inputs',True),('warmup',0),('numerical_ns',True),('generation_ns',0),
                            ('warmed_rust_allocations',False),('all_state_logits_bits_exact',1),('position_offset',0)]:
            with self.subTest(field=field):
                altered=rows();altered[6][field]=value
                with self.assertRaises(ValueError):c.samples(altered,0)

    def test_confirmation_reverses_order_and_requires_cohort_identity(self):
        self.assertEqual(len(c.samples(rows(1),1)['numerical_ns']['samples_ns']),10)
        with self.assertRaises(ValueError):c.samples(rows(0),1)
        altered=rows(1);altered[4]['first_role']='numerical'
        with self.assertRaises(ValueError):c.samples(altered,1)


class Owners(unittest.TestCase):
    def setUp(self):
        self.expected=dict(numerical_memory=dict(admitted_bytes=100,weights_and_model_bytes=40),
                           pending_history_bytes=24,wrapper_metadata_bytes=8)
        self.report=dict(memory=dict(numerical=self.expected['numerical_memory'].copy(),pending_history_bytes=24,
                                     wrapper_metadata_bytes=8,admitted_bytes=132),
                         combined_admitted_bytes=134217920,diagnostic_allowance_bytes=134217728)

    def test_shared_model_is_counted_once_and_extra_owners_remain_explicit(self):
        self.assertEqual(c.memory(self.report,self.expected),134217920)
        for field in ['combined_admitted_bytes','diagnostic_allowance_bytes']:
            altered=copy.deepcopy(self.report);altered[field]+=40
            with self.assertRaises(ValueError):c.memory(altered,self.expected)

    def test_rejects_bool_or_missing_pending_history_and_hidden_metadata(self):
        for field,value in [('pending_history_bytes',True),('wrapper_metadata_bytes',0),('admitted_bytes',100)]:
            altered=copy.deepcopy(self.report);altered['memory'][field]=value
            with self.assertRaises(ValueError):c.memory(altered,self.expected)

    def test_all_simultaneous_owners_must_fit_ceiling(self):
        self.expected['numerical_memory']['admitted_bytes']=c.LIMIT
        self.report['memory']['numerical']['admitted_bytes']=c.LIMIT
        self.report['memory']['admitted_bytes']=c.LIMIT+32
        self.report['combined_admitted_bytes']=2*c.LIMIT+32-40+134217728
        with self.assertRaises(ValueError):c.memory(self.report,self.expected)


if __name__ == '__main__':unittest.main()
