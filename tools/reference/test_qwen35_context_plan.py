import unittest
import qwen35_real_reference as reference
import qwen35_cpu_graph as graph


class Controls(unittest.TestCase):
    def test_prior_point_inputs_and_size_remain_exact(self):
        self.assertEqual(reference.plan('one'), ([1], 0, 16067328))

    def test_context_inputs_offset_last_capture_and_bound_are_frozen(self):
        tokens, offset, values = reference.plan('context17')
        self.assertEqual(tokens, [1,7,13,3,9,5,11,2,17,4,6,10,19,8,21,23,29])
        self.assertEqual((offset, len(tokens)-1, values, values*12), (37, 16, 16333568, 196002816))
        self.assertLess(values*12, reference.MAX_CAPTURE)
        tokens[0] = 0
        self.assertEqual(reference.plan('context17')[0][0], 1)

    def test_unknown_phases_reject_before_allocation(self):
        for phase in ['context512', '', True, 17]:
            with self.assertRaises(ValueError): reference.plan(phase)


if __name__ == '__main__': unittest.main()
