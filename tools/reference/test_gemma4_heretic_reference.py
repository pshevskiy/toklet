"""Portable capture planning and position selection for the explicit oracle."""
import math
import unittest
import gemma4_heretic_reference as reference


class HereticReferenceTests(unittest.TestCase):
    def test_named_graph_has_bounded_last_position_capture(self):
        shapes = reference.shapes()
        self.assertEqual(len(shapes), 713)
        self.assertEqual(shapes['logits'], (1, 1, 262144))
        self.assertEqual(shapes['head'], (1, 1, 262144))
        for name, shape in shapes.items():
            with self.subTest(name=name):
                self.assertEqual(shape[reference.position_axis(name)], 1 if name in ['head', 'logits'] else 15)
        planned = sum(math.prod(shape) // shape[reference.position_axis(name)] * 12
                      for name, shape in shapes.items())
        self.assertGreater(planned, 262144 * 24)
        self.assertLessEqual(planned, reference.MAX_CAPTURE)

    def test_head_is_selected_after_context_and_position_axes_are_distinct(self):
        self.assertEqual(reference.position_axis('layer.0.q_rotary'), 2)
        self.assertEqual(reference.position_axis('layer.0.context'), 2)
        self.assertEqual(reference.position_axis('layer.0.self_attn.v_norm'), 1)
        self.assertEqual(reference.position_axis('ple_projection'), 1)
        self.assertEqual(reference.TOKENS, [2, 105, 2364, 107, 1567, 506, 5279, 529, 7001, 236761, 106, 107, 105, 4368, 107])

    def test_named_contexts_retain_frozen_ids_and_never_truncate(self):
        full, selected = reference.inputs('full512')
        self.assertEqual((len(full), selected), (512, [0, 31, 255, 511]))
        for phase, count in [('compat32', 32), ('pilot64', 64)]:
            ids, positions = reference.inputs(phase)
            self.assertEqual(ids, full[:count])
            self.assertTrue(all(0 <= p < count for p in positions))
        with self.assertRaises(ValueError):
            reference.inputs('truncate1024')
        for counts in [(0, 1), (513, 1), (512, 0), (512, 9), (True, 1)]:
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                reference.shapes(*counts)

    def test_full_context_stage_shapes_preserve_positions_and_selected_head(self):
        shapes = reference.shapes(512, 4)
        self.assertEqual(shapes['logits'], (1, 4, 262144))
        self.assertEqual(shapes['layer.0.q_rotary'], (1, 8, 512, 256))
        self.assertEqual(shapes['layer.4.context'], (1, 8, 512, 512))
        self.assertEqual(shapes['ple_projection'], (1, 512, 8960))


if __name__ == '__main__':
    unittest.main()
