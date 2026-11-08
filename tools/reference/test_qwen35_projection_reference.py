import copy
import unittest
from unittest.mock import patch
import qwen35_projection_reference as reference


class OperandTests(unittest.TestCase):
    def test_all_original_matrix_roles_resolve(self):
        matrices = {n: shape for n, (kind, shape) in reference.graph.specs(reference.weights.GEOMETRY).items() if kind == 'matrix'}
        self.assertEqual(len(matrices), 249)
        self.assertEqual(sum(s[0] for s in matrices.values()), 1380352)
        roles = [reference.operand(n) for n in matrices]
        self.assertEqual(roles.count((None, 'final_norm')), 1)
        for layer in range(32):
            self.assertEqual(roles.count((layer,'input_norm')), 3 if layer % 4 == 3 else 4)
            self.assertEqual(roles.count((layer,'post_norm')), 2)
            self.assertEqual(roles.count((layer,'mlp_activated')), 1)
            self.assertEqual(roles.count((layer,'attention_gated' if layer % 4 == 3 else 'delta_gated')), 1)

    def test_reject_unknown_roles_or_outside_layer(self):
        for name in ['lm_head.weight','language_model.model.layers.32.mlp.up_proj.weight',
                     'language_model.model.layers.0.self_attn.unknown.weight','language_model.model.layersX0.mlp.up_proj.weight']:
            with self.subTest(name=name), self.assertRaises(ValueError): reference.operand(name)

    def test_closed_operand_receipt_is_required(self):
        valid = dict(passed=True,exit_code=0,group_absent=True,error=None,changed_inputs=[],files_sha256={})
        for key, value in [('passed',False),('exit_code',True),('group_absent',False),('error','failed'),('changed_inputs',['changed'])]:
            receipt = copy.deepcopy(valid); receipt[key] = value
            with self.subTest(key=key), patch.object(reference,'document',return_value=receipt), self.assertRaises(ValueError):
                reference.operands(reference.ROOT/'unused')

    def test_closed_receipt_cannot_omit_files(self):
        with patch.object(reference,'document',return_value=dict(passed=True,exit_code=0,group_absent=True,error=None,changed_inputs=[],files_sha256={})), self.assertRaisesRegex(ValueError,'seals'):
            reference.operands(reference.ROOT/'unused')


if __name__ == '__main__': unittest.main()
