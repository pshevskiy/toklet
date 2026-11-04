import copy
import math
import struct
import unittest
import gemma_mlx_decisions as subject


class Controls(unittest.TestCase):
    def fixture(self):
        cases = [dict(id=str(i), category='code' if i < 40 else 'prose' if i < 70 else 'continuation', prompt_tokens=[2, 9]) for i in range(100)]
        original = [dict(id=c['id'], category=c['category'], generation=dict(samples=[dict(token_id=t) for t in [7, 8, 106]])) for c in cases]
        native = [dict(id=c['id'], category=c['category'], visible_token_ids=[7, 10], stop_reason=dict(kind='stop_token', token_id=106)) for c in cases]
        outcomes = {role: [dict(id=c['id'], category=c['category'], passed=False) for c in cases] for role in ['baseline', 'native']}
        for role, indices in [('baseline', [1, 3, 40, 42]), ('native', [24, 30, 43, 45])]:
            for index in indices:
                outcomes[role][index]['passed'] = True
        return cases, outcomes, original, native

    def test_selection_rule_and_exact_shared_prefix(self):
        selected = subject.select(*self.fixture())
        self.assertEqual([c['index'] for c in selected], [1, 24, 40, 43])
        for case in selected:
            self.assertEqual(case['agreed_ids'], [7])
            self.assertEqual((case['decision_index'], case['original_id'], case['native_id']), (1, 8, 10))
        cases, outcomes, original, native = self.fixture()
        native[1]['visible_token_ids'] = [7, 8]
        with self.assertRaises(ValueError):
            subject.select(cases, outcomes, original, native)

    def test_missing_side_invalid_id_and_case_swap_reject(self):
        for mode in ['side', 'id', 'order']:
            values = copy.deepcopy(self.fixture())
            if mode == 'side':
                for row in values[1]['native'][:40]:
                    row['passed'] = False
            elif mode == 'id':
                values[2][1]['generation']['samples'][1]['token_id'] = 262144
            else:
                values[3][0], values[3][1] = values[3][1], values[3][0]
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                subject.select(*values)

    def test_full_row_ties_candidates_hash_and_corruption(self):
        raw = struct.pack('<4f', -1., 2., 2., .5)
        result = subject.statistics(raw, 2, 1, 1, 4)
        self.assertEqual(result['top_two_ids'], [1, 2])
        self.assertEqual(result['top_two_margin'], 0)
        self.assertEqual(result['agrees_with'], 'native')
        for values, greedy in [([0., math.nan, 1., 2.], 3), ([-1., 2., 2., .5], 2)]:
            with self.assertRaises(ValueError):
                subject.statistics(struct.pack('<4f', *values), 2, 1, greedy, 4)

    def test_selected_raw_inventory_excludes_shared_consumer_kv(self):
        inventory = subject.inventory()
        self.assertEqual(len(inventory), 1096)
        self.assertEqual(inventory['language_model.model.embed_tokens_per_layer.weight'], dict(shape=[262144, 1120], dtype='U32'))
        self.assertIn('language_model.model.layers.14.self_attn.k_norm.weight', inventory)
        self.assertNotIn('language_model.model.layers.15.self_attn.k_norm.weight', inventory)
