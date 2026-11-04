import copy
import unittest
import check_gemma_projection_environment as subject


def events():
    result = [('META', dict(device='fixture', simd_width=32, math_mode='Safe', functions='Precise',
                            language='2.3', cases=6, coupled_graph_qualified=False))]
    for name, weight, first in subject.CASES:
        inputs = first+[0.]*62
        value = subject.bits(subject.rounded(sum((subject.Fraction(weight)*subject.Fraction(v) for v in inputs), subject.Fraction())))
        for repeat in range(2):
            output = [value]
            buffers = dict(codes=[0]*8+[subject.SENTINEL]*7, scales=[0]+[subject.SENTINEL]*7,
                biases=[subject.bits(weight) >> 16]+[subject.SENTINEL]*7,
                input=[subject.bits(v) for v in inputs]+[subject.SENTINEL]*7,
                output=output+[subject.SENTINEL]*7, status=[0]+[subject.SENTINEL]*7)
            result.append(('CASE', dict(name=name, repeat=repeat, completed=True, error=None,
                weight_bits=subject.bits(weight), input_bits=[subject.bits(v) for v in inputs],
                output_bits=output, device=buffers, owned_bytes=700)))
    return result+[('COMPLETE', dict(cases=6, repeats=2, coupled_graph_qualified=False))]


class Controls(unittest.TestCase):
    def test_ieee_and_zero_observations_do_not_qualify_flush_model(self):
        raw = events()
        self.assertTrue(subject.verify(raw)['passed'])
        for index in range(1, 9):
            raw[index][1]['output_bits'] = [0]
            raw[index][1]['device']['output'][0] = 0
        result = subject.verify(raw)
        self.assertEqual([v['observation'] for v in result['observations'][:4]], ['zero_output']*4)
        self.assertFalse(result['general_flush_semantics_qualified'])

    def test_coherent_residual_corruption_missing_row_and_guard_reject(self):
        original = events()
        for change in ['residual', 'missing', 'guard', 'width', 'operand']:
            raw = copy.deepcopy(original)
            if change == 'residual':
                for index in [9, 10]:
                    raw[index][1]['output_bits'] = [subject.bits(2.**-23)]
                    raw[index][1]['device']['output'][0] = subject.bits(2.**-23)
            elif change == 'missing':
                raw.pop()
            elif change == 'guard':
                raw[1][1]['device']['input'][-1] = 0
            elif change == 'width':
                raw[0][1]['simd_width'] = 3
            else:
                raw[1][1]['input_bits'][0] = 0
            with self.subTest(change=change), self.assertRaises(ValueError):
                subject.verify(raw)
