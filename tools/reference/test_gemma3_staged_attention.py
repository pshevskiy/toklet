"""Portable independent window/copy/operator controls; no target/GPU/export."""
import importlib.util
import math
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('gemma3_staged_attention', Path(__file__).with_name('gemma3_staged_attention.py'))
subject = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subject)


class Controls(unittest.TestCase):
    def test_fixed_protocol_minimal_view_and_bounds(self):
        self.assertEqual(subject.POLICY, dict(fatal_reference='main', atol_floor=1e-4, envelope_factor=2, rtol=1e-3))
        self.assertEqual(len(subject.CASES), 8)
        self.assertEqual(sum(c['tokens'] * 4 * c['dim'] for c in subject.CASES), 71424)
        c = subject.CASES[0]
        v = subject.view(c)
        self.assertEqual((v['start'], v['end'], v['retention']), (2, 12, 4))
        self.assertEqual(v['old_spans'], [dict(absolute_start=2, slot=2, rows=2), dict(absolute_start=4, slot=0, rows=1)])
        self.assertEqual(subject.view(subject.CASES[2])['start'], 1)
        self.assertEqual(subject.view(subject.CASES[4])['start'], 2)
        self.assertEqual(subject.view(subject.CASES[3])['start'], 0)
        for mutation in [dict(tokens=0), dict(tokens=33), dict(committed=6), dict(window=0), dict(committed=True), dict(dim=64)]:
            with self.assertRaises(ValueError):
                subject.shape(dict(c, **mutation))

    def test_copy_runs_vs_absolute_reconstruction_and_chain(self):
        previous = {}
        for c in subject.CASES:
            old, staged, _ = subject.operands(c)
            before = {k: v[:] for k, v in old.items()}
            staging_before = {k: v[:] for k, v in staged.items()}
            after = subject.commit_reference(c, old, staged)
            if c['name'].startswith('boundary514'):
                self.assertEqual(old, previous[c['kind']])
            if c['name'].startswith('boundary513'):
                previous[c['kind']] = after
            for kind in ('k', 'v'):
                expected = subject.physical(c['committed'] + c['tokens'], c['window'] or c['capacity'],
                    c['dim'], kind, c['pattern'], c['window'] is not None)
                self.assertEqual(after[kind], expected)
                self.assertEqual(after[kind][-7:], [0x7e55] * 7)
            self.assertEqual(old, before)
            self.assertEqual(staged, staging_before)
        self.assertEqual(subject.copy_plan(subject.CASES[0]), [dict(source_row=3, destination_row=0, rows=4)])
        self.assertEqual(subject.copy_plan(subject.CASES[6]), [dict(source_row=0, destination_row=500, rows=12), dict(source_row=12, destination_row=0, rows=20)])

    def test_small_attention_causal_gqa_and_mask(self):
        c = subject.CASES[0]
        old, staged, q = subject.operands(c)
        original = ({k: v[:] for k, v in old.items()}, {k: v[:] for k, v in staged.items()}, q[:])
        for profile in ('main', 'wide'):
            out = subject.attention(c, old, staged, q, profile)
            self.assertEqual(len(out['context']), 7 * 4 * 32)
            self.assertEqual(len(out['scores']), 4 * 7 * 10)
            self.assertTrue(all(math.isfinite(subject.half_value(x)) for x in out['context']))
            self.assertNotEqual(out['scores'][:70], out['scores'][70:140])
            for head in range(4):
                first = out['probabilities'][head * 70:head * 70 + 10]
                self.assertTrue(all(x > 0 for x in first[:4]))
                self.assertEqual(first[4:], [0.] * 6)
            changed = {k: v[:] for k, v in staged.items()}
            changed['v'][6 * 32:7 * 32] = [subject.half_word(8.)] * 32
            altered = subject.attention(c, old, changed, q, profile)
            self.assertEqual(out['context'][:6 * 4 * 32], altered['context'][:6 * 4 * 32])
            self.assertNotEqual(out['context'][6 * 4 * 32:], altered['context'][6 * 4 * 32:])
        self.assertEqual((old, staged, q), original)

    def test_active_finiteness_and_unused_padding(self):
        c = subject.CASES[0]
        old, staged, q = subject.operands(c)
        subject.decoded_view(c, old, staged, q)  # QuietNaN unused rows/guards are ignored.
        for kind in ('k', 'v'):
            bad = {k: v[:] for k, v in staged.items()}
            bad[kind][0] = 0x7e00
            with self.assertRaises(ValueError):
                subject.attention(c, old, bad, q, 'main')
        bad_q = q[:]
        bad_q[0] = 0x7c00
        with self.assertRaises(ValueError):
            subject.attention(c, old, staged, bad_q, 'main')
        with self.assertRaises(ValueError):
            subject.decoded_view(c, old, staged, q[:-1])
        # Physically retained position1 is already outside the minimal view2..11.
        ignored = {k: v[:] for k, v in old.items()}
        ignored['k'][32:64] = [0x7e00] * 32
        self.assertEqual(subject.attention(c, old, staged, q, 'main'), subject.attention(c, ignored, staged, q, 'main'))

    def test_all_owner_commit_predicate_rejections(self):
        owners = [dict(owner=i, generation=7, committed=5, capacity=12) for i in (1, 2)]
        frames = [dict(owner=i, generation=7, committed=5, tokens=7) for i in (1, 2)]
        before = [item.copy() for item in owners]
        subject.validate_frames(owners, frames)
        for kwargs in [dict(math_ok=False), dict(poisoned=True)]:
            with self.assertRaises(ValueError):
                subject.validate_frames(owners, frames, **kwargs)
        with self.assertRaises(ValueError):
            subject.validate_frames(owners, frames[:1])
        for mutation in [dict(owner=1), dict(owner=3), dict(generation=6), dict(committed=4), dict(tokens=8), dict(tokens=True)]:
            bad = [frames[0].copy(), dict(frames[1], **mutation)]
            with self.assertRaises(ValueError):
                subject.validate_frames(owners, bad)
        with self.assertRaises(ValueError):
            subject.validate_frames([owners[0], owners[0]], frames)
        self.assertEqual(owners, before)


if __name__ == '__main__':
    unittest.main()
