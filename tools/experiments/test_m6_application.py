"""Max-context CLI receipt policy, including published special IDs; no GPU."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import m6_application as app
import test_m6_tasks as tasks


def fixture():
    ready = tasks.ready(); ready['context_capacity'] = 32768
    prompt = [1, 2, 3]; cap = 2
    generation = dict(reused_tokens=0, prefill_tokens=3, sampled_tokens=2, emitted_tokens=2,
                      history_tokens=5, evaluated_tokens=4, pending_tokens=1, reason={'kind': 'output_limit'},
                      prefill_seconds=.1, decode_seconds=.1, sampling_seconds=.001, total_seconds=.3,
                      first_token_seconds=.1)
    stats = {k: v for k, v in ready.items() if k not in ('event', 'context_capacity')}
    stats['generation'] = generation
    stats.update({k: 0. for k in ('input_preparation_seconds', 'input_formatting_seconds', 'tokenization_seconds',
                                  'detokenization_seconds', 'publication_seconds', 'load_seconds', 'conversion_seconds', 'tokenizer_load_seconds')})
    events = [dict(event='token', index=0, token_id=151645, elapsed_seconds=.1, text=''),
              dict(event='token', index=1, token_id=5, elapsed_seconds=.2, text='A'), dict(event='done', statistics=stats)]
    return ready, prompt, cap, events, [dict(event='text', text='B'), dict(event='reset')]


class Application(unittest.TestCase):
    def test_no_eos_boundary_keeps_every_published_id_and_finalized_text(self):
        ready, prompt, cap, events, reset = fixture()
        value = app.response(events, reset, ready, prompt, cap)
        self.assertEqual(value['visible_token_ids'], [151645, 5])
        self.assertEqual(value['visible_text'], 'AB')
        self.assertEqual(value['history_sha256'], app.context.shared.history_sha(prompt+[151645, 5]))

    def test_token_order_completion_reset_timing_and_state_mutations_reject(self):
        for mutate in (lambda e: e.pop(), lambda e: e[0].update(index=True),
                       lambda e: e[0].update(token_id=151936), lambda e: e[0].update(text=1),
                       lambda e: e[1].update(elapsed_seconds=.01),
                       lambda e: e[-1]['statistics']['generation'].update(pending_tokens=0),
                       lambda e: e[-1]['statistics']['generation'].update(reason={'kind': 'stop_token', 'token_id': 151645}),
                       lambda e: e[-1]['statistics']['generation'].update(sampled_tokens=True),
                       lambda e: e[-1]['statistics'].update(tokenization_seconds=.001),
                       lambda e: e[-1]['statistics'].update(load_seconds=float('nan'))):
            ready, prompt, cap, events, reset = fixture(); mutate(events)
            with self.assertRaises(ValueError): app.response(events, reset, ready, prompt, cap)
        for reset in ([], [dict(event='text', text='x')], [dict(event='error')], [dict(event='text', text=True), dict(event='reset')]):
            ready, prompt, cap, events, _ = fixture()
            with self.assertRaises(ValueError): app.response(events, reset, ready, prompt, cap)

    def test_existing_evidence_fails_before_model_admission(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'out.json'; output.write_text('preserve')
            with patch.object(app.producer, 'provenance') as admit, self.assertRaises(ValueError): app.run(output, 'small')
            admit.assert_not_called(); self.assertEqual(output.read_text(), 'preserve')

    def test_startup_reassertion_precedes_native_launch(self):
        def reject(): raise ValueError('changed')
        w = app.Worker(Path('bin'), Path('source'), None, 32768, 128, False, 'small', None, None, reject)
        with patch.object(app.subprocess, 'Popen') as launch, self.assertRaisesRegex(ValueError, 'changed'):
            with w: pass
        launch.assert_not_called()


if __name__ == '__main__': unittest.main()
