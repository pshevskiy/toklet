"""Explicit diagnostic time bounds and inherited owned failure cleanup; no GPU."""
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import m6_context_long_budget as long
from test_m6_context import admission


class LongBudget(unittest.TestCase):
    def test_only_explicit_bounded_time_choices_are_accepted_before_launch(self):
        for value in (7200, 21600, 43200): self.assertEqual(long.budget(value), value)
        for value in (True, 0, -1, 7200., float('nan'), 86400, 43201):
            with patch.object(long.subprocess, 'Popen') as launch, self.assertRaises(ValueError):
                long.Worker(Path('bin'), Path('source'), None, 4096, 128, False, 'small', None, None,
                            lambda: None, request_timeout=value)
            launch.assert_not_called()

    def test_configuration_is_passed_to_bounded_transport_and_recorded(self):
        for seconds in long.BUDGETS:
            child, transport = Mock(), Mock(); transport.read.return_value = admission()
            w = long.Worker(Path('bin'), Path('source'), None, 4096, 128, False, 'small', io.StringIO(), io.BytesIO(),
                            lambda: None, request_timeout=seconds)
            with patch.object(long.subprocess, 'Popen', return_value=child),\
                 patch.object(long.base.shared.metal_engine, 'Transport', return_value=transport) as create,\
                 patch.object(w, 'usage') as usage,\
                 patch.object(long.base.shared.metal_engine, 'kill_owned_group') as kill:
                with self.assertRaisesRegex(RuntimeError, 'body failure'):
                    with w:
                        self.assertEqual(w.record['event_read_timeout_seconds'], seconds)
                        self.assertEqual(w.record['lifetime_timeout_seconds'], 86400)
                        raise RuntimeError('body failure')
                self.assertEqual(create.call_args.kwargs['request_timeout'], seconds)
                self.assertEqual(create.call_args.kwargs['lifetime_timeout'], 86400)
                usage.assert_called_once_with('ready'); kill.assert_called_once_with(child)
                self.assertEqual(w.record['error'], 'RuntimeError: body failure')
                self.assertFalse(w.record['passed']); transport.close.assert_called_once()

    def test_prelaunch_reassertion_still_precedes_popen(self):
        def reject(): raise ValueError('changed source')
        w = long.Worker(Path('bin'), Path('source'), None, 4096, 128, False, 'small', None, None,
                        reject, request_timeout=21600)
        with patch.object(long.subprocess, 'Popen') as launch, self.assertRaisesRegex(ValueError, 'changed source'):
            with w: pass
        launch.assert_not_called()

    def test_unknown_model_plan_depth_and_existing_evidence_fail_before_admission(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'out.json'
            for model, plan, depth in [('other', 'quality', 4096), ('small', 'matrix', 4096), ('small', 'quality', True), ('small', 'quality', 4097)]:
                with patch.object(long.base, 'compiled_admission') as compile_check, self.assertRaises(ValueError):
                    long.run(output, model, plan, depth)
                compile_check.assert_not_called()
            output.write_text('preserve')
            with patch.object(long.base, 'compiled_admission') as compile_check, self.assertRaises(ValueError):
                long.run(output, 'small', 'quality', 4096)
            compile_check.assert_not_called(); self.assertEqual(output.read_text(), 'preserve')


if __name__ == '__main__': unittest.main()
