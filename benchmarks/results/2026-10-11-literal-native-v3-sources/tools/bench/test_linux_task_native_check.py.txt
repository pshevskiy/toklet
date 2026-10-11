"""Offline mutations for precise authored native expectations."""
import copy
import hashlib
import unittest

import linux_task_native_cases as cases
import linux_task_native_check as checker
import test_linux_task_evidence as fixtures


class NativeExpectationTests(unittest.TestCase):
    def case(self, suffix):
        return next(case for case in cases.cases() if case['payload']['case_id'].endswith('/' + suffix))

    def receipt(self, case, events, code=0, stderr=b'', **flags):
        record = fixtures.capture(events, code=code, stderr=stderr, **flags)
        data = checker.evidence.serialized_payload(case['payload'])
        record.update(input_bytes=len(data), input_sent_bytes=len(data),
                      input_sha256=hashlib.sha256(data).hexdigest())
        return record

    def ready(self, case):
        event = fixtures.ready()
        event['binding'] = checker.evidence.child.admit_payload(case['payload'])
        return event

    def terminal(self, case, outcome='passed', exception=None):
        event = fixtures.terminal(outcome, exception)
        event['binding'] = checker.evidence.child.admit_payload(case['payload'])
        return event

    def check(self, record, case):
        return checker.check(record, case, fixtures.RUNTIME, fixtures.PYTHON)

    def test_inventory_has_precise_expectations(self):
        inventory = cases.cases()
        self.assertEqual(len(inventory), 28)
        self.assertEqual(len({case['payload']['case_id'] for case in inventory}), 28)
        for case in inventory:
            checker.evidence.child.admit_payload(case['payload'])
            expected = case['expectation']
            for key in ['outcome', 'normal_completion', 'terminal_event', 'container_exit_codes',
                        'wall_timed_out', 'output_limit_exceeded', 'oom_killed',
                        'process_group_killed', 'container_forced_kill',
                        'unframed_stderr_bytes', 'admission_only']:
                self.assertIn(key, expected)

    def test_expected_ambiguous_exits_require_only_admission(self):
        for suffix, code in [('raw_exit137_ambiguous', 137), ('cpu_exit_ambiguous', 152)]:
            case = self.case(suffix)
            ready = self.ready(case)
            good = self.receipt(case, [ready], code=code)
            self.assertEqual(self.check(good, case)['outcome'], 'infrastructure_failure')
            frame = {'event': 'program_output', 'channel': 'stdout', 'text': 'unexpected'}
            for record in [self.receipt(case, [ready], code=code, stderr=b'unexpected'),
                           self.receipt(case, [ready, frame], code=code),
                           self.receipt(case, [ready, self.terminal(case)], code=code)]:
                with self.subTest(suffix=suffix), self.assertRaises(ValueError):
                    self.check(record, case)

    def test_sleeping_wall_stop_rejects_stray_output(self):
        case = self.case('wall_watchdog')
        ready = self.ready(case)
        flags = {'wall_timed_out': True, 'process_group_killed': True, 'transport_exit_code': -9}
        self.assertEqual(self.check(self.receipt(case, [ready], code=137, **flags), case)['reason'], 'wall_limit')
        frame = {'event': 'program_output', 'channel': 'stdout', 'text': 'unexpected'}
        for record in [self.receipt(case, [ready], code=137, stderr=b'unexpected', **flags),
                       self.receipt(case, [ready, frame], code=137, **flags)]:
            with self.assertRaises(ValueError):
                self.check(record, case)

    def test_wrong_exception_does_not_qualify_assertion_failure(self):
        case = self.case('assertion_fails')
        events = [self.ready(case), self.terminal(case, 'model_failure', 'AssertionError')]
        self.assertEqual(self.check(self.receipt(case, events), case)['reason'], 'AssertionError')
        events[-1]['exception'] = 'SyntaxError'
        with self.assertRaises(ValueError):
            self.check(self.receipt(case, events), case)

    def test_output_text_and_channel_are_part_of_qualification(self):
        for suffix, outcome, exception in [('ordinary_output', 'passed', None),
                                            ('status_text_is_output', 'model_failure', 'AssertionError')]:
            case = self.case(suffix)
            frame = {'event': 'program_output', 'channel': 'stdout',
                     'text': case['expectation']['program_output']['stdout']}
            events = [self.ready(case), frame, self.terminal(case, outcome, exception)]
            self.assertEqual(self.check(self.receipt(case, events), case)['outcome'], outcome)
            for changed in (None, {**frame, 'channel': 'stderr'}, {**frame, 'text': 'different'}):
                mutated = [events[0], events[2]] if changed is None else [events[0], changed, events[2]]
                with self.subTest(suffix=suffix, changed=changed), self.assertRaises(ValueError):
                    self.check(self.receipt(case, mutated), case)

    def test_setup_or_forced_kill_does_not_qualify_ambiguous_exit(self):
        case = self.case('raw_exit137_ambiguous')
        record = self.receipt(case, [self.ready(case)], code=137)
        for update in [{'container_forced_kill': True}, {'process_group_killed': True},
                       {'failure': {'type': 'ValueError', 'message': 'authored failure'}}]:
            altered = copy.deepcopy(record)
            altered.update(update)
            with self.subTest(update=update), self.assertRaises(ValueError):
                self.check(altered, case)

    def test_malformed_terminal_requires_exact_observed_sequence(self):
        case = self.case('malformed_raw_terminal')
        malformed = {'event': 'program_result', 'outcome': 'passed'}
        events = [self.ready(case), malformed, self.terminal(case)]
        self.assertEqual(self.check(self.receipt(case, events), case)['outcome'], 'infrastructure_failure')
        for key in ('normal_completion', 'program_bytes'):
            altered = copy.deepcopy(events)
            if key == 'normal_completion':
                altered[-1][key] = 1
            else:
                altered[-1]['binding'][key] = float(altered[-1]['binding'][key])
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.check(self.receipt(case, altered), case)
        events.append({'event': 'program_output', 'channel': 'stdout', 'text': 'unrelated'})
        with self.assertRaises(ValueError):
            self.check(self.receipt(case, events), case)


if __name__ == '__main__':
    unittest.main()
