import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import linux_process_cpu_retry as subject
import test_linux_task_evidence as literal_fixture
import test_linux_process_cpu_evidence as cpu_fixture
from test_linux_code_isolation import configuration
import test_paired_task_grading as shared_fixture


def record(identifier, terminal=False):
    events = [literal_fixture.ready()]
    if terminal:
        events.append(literal_fixture.terminal())
    raw = literal_fixture.capture(events, code=0 if terminal else 137)
    source = subject.literal.bundled_source()
    before = configuration()
    before.update(Id=identifier, Image='sha256:'+subject.shared.IMAGE)
    before['Config'].update(OpenStdin=True, Tty=False, Cmd=['-I', '-S', '-B', '-c', source])
    raw.update(container_id=identifier, container_name='toklet-authored-'+identifier[:32],
               container_inspect_before=before, configuration=subject.literal.validate_configuration(before, source))
    return raw


def pair(prof=4_000_000_000):
    original, retry = record('f'*64), record('a'*64)
    witness = cpu_fixture.witness(prof=prof)
    witness.update(image=subject.shared.IMAGE, program_sha256=literal_fixture.PAYLOAD['program_sha256'])
    retry['process_cpu_captures'] = [cpu_fixture.capture(witness)]
    original['observed_cpu_retry'] = dict(qualification_sha256=subject.PROOF_SHA,
        program_sha256=literal_fixture.PAYLOAD['program_sha256'], raw=retry)
    return original


def runtime():
    value = subject.Runtime()
    value.meta = {k: cpu_fixture.EXPECTED[k] for k in ['boot_id', 'kernel', 'observer_namespace', 'observer_python_sha']}
    value.observer_source_sha = cpu_fixture.EXPECTED['observer_source_sha']
    value.literal_runtime = literal_fixture.RUNTIME
    value.literal_python = literal_fixture.PYTHON
    value.prepare = lambda _: None
    return value


class RetryControls(unittest.TestCase):
    def classify(self, raw):
        return runtime().classify(raw, literal_fixture.PAYLOAD, literal_fixture.RUNTIME, literal_fixture.PYTHON)

    def test_qualified_budget_retry_resolves_case_without_rewriting_original(self):
        raw = pair()
        saved = copy.deepcopy(raw)
        outcome = self.classify(raw)
        self.assertEqual(raw, saved)
        self.assertFalse(outcome['passed'])
        self.assertEqual(outcome['outcome'], 'resource_failure')
        self.assertTrue(outcome['original_unresolved_attempt_preserved'])

    def test_missing_underbudget_watchdog_completed_or_changed_retry_stays_unresolved(self):
        for change in ['missing', 'under', 'wall', 'completed', 'image', 'payload', 'same-lease', 'config']:
            raw = pair(prof=3_999_999_999 if change == 'under' else 4_000_000_000)
            retry = raw['observed_cpu_retry']['raw']
            if change == 'missing':
                del raw['observed_cpu_retry']
            elif change == 'wall':
                retry['wall_timed_out'] = True
            elif change == 'completed':
                raw['observed_cpu_retry']['raw'] = record('a'*64, terminal=True)
            elif change == 'image':
                retry['container_inspect_before']['Image'] = '0'*64
            elif change == 'payload':
                retry['input_sha256'] = '0'*64
            elif change == 'same-lease':
                retry['container_id'] = raw['container_id']
                retry['container_inspect_before']['Id'] = raw['container_id']
                retry['container_name'] = raw['container_name']
            elif change == 'config':
                retry['container_inspect_before']['HostConfig']['PidsLimit'] = 2
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.classify(raw)

    def test_execute_retries_only_admitted_unknown137_with_identical_payload(self):
        value = runtime()
        original = pair()
        retry = original.pop('observed_cpu_retry')['raw']
        source = subject.literal.bundled_source()
        with patch.object(subject.literal, 'execute', return_value=original), \
             patch.object(subject.observed, 'execute', return_value=retry) as observed:
            result = value.execute('api', subject.shared.IMAGE, source, literal_fixture.PAYLOAD)
        observed.assert_called_once_with('api', subject.shared.IMAGE, source, literal_fixture.PAYLOAD)
        self.assertEqual(result['observed_cpu_retry']['raw'], retry)
        self.assertEqual(value.policy['retried_cases'], [literal_fixture.PAYLOAD['case_id']])
        for terminal, invalid in [(True, False), (False, True)]:
            original = record('f'*64, terminal=terminal)
            if invalid:
                original['input_sha256'] = '0'*64
            with patch.object(subject.literal, 'execute', return_value=original), \
                 patch.object(subject.observed, 'execute') as observed:
                if invalid:
                    result = runtime().execute('api', subject.shared.IMAGE, source, literal_fixture.PAYLOAD)
                    self.assertIn('ordinary_classification_failure', result)
                    with self.assertRaises(ValueError):
                        self.classify(result)
                else:
                    self.assertNotIn('observed_cpu_retry', runtime().execute('api', subject.shared.IMAGE, source, literal_fixture.PAYLOAD))
                observed.assert_not_called()

    def test_scoped_install_restores_qualified_interfaces_after_error(self):
        value = runtime()
        before = subject.shared.controller, subject.shared.evidence
        with self.assertRaises(RuntimeError):
            with value.install():
                self.assertIs(subject.shared.controller, value)
                self.assertIs(subject.shared.evidence, value)
                raise RuntimeError('authored interruption')
        self.assertEqual((subject.shared.controller, subject.shared.evidence), before)

    def test_terminal_or_unframed_data_with137_never_triggers_retry(self):
        source = subject.literal.bundled_source()
        for events, stderr in [([literal_fixture.ready(), literal_fixture.terminal()], b''),
                               ([literal_fixture.ready()], b'unframed')]:
            original = literal_fixture.capture(events, code=137, stderr=stderr)
            with patch.object(subject.literal, 'execute', return_value=original), \
                 patch.object(subject.observed, 'execute') as observed:
                result = runtime().execute('api', subject.shared.IMAGE, source, literal_fixture.PAYLOAD)
            observed.assert_not_called()
            with self.assertRaises(ValueError):
                self.classify(result)

    def test_shared_runner_journals_ineligible_and_retry_start_failure_before_unresolved(self):
        for failure in ['ineligible', 'observer-start']:
            value = runtime()
            value.literal_runtime, value.literal_python = 'fixture-runtime', 'fixture-python'
            def original_execute(api, image, source, payload):
                ready = literal_fixture.ready()
                ready.update(binding=subject.legacy.child.admit_payload(payload),
                             runtime=value.literal_runtime, python=value.literal_python)
                raw = literal_fixture.capture([ready], code=137)
                data = subject.legacy.serialized_payload(payload)
                raw.update(input_bytes=len(data), input_sent_bytes=len(data), input_sha256=hashlib.sha256(data).hexdigest(),
                           attempt_marker='original-preserved')
                if failure == 'ineligible':
                    raw['input_sha256'] = '0'*64
                return raw
            controller = SimpleNamespace(bundled_source=lambda: b'trusted', admit_bootstrap=lambda _: None, execute=None)
            evidence = SimpleNamespace(classify=None)
            with tempfile.TemporaryDirectory(dir=subject.ROOT/'artifacts') as name, \
                 patch.object(subject.shared, 'controller', controller), patch.object(subject.shared, 'evidence', evidence), \
                 patch.object(subject.literal, 'execute', side_effect=original_execute), \
                 patch.object(subject.observed, 'execute', side_effect=OSError('authored observer startup')):
                folder = Path(name)
                result, _, _, decision = shared_fixture.SharedRunnerTests().exercise(folder,
                    execute=value.execute, classify_override=value.classify)
                journal = [json.loads(line) for line in (folder/'grading.jsonl').read_text().splitlines()]
            self.assertFalse(result['completed'])
            self.assertEqual(result['unresolved_case_ids'], [dict(profile='baseline', id='case-0')])
            self.assertEqual(len(journal), 1)
            self.assertEqual(journal[0]['raw']['attempt_marker'], 'original-preserved')
            field = 'ordinary_classification_failure' if failure == 'ineligible' else 'observed_cpu_retry_failure'
            self.assertIn(field, journal[0]['raw'])
            self.assertTrue(result['restored_stopped'])
            decision.assert_not_called()
