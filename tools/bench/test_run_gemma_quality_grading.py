import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import run_gemma_quality_grading as subject


class WindowControls(unittest.TestCase):
    def exercise(self, change=None):
        with tempfile.TemporaryDirectory(dir=subject.ROOT/'artifacts') as directory:
            folder = Path(directory)
            client = folder/'podman'
            client.write_bytes(b'pinned-client')
            collection = dict(format='toklet-gemma-paired-task-collection-v1', completed=True, cases=100,
                              manifest_sha256=subject.adapter.MANIFEST_SHA, outputs={})
            for index, role in enumerate(['original', 'native']):
                journal = folder/(role+'.jsonl')
                journal.write_bytes(b'{}\n')
                receipt = folder/(role+'.json')
                worker = dict(passed=True, cases=100, tasks_sha256=hashlib.sha256(journal.read_bytes()).hexdigest(),
                              command=dict(pid=900001+index, exit_code=0, primary_error=None, cleanup_errors=[]))
                if change == 'live-model' and index == 0:
                    worker['command']['cleanup_errors'] = ['surviving model']
                receipt.write_text(json.dumps(worker))
                collection['outputs'][role] = dict(worker_receipt=str(receipt.relative_to(subject.ROOT)),
                    worker_receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest(),
                    journal=str(journal.relative_to(subject.ROOT)), journal_sha256=worker['tasks_sha256'])
            path = folder/'collection.json'
            path.write_text(json.dumps(collection))
            output = folder/'outer'
            cases = subject.adapter.task_cases.load_cases()
            outcomes = {role: [dict(id=c['id'], category=c['category'], passed=False, status='model_fail')
                               for c in cases] for role in ['baseline', 'native']}
            decision = subject.adapter.results.decision(cases, outcomes)

            def supervised(command, stdout, stderr, environment, observe, group_alive, timeout):
                self.assertEqual(command[-1], str(output/'inner'))
                self.assertEqual(command[-3], str(path))
                self.assertEqual(environment['TZ'], 'Europe/Moscow')
                (output/'inner').mkdir()
                inner = dict(completed=True, restored_stopped=True,
                    pipe_lifetime_finished=True, unresolved_case_ids=['HumanEval/0'] if change == 'unresolved' else [],
                    format='toklet-gemma-full-task-grading-v1', collection_sha256=command[-2],
                    manifest_sha256=subject.adapter.MANIFEST_SHA, outcomes=outcomes, decision=decision)
                if change == 'missing-outcomes':
                    inner['outcomes'] = dict(baseline=[], native=[])
                if change == 'wrong-decision':
                    inner['decision'] = dict(task_gate_passed=True)
                if change == 'wrong-identity':
                    inner['collection_sha256'] = '0'*64
                subject.support.write_json(output/'inner/result.json', inner)
                if change == 'changed-journal':
                    (folder/'original.jsonl').write_bytes(b'{"changed":true}\n')
                return dict(pid=900003, exit_code=0, primary_error=None, cleanup_errors=[],
                            independent_stopped_verification=change != 'running-vm')

            with patch.object(subject.runner, 'PODMAN', str(client)), \
                 patch.object(subject.runner, 'checked_json', return_value=dict(podman_sha256=hashlib.sha256(client.read_bytes()).hexdigest())), \
                 patch.object(subject.os, 'kill', side_effect=ProcessLookupError), \
                 patch.object(subject.support, 'supervise_window', side_effect=supervised) as supervisor:
                result = subject.run(path, hashlib.sha256(path.read_bytes()).hexdigest(), output)
            self.assertEqual(json.loads((output/'result.json').read_text()), result)
            if change:
                self.assertFalse(result['completed'])
                self.assertNotIn('decision', result)
                self.assertIn('failure', result)
            else:
                self.assertTrue(result['completed'])
                self.assertTrue(result['independent_group_absence_verified'])
                self.assertEqual(result['decision'], decision)
            if change == 'live-model':
                supervisor.assert_not_called()
            else:
                supervisor.assert_called_once()

    def test_clean_restored_window_publishes_decision(self):
        self.exercise()

    def test_unresolved_outcomes_and_running_vm_withhold_decision(self):
        for change in ['unresolved', 'running-vm']:
            with self.subTest(change=change):
                self.exercise(change)

    def test_changed_journal_withholds_decision(self):
        self.exercise('changed-journal')

    def test_incomplete_or_inconsistent_inner_result_withholds_decision(self):
        for change in ['missing-outcomes', 'wrong-decision', 'wrong-identity']:
            with self.subTest(change=change):
                self.exercise(change)

    def test_unclean_model_exit_prevents_grader_launch(self):
        self.exercise('live-model')
