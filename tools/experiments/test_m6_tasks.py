"""Online task producer/admission and original isolated-window hooks; no GPU/VM."""
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import m6_tasks as producer
import m6_task_admission as admission
import m6_task_window as window
import test_metal_tasks as fixtures
import test_metal_tasks_v2 as v2_fixtures
import test_task_grade_admission as full_fixtures
from test_task_cases import inventory


def ready(model='small'):
    value = v2_fixtures.admission() if model == 'larger' else fixtures.MetalTasks().admission()
    value['execution']['attention_mode'] = 'Online'
    return value


def collection(model='small'):
    record, cases, raw = full_fixtures.GradingAdmission().fixture()
    initial = ready(model); record['run']['ready'] = initial; raw[0] = initial
    record.update(format='toklet-m6-online-task-collection-v1', model=model, attention_mode='Online',
                  projection_mode='Q4CompensatedTiled' if model == 'larger' else 'HalfCompensated',
                  normalization_mode='Compensated' if model == 'larger' else 'Original')
    for case, item in zip(cases, record['run']['cases'], strict=True):
        for event in item['native_events']:
            if event['event'] == 'done':
                event['statistics'].update({k: v for k, v in initial.items() if k not in ('event', 'context_capacity')})
        item['response'] = producer.shared.collect_response(case, item['native_events'], initial)
    record['run']['idle_usage'] = [dict(phase=phase, worker={'lifetime_max_phys_footprint': 1},
                                       driver={'lifetime_max_phys_footprint': 2}, joint_native_lifetime_peak_bytes=3)
                                  for phase in ['ready']+[f'case_{i}' for i in range(1, 101)]]
    return record, cases, raw


class OnlineTasks(unittest.TestCase):
    def test_actual_online_and_selected_modes_complete_app_memory(self):
        for model in ('small', 'larger'):
            event = ready(model); producer.ready(event, model)
            for key, value in [('attention_mode', 'Reference'), ('attention_mode', True),
                               ('projection_mode', 'Original'), ('kv_dtype', 'I8'), ('prefill_chunk_size', True)]:
                bad = copy.deepcopy(event); bad['execution'][key] = value
                with self.subTest(model=model, key=key), self.assertRaises(ValueError): producer.ready(bad, model)
            bad = copy.deepcopy(event); bad['logical_memory']['tokenizer_application_allowance_bytes'] = 1
            bad['memory_admitted_bytes'] = sum(bad['logical_memory'].values())
            with self.assertRaises(ValueError): producer.ready(bad, model)

    def test_full_loss_prerequisite_for_each_selected_model(self):
        for model in ('small', 'larger'):
            value = dict(passed=True, review={'accepted': True}, models=[dict(model=model, loss_gate_passed=True, scored_targets=10000)])
            with patch.object(producer.context, 'bounded', return_value=json.dumps(value).encode()): producer.loss_prerequisite(model)
            for change in ({'loss_gate_passed': False}, {'scored_targets': 9999}, {'model': 'other'}):
                bad = copy.deepcopy(value); bad['models'][0].update(change)
                with patch.object(producer.context, 'bounded', return_value=json.dumps(bad).encode()), self.assertRaises(ValueError): producer.loss_prerequisite(model)

    def test_changed_prelaunch_inputs_never_start_worker_or_decoder(self):
        def reject(): raise ValueError('changed inputs')
        with patch.object(producer.shared.subprocess, 'Popen') as launch, patch.object(producer.shared, 'finalized') as decoder:
            value = producer.shared.child(Path('bin'), Path('source'), 'small', inventory()[0], io.StringIO(), io.BytesIO(), True, before_launch=reject)
        self.assertFalse(value['passed']); self.assertIn('changed inputs', value['error'])
        launch.assert_not_called(); decoder.assert_not_called()

    def test_full_caps_original_event_grammar_and_exact_online_identity(self):
        for model in ('small', 'larger'):
            record, cases, raw = collection(model)
            valid = admission.validate_records(record, cases, raw)
            self.assertEqual(len(valid['responses']), 100)
            self.assertEqual(valid['identity']['attention_mode'], 'Online')
            self.assertEqual(valid['identity']['model'], model)

    def test_partial_changed_caps_exit_and_raw_cannot_enter_grading(self):
        for mutation in (lambda p: p.update(check_only=True), lambda p: p.update(quality_completed=True),
                         lambda p: p.update(collection_completed=False), lambda p: p.update(attention_mode='Reference'),
                         lambda p: p['task_ids'].reverse(), lambda p: p['run'].update(model_exited=False),
                         lambda p: p['run'].update(exit_code=True), lambda p: p['run'].update(worker_pid=True),
                         lambda p: p['run']['cases'][0].update(max_output_tokens=8)):
            record, cases, raw = collection(); mutation(record)
            with self.assertRaises(ValueError): admission.validate_records(record, cases, raw)
        record, cases, raw = collection()
        with self.assertRaises(ValueError): admission.validate_records(record, cases, raw[:-1])

    def test_ordered_native_coverage_and_charge_are_required(self):
        for mutation in (lambda u: u.pop(), lambda u: u.reverse(), lambda u: u[1].update(phase='ready'),
                         lambda u: u[0].update(joint_native_lifetime_peak_bytes=True),
                         lambda u: u[0]['worker'].update(lifetime_max_phys_footprint=0),
                         lambda u: u[0]['driver'].update(lifetime_max_phys_footprint=producer.shared.LIMIT)):
            record, cases, raw = collection(); mutation(record['run']['idle_usage'])
            with self.assertRaises(ValueError): admission.validate_records(record, cases, raw)

    def test_idle_proof_matches_one_completed_collection(self):
        with tempfile.TemporaryDirectory() as directory:
            selected = Path(directory)/'collection.json'; proof = Path(directory)/'idle.json'
            proof.write_text(json.dumps(dict(format='toklet-m6-online-task-idle-v1', collections=[str(selected.resolve())])))
            accepted = dict(collection_path=str(selected), collection_sha256='summary', raw_sha256='raw', identity={'model': 'small'}, worker_pid=12345)
            with patch.object(admission, 'validate_collection', return_value=accepted):
                value = admission.admit_idle(proof, selected)
                self.assertEqual(value['collections'][0]['worker_pid'], 12345)
            proof.write_text(json.dumps(dict(format='toklet-m6-online-task-idle-v1', collections=[str(selected)]*2)))
            with patch.object(admission, 'validate_collection') as validate, self.assertRaises(ValueError): admission.admit_idle(proof, selected)
            validate.assert_not_called()

    def test_window_preserves_existing_literal_controller_and_idle_checks(self):
        args = SimpleNamespace(collection=Path('c'), idle_proof=Path('i'), prefix='p')
        with patch.object(window.shared, 'run', return_value=0) as run:
            self.assertEqual(window.run(args), 0)
        self.assertIs(run.call_args.kwargs['collection_admission'], admission)
        self.assertIs(run.call_args.kwargs['idle_admission'], admission.admit_idle)
        self.assertIn(producer, run.call_args.kwargs['extra_grading_modules'])

    def test_exclusive_output_rejects_before_model_admission(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'out.json'; p.write_text('preserve')
            with patch.object(producer, 'provenance') as check, self.assertRaises(ValueError): producer.run(p, 'small')
            check.assert_not_called(); self.assertEqual(p.read_text(), 'preserve')


if __name__ == '__main__': unittest.main()
