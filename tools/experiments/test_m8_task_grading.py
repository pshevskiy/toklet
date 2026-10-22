"""Authored admission/replay tests; no tokenizer, phone, VM or response execution."""
from contextlib import ExitStack
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import m8_task_grading as m
from test_task_cases import inventory


def decoded(cases):
    return [dict(id=c['id'], category=c['category'], visible_text='  ```python\nreturn 7\n```\n'
        if c['category']=='code' else c['expected'], visible_token_ids=[7],
        visible_text_sha256=hashlib.sha256(('  ```python\nreturn 7\n```\n'
        if c['category']=='code' else c['expected']).encode()).hexdigest()) for c in cases]


def grading_fixture(cases, rows):
    runs, outcomes = [], []
    for case, row in zip(cases, rows, strict=True):
        if case['category']=='code':
            program=m.window.probe.tasks.human_eval_program(case,row['visible_text'])
            runs.append(dict(id=case['id'],completion_sha256=row['visible_text_sha256'],
                program_sha256=hashlib.sha256(program.encode()).hexdigest(),raw=dict(authored=True),
                resolved=dict(passed=False,outcome='model_failure',reason='SyntaxError')))
        passed=case['category']!='code'
        outcomes.append(dict(id=case['id'],category=case['category'],passed=passed,
                             status='model_pass' if passed else 'model_fail'))
    return dict(completed=True,quality_completed=True,unresolved_case_ids=[],
        task_payload_sha256=m.window.probe.tasks.TASK_SHA,runs=runs,outcomes=outcomes,
        aggregate=m.window.probe.tasks.aggregate(cases,outcomes),runtime=dict(authored=True),
        python_version='authored-python')


class Admission(unittest.TestCase):
    def fixture(self, root):
        cases=inventory()[0];rows=decoded(cases);pids=[111,222];runtime=dict(version='authored',files={'authored.so':'pin'})
        normal=dict(passed=True,primary_error=None,endpoint_errors=[],bindings={},idle=dict(observations=[{},{}]))
        def save(path,data):
            path.parent.mkdir(parents=True,exist_ok=True);raw=m.responses.encoded(data);path.write_bytes(raw)
            return hashlib.sha256(raw).hexdigest()
        normal_sha=save(root/'normalized/normalization.json',normal)
        derivatives={}
        for platform in m.responses.PLATFORMS:
            raw=b'{"authored-event":true}\n';path=root/'normalized'/(platform+'.json')
            data=dict(format='toklet-m8-app-task-response-derivative-v1',platform=platform,
                quality_completed=False,source_qualification=dict(path=m.responses.QUAL,sha256=m.responses.QUAL_SHA),
                tokenizer=dict(path=m.responses.TOKENIZER,sha256=m.responses.TOKENIZER_SHA,
                    skip_special_tokens=True,runtime=runtime),local_producer_pids=pids,responses=rows,
                raw_sha256=hashlib.sha256(raw).hexdigest())
            derivatives[platform]=dict(path=str(path.relative_to(root)),sha256=save(path,data),raw_sha256=data['raw_sha256'])
            path.with_suffix('.raw.jsonl').write_bytes(raw)
        qual=dict(passed=True,task_quality_qualified=False,new_vm_runs=0,new_generated_program_executions=0,
            source_pins={},normalization=dict(path='normalized/normalization.json',sha256=normal_sha),
            execution=dict(normalization=normal,commands=[dict(pid=333,exit_code=0,primary_error=None,cleanup_errors=[])],
                           protected_binary_pins={'protected':'authored'}),derivatives=derivatives)
        qual_sha=save(root/'qualification.json',qual)
        q=dict(execution=dict(nested_receipts={'qwen8-tasks-full-run':dict(signed_bundle={'authored':'signed'})}))
        packets={p:dict(raw=b'{"authored-event":true}\n',checked={}) for p in m.responses.PLATFORMS}
        a=m.Admission.__new__(m.Admission);a.pins={};a.output=root/'logs';a.collections={};a.barrier=None
        return a,cases,rows,runtime,qual_sha,normal_sha,q,packets

    def run_fixture(self, root, *, idle_failure=False, corrupt=False):
        a,cases,rows,runtime,qual_sha,normal_sha,q,packets=self.fixture(root)
        events=[];barrier=Mock()
        def idle():
            events.append('idle')
            if idle_failure:raise ValueError('authored live phone')
        barrier.check.side_effect=idle
        construct=Mock(side_effect=lambda _:events.append('decoder') or Mock())
        def tokens(_):events.append('tokenizer_inputs');return '{}',runtime
        if corrupt:
            path=root/'normalized/mac.json';data=json.loads(path.read_bytes());data['responses'][0]['visible_text']='repaired'
            # Deliberately keep the admitted original SHA: replacement must reject before decode use.
            path.write_bytes(m.responses.encoded(data))
        with ExitStack() as stack:
            for obj,name,value in [(m,'ROOT',root),(m.responses,'ROOT',root),(m.responses.tasks.measurement,'ROOT',root),
                (m,'QUAL','qualification.json'),(m,'QUAL_SHA',qual_sha),
                (m,'NORMAL','normalized/normalization.json'),(m,'NORMAL_SHA',normal_sha)]:
                stack.enter_context(patch.object(obj,name,value))
            stack.enter_context(patch.object(m.responses,'admit_source',return_value=(q,cases,packets,[111,222])))
            stack.enter_context(patch.object(m.window.probe.tasks,'load_cases',return_value=cases))
            stack.enter_context(patch.object(m.responses,'IdleBarrier',return_value=barrier))
            stack.enter_context(patch.object(m.responses,'tokenizer_inputs',side_effect=tokens))
            stack.enter_context(patch.object(m.responses,'decode_responses',return_value=rows))
            stack.enter_context(patch.dict(sys.modules,{'tokenizers':SimpleNamespace(Tokenizer=SimpleNamespace(from_str=construct))}))
            if idle_failure or corrupt:
                with self.assertRaisesRegex(ValueError,'live phone|consumed input differs'):a.admit()
            else:a.admit()
        return a,events,construct

    def test_independent_reconstruction_requires_idle_before_tokenizer(self):
        with tempfile.TemporaryDirectory() as d:
            a,events,construct=self.run_fixture(Path(d).resolve())
            self.assertEqual(events,['idle','tokenizer_inputs','decoder']);construct.assert_called_once()
            self.assertEqual(a.pids,[111,222,333]);self.assertEqual(len(a.collections),2)

    def test_live_phone_stops_before_tokenizer_and_retains_barrier(self):
        with tempfile.TemporaryDirectory() as d:
            a,events,construct=self.run_fixture(Path(d).resolve(),idle_failure=True)
            self.assertEqual(events,['idle']);construct.assert_not_called();self.assertIsNotNone(a.barrier)

    def test_replaced_derivative_rejects_even_if_text_is_plausible(self):
        with tempfile.TemporaryDirectory() as d:self.run_fixture(Path(d).resolve(),corrupt=True)

    def test_literal_validator_rejects_repair_wrong_runtime_stop_state_or_raw(self):
        cases=inventory()[0];rows=decoded(cases);runtime=dict(version='authored',files={});raw=b'raw'
        data=dict(format='toklet-m8-app-task-response-derivative-v1',platform='mac',quality_completed=False,
            source_qualification=dict(path=m.responses.QUAL,sha256=m.responses.QUAL_SHA),
            tokenizer=dict(path=m.responses.TOKENIZER,sha256=m.responses.TOKENIZER_SHA,skip_special_tokens=True,runtime=runtime),
            local_producer_pids=[111],responses=rows,raw_sha256=hashlib.sha256(raw).hexdigest())
        m.validate_derivative(data,'mac',rows,[111],runtime,raw)
        for mutate in [lambda d:d['responses'][0].update(visible_text='trimmed'),
            lambda d:d['responses'][0].update(stop_reason=dict(kind='stop_token',token_id=7)),
            lambda d:d['responses'][0].update(state=dict(pending=0)),
            lambda d:d['tokenizer']['runtime'].update(version='different'),
            lambda d:d.update(platform='iphone14pro'),lambda d:d.update(raw_sha256='0'*64)]:
            bad=copy.deepcopy(data);mutate(bad)
            with self.subTest(mutate=mutate),self.assertRaises(ValueError):
                m.validate_derivative(bad,'mac',rows,[111],runtime,raw)

    def test_idle_binding_covers_both_derivatives_and_all_host_producers(self):
        a=m.Admission.__new__(m.Admission);a.pids=[111,222,333];a.protected={};a.pins={}
        a.paths=[m.ROOT/'authored/mac.json',m.ROOT/'authored/iphone14pro.json']
        a.collections={str(p):dict(collection_path=str(p),collection_sha256='json',raw_sha256='raw',identity={}) for p in a.paths}
        with patch.object(m.responses,'capture',return_value=m.responses.encoded(a.idle_proof())),patch.object(a,'guard'):
            binding=a.admit_idle(m.ROOT/'authored/proof.json',a.paths[0])
            self.assertEqual([r['worker_pid'] for r in binding['collections']],a.pids)
            self.assertEqual({r['path'] for r in binding['collections']},{str(p) for p in a.paths})
            with self.assertRaises(ValueError):a.admit_idle(m.ROOT/'authored/proof.json',a.paths[1])

    def test_cached_collection_still_requires_full_endpoint_guard(self):
        a=m.Admission.__new__(m.Admission);a.collections={str(m.ROOT/'authored/mac.json'):dict(authored=True)}
        with patch.object(a,'guard',side_effect=ValueError('authored changed')):
            with self.assertRaisesRegex(ValueError,'authored changed'):a.validate_collection(m.ROOT/'authored/mac.json')
        with patch.object(a,'guard'):
            with self.assertRaises(ValueError):a.validate_collection(m.ROOT/'authored/other.json')

    def test_remote_hook_repeats_guarded_barrier_and_never_resets(self):
        a=m.Admission.__new__(m.Admission);a.barrier=Mock()
        with patch.object(a,'guard') as guard:
            a.idle_check({});a.idle_check({});self.assertEqual(guard.call_count,4)
        self.assertEqual(a.barrier.check.call_count,2)
        a.barrier.stop_remote.assert_not_called()
    def test_private_ledger_keeps_original_causes_and_late_guard_failure(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve();a=m.Admission.__new__(m.Admission)
            a.output=root;a.pins={};a.barrier=SimpleNamespace(commands=[dict(primary_error='original private-id',cleanup_errors=['original cleanup'],capture_errors=['original capture'])])
            with patch.object(m.responses,'ROOT',root),patch.object(a,'guard',side_effect=[None,ValueError('late private-id guard')]):
                errors=a.finalize_private('primary private-id',['entry private-id'])
            ledger=json.loads((root/'private-execution.json').read_bytes())
            final=json.loads((root/'private-final-guard.json').read_bytes())
            self.assertEqual(ledger['commands'],a.barrier.commands)
            self.assertEqual(ledger['endpoint_errors'],['entry private-id'])
            self.assertEqual(final['endpoint_errors'],errors)
            self.assertIn('late private-id guard',errors[-1])
            public=m.responses.physical.redact_record(final,dict(device_identifier='private-id'))
            self.assertIn('private-device_identifier',public['endpoint_errors'][-1])
            self.assertEqual(final['primary_error'],'primary private-id')
            self.assertEqual(len(a.pins),2)


class Replay(unittest.TestCase):
    def test_40_program_hashes_and_60_text_outcomes_replay_literally(self):
        cases=inventory()[0];rows=decoded(cases);g=grading_fixture(cases,rows);seen=[]
        def classify(raw,payload,runtime,version):
            self.assertEqual(raw,dict(authored=True));self.assertEqual(runtime,g['runtime'])
            self.assertEqual(version,g['python_version']);seen.append(payload['program'])
            return dict(passed=False,outcome='model_failure',reason='SyntaxError')
        with patch.object(m.window.probe.evidence,'classify',side_effect=classify):
            outcomes,aggregate=m.replay_outcomes(cases,rows,g)
        self.assertEqual(len(seen),40);self.assertEqual(aggregate['passed'],60)
        self.assertIn('  ```python\nreturn 7\n```\n',seen[0]);self.assertEqual(outcomes,g['outcomes'])

    def test_mismatch_or_unresolved_never_reuses_outcomes(self):
        cases=inventory()[0];rows=decoded(cases);original=grading_fixture(cases,rows)
        for mutate in [lambda d:d.update(quality_completed=False),lambda d:d.update(unresolved_case_ids=['unknown']),
            lambda d:d['runs'].pop(),lambda d:d['runs'][0].update(program_sha256='0'*64),
            lambda d:d['runs'][0].update(completion_sha256='0'*64),lambda d:d['runs'][0].update(infrastructure_failure='unknown')]:
            bad=copy.deepcopy(original);mutate(bad)
            with self.subTest(mutate=mutate),self.assertRaises(ValueError):m.replay_outcomes(cases,rows,bad)
        with patch.object(m.window.probe.evidence,'classify',return_value=dict(passed=True)):
            with self.assertRaisesRegex(ValueError,'outcome replay differs'):m.replay_outcomes(cases,rows,original)

    def test_unrepaired_text_replacement_cannot_use_saved_literal_hash(self):
        cases=inventory()[0];rows=decoded(cases);g=grading_fixture(cases,rows)
        rows[0]['visible_text']='return 7'
        with self.assertRaisesRegex(ValueError,'text binding differs'):m.replay_outcomes(cases,rows,g)

    def test_original_zero_pass_boundary_remains_explicit(self):
        baseline=dict(passed=True,quality_completed=True,review=dict(accepted=True),infrastructure_unresolved=0,
                      models=[dict(model='small',fp32_passed=2,unchanged_allowed_drop_percentage_points=2)])
        with patch.object(m.responses,'capture',side_effect=lambda *args:json.dumps(baseline).encode()):
            for n in [0,2,3]:
                gate=m.quality_gate(dict(cases=100,passed=n),{})
                self.assertTrue(gate['gate_passed']);self.assertEqual(gate['drop_percentage_points'],2-n)
            for bad in [dict(cases=99,passed=2),dict(cases=100,passed=True)]:
                with self.assertRaises(ValueError):m.quality_gate(bad,{})
            baseline['models'][0]['fp32_passed']=3
            with self.assertRaises(ValueError):m.quality_gate(dict(cases=100,passed=2),{})


if __name__=='__main__':unittest.main()
