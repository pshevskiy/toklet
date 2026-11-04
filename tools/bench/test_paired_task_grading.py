"""Shared runner controls; all VM/container operations are mocked."""
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch
import paired_task_grading as subject


def fixtures():
    cases=[];decoded={name:[]for name in ['baseline','native']}
    for index in range(100):
        category='code'if index<40 else'prose'if index<70 else'continuation';cap=subject.task_cases.CAPS[category]
        case=dict(id=f'case-{index}',category=category,prompt_text='fixture',prompt_tokens=[1],max_output_tokens=cap,
                  stop_policy=f'eos_or_{cap}',scoring=subject.task_cases.SCORERS[category])
        if category=='code':case.update(entry_point='f',code_prefix='def f():\n',tests='def check(f):\n    assert f()==1\n')
        else:case['expected']='A'if category=='prose'else'word'
        text='    return 1\n'if category=='code'else'A'if category=='prose'else'word extra'
        cases.append(case)
        for profile in decoded:decoded[profile].append(dict(id=case['id'],category=category,visible_text=text,visible_text_sha256=hashlib.sha256(text.encode()).hexdigest()))
    return cases,decoded


class SharedRunnerTests(unittest.TestCase):
    def exercise(self, folder, unresolved=False, stop_failure=False, execute=None, classify_override=None):
        cases,decoded=fixtures();state={'running':False};owner=Mock()
        image=dict(Id='sha256:'+subject.IMAGE,Digest=None,RepoDigests=None,Architecture='arm64',Os='linux',RootFS=None)
        binding=dict(URI='ssh://fixture',Identity='fixture-key')
        prior=dict(completed=True,runs=[{}for _ in range(14)],source_sha256={},podman_sha256='0'*64,
                   admitted_runtime='fixture-runtime',admitted_python_version='fixture-python',connection_binding=binding,
                   podman_version='{}',image=image)
        literal=dict(completed=True,runs=[dict(expectation_passed=True)for _ in range(28)],source_sha256={})
        def capture(argv,*_):
            code=0;text='{}'
            if argv[1:3]==['machine','list']:
                text=json.dumps([dict(Name=subject.lifecycle.MACHINE,Running=state['running'],Starting=False,Memory=2147483648)])
            elif argv[1:3]==['machine','inspect']:text='[{}]'
            elif argv[1:4]==['system','connection','list']:text='[]'
            elif 'image'in argv and 'inspect'in argv:text=json.dumps([image])
            elif argv[1:3]==['machine','stop']:
                if stop_failure:code=1
                else:state['running']=False
            return dict(exit_code=code,wall_timed_out=False,output_limit_exceeded=False,cleanup_errors=[],surviving_group_members=[],text=text)
        def start(_):state['running']=True;return dict(client_succeeded=True,helper_capture_failure=None),owner
        record=dict(completed=False,commands=[],runs=[],outcomes={},unresolved_case_ids=[],generated_code_runs_on_host=False)
        guard=Mock();decide=Mock(return_value=dict(tasks_passed=True));classify=Mock(side_effect=ValueError('fixture unresolved'))if unresolved else Mock(return_value=dict(passed=True))
        with ExitStack()as stack:
            for obj,name,value in [
                (subject,'checked_json',lambda path,*_:prior if path==subject.PRIOR else literal),
                (subject.boundary,'check_record',lambda _:None),(subject.boundary,'digest',lambda _:'0'*64),
                (subject.boundary,'capture',capture),(subject.boundary,'output',lambda item:item['text']),
                (subject.boundary,'bound_machine_connection',lambda *_:binding),
                (subject.lifecycle,'start_machine',start),(subject.lifecycle,'await_stopped_owner',Mock()),
                (subject,'running_containers',lambda *_:[]),(subject.controller,'bundled_source',lambda:b'trusted'),
                (subject.controller,'admit_bootstrap',lambda _:None),(subject.controller,'execute',execute or Mock(return_value=dict(raw='fixture'))),
                (subject.evidence,'classify',classify_override or classify)]:stack.enter_context(patch.object(obj,name,value))
            result=subject.grade_paired_cases(cases,decoded,guard,folder,record,{},decide)
        return result,owner,guard,decide
    def test_all80_literal_programs_journal_before_final_restored_decision(self):
        with tempfile.TemporaryDirectory(dir=subject.ROOT/'artifacts')as name:
            folder=Path(name);result,owner,guard,decide=self.exercise(folder)
            self.assertTrue(result['completed']);self.assertTrue(result['restored_stopped']);self.assertTrue(result['pipe_lifetime_finished'])
            self.assertEqual(len(result['runs']),80);self.assertEqual(len((folder/'grading.jsonl').read_bytes().splitlines()),80)
            self.assertEqual(len(result['outcomes']['baseline']),100);self.assertEqual(len(result['outcomes']['native']),100)
            owner.finish.assert_called_once();decide.assert_called_once();self.assertGreaterEqual(guard.call_count,3)
    def test_unresolved_evidence_is_preserved_and_not_counted_as_model_failure(self):
        with tempfile.TemporaryDirectory(dir=subject.ROOT/'artifacts')as name:
            folder=Path(name);result,owner,guard,decide=self.exercise(folder,unresolved=True)
            self.assertFalse(result['completed']);self.assertEqual(len(result['unresolved_case_ids']),1);self.assertEqual(len(result['runs']),1)
            self.assertEqual(len((folder/'grading.jsonl').read_bytes().splitlines()),1);decide.assert_not_called();self.assertNotIn('decision',result)
            self.assertTrue(result['restored_stopped'])
    def test_restoration_failure_withholds_decision_after_all_programs_run(self):
        with tempfile.TemporaryDirectory(dir=subject.ROOT/'artifacts')as name:
            result,owner,guard,decide=self.exercise(Path(name),stop_failure=True)
            self.assertEqual(len(result['runs']),80);self.assertFalse(result['completed']);self.assertIn('restoration_failure',result)
            self.assertNotIn('decision',result);owner.finish.assert_not_called()
