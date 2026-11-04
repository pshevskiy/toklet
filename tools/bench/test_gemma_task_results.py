"""Reject invalid inference or grading inputs before any isolated execution."""
import copy
import hashlib
from unittest.mock import patch
import unittest
import gemma_task_results as target
import task_cases


def cases():
    result=[]
    for i in range(100):
        category='code' if i<40 else 'prose' if i<70 else 'continuation'
        cap={'code':256,'prose':32,'continuation':16}[category]
        item=dict(id=str(i),category=category,prompt_tokens=[1],prompt_text='prompt',
                  max_output_tokens=cap,stop_policy=f'eos_or_{cap}',scoring=task_cases.SCORERS[category],stop_ids=[1,106,50])
        if category=='code':item.update(code_prefix='def f(x):\n',tests='def check(f):\n    assert f(1) == 1\n',entry_point='f')
        else:item['expected']='A' if category=='prose' else 'word.'
        result.append(item)
    return result


def original(frozen):
    records=[]
    for case in frozen:
        generation=dict(visible_token_ids=[],stop_reason=dict(kind='stop_token',token_id=106),
            samples=[dict(index=0,token_id=106,margin=1.,row_sha256='0'*64)],
            committed_tokens=1,history_tokens=2,pending_tokens=1)
        replay=dict(verified=True,mismatches=[],visible_predictions=0,terminal_stop_checked=True,
            rows=[dict(index=0,absolute_row=0,expected_id=106,stateless_id=106,
                       incremental_margin=1.,stateless_margin=1.,stateless_row_sha256='1'*64)])
        records.append(dict(id=case['id'],category=case['category'],prompt_tokens=1,
                            output_reserve=case['max_output_tokens'],generation=generation,trajectory=replay))
    return records


class Controls(unittest.TestCase):
    def test_literal_mapping_keeps_actual_gemma_prompts_separate(self):
        canonical=cases();gemma=copy.deepcopy(canonical)
        for case in gemma:case['prompt_tokens']=[200000]
        self.assertIs(target.canonical_grading_cases(gemma,canonical),canonical)
        for change in [lambda r:r.__setitem__(0,r[1]),lambda r:r[0].update(tests=r[0]['tests']+' '),
                       lambda r:r[40].update(expected='B')]:
            bad=copy.deepcopy(gemma);change(bad)
            with self.assertRaises(ValueError):target.canonical_grading_cases(bad,canonical)

    def test_unverified_invalid_id_stop_and_case_swap_reject(self):
        frozen=cases();records=original(frozen)
        self.assertEqual(len(target.collection(frozen,records,'original')),100)
        for change in [lambda r:r[0]['trajectory'].update(verified=False),
            lambda r:r[0]['generation'].update(visible_token_ids=[262144]),
            lambda r:r[0]['generation'].update(visible_token_ids=[106]),
            lambda r:r.__setitem__(0,r[1]),lambda r:r[0]['generation'].update(pending_tokens=0)]:
            bad=copy.deepcopy(records);change(bad)
            with self.assertRaises(ValueError):target.collection(frozen,bad,'original')

    def test_whole_visible_decode_preserves_fences_whitespace_and_hashes(self):
        frozen=cases();records=target.collection(frozen,original(frozen),'original')
        raw=b'{}';calls=[];text=' \n```python\nx\n```  '
        class Decoder:
            def decode(self,ids,skip_special_tokens):
                calls.append((list(ids),skip_special_tokens));return text
        spec=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        decoded=target.decode(frozen,records,raw,spec,factory=lambda body:Decoder())
        self.assertEqual(len(calls),100);self.assertTrue(all(flag for ids,flag in calls))
        self.assertEqual(decoded[0]['visible_text'],text)
        self.assertEqual(decoded[0]['visible_text_sha256'],hashlib.sha256(text.encode()).hexdigest())
        with self.assertRaises(ValueError):target.decode(frozen,records,b'bad',spec,factory=lambda body:Decoder())

    def test_two_point_gate_and_unresolved_outcomes(self):
        frozen=cases()
        def outcomes(passed):
            return [dict(id=c['id'],category=c['category'],passed=i<passed,
                         status='model_pass' if i<passed else 'model_fail')for i,c in enumerate(frozen)]
        self.assertTrue(target.decision(frozen,dict(baseline=outcomes(5),native=outcomes(3)))['task_gate_passed'])
        self.assertFalse(target.decision(frozen,dict(baseline=outcomes(5),native=outcomes(2)))['task_gate_passed'])
        unresolved=outcomes(5);unresolved[0]['status']='infrastructure_failure'
        with self.assertRaises(ValueError):target.decision(frozen,dict(baseline=unresolved,native=outcomes(3)))

    def test_live_or_failed_model_blocks_grading(self):
        workers=[dict(passed=True,cases=100,command=dict(pid=i,exit_code=0,primary_error=None,cleanup_errors=[]))for i in [100,200]]
        with patch.object(target.os,'kill',side_effect=ProcessLookupError):self.assertEqual(target.require_idle(workers),[100,200])
        with patch.object(target.os,'kill',return_value=None):
            with self.assertRaises(ValueError):target.require_idle(workers)
        workers[0]['passed']=False
        with self.assertRaises(ValueError):target.require_idle(workers)


if __name__=='__main__':unittest.main()
