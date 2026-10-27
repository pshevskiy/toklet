"""Complete protocol, pending/EOS and unchanged acceptance-limit controls."""
import copy
import math
import unittest

import qwen3_quality_results as subject


def manifest():
    return dict(files={'held_out_loss.u32le':dict(sha256='1'*64),'execution.jsonl':dict(sha256='2'*64)})


def ready(profile,mode):
    value=dict(event='ready',format=subject.FORMATS[profile],mode=mode,full_quality_workload=True,
        targets=10000 if mode=='loss'else 0,tasks=100 if mode=='tasks'else 0)
    if profile=='native':
        value.update(input_sha256='1'*64 if mode=='loss'else'2'*64,context_capacity=1025,prefill_chunk_size=128,
            debug_assertions=False,c512_numerical_gate_status='rejected_unchanged',projection='mlx_affine',
            normalization='compensated',attention='reference')
    else:value['input_manifest_sha256']=subject.contract.sha(subject.contract.encoded(manifest()))
    return value


def loss_events(profile):
    result=[ready(profile,'loss')]
    for block,start in enumerate(range(0,10000,1024)):
        end=min(start+1024,10000)
        for first in range(0,end-start+1,128):
            rows=min(128,end-start+1-first)
            for p in range(first,min(first+rows,end-start)):
                result.append(dict(event='loss',target_index=start+p,block=block,position=p,target_token=0,
                    negative_log_likelihood=1.0,target_logit_f32_bits=0,logits_sha256='0'*64))
            if profile=='native':result.append(dict(event='chunk_done',block=block,rows=rows,evaluated_tokens=first+rows,pending_tokens=0))
        result.append(dict(event='block_done',block=block,target_start=start,target_end=end,targets=end-start,
            evaluated_tokens=end-start+1,pending_tokens=0,total_negative_log_likelihood=float(end-start)))
    result.append(dict(event='done',mode='loss',full_quality_workload=True,targets=10000,tasks=0,
        total_negative_log_likelihood=10000.0,mean_negative_log_likelihood=1.0))
    return result


def cases():
    result=[]
    for i in range(100):
        category,cap=('code',256)if i<40 else('prose',32)if i<70 else('continuation',16)
        row=dict(id=f'case/{i}',category=category,prompt_tokens=[11,12],max_output_tokens=cap,prompt_text='A prompt')
        if category=='code':row.update(code_prefix='def f():\n',tests='def check(f):\n    assert f()==1\n',entry_point='f')
        else:row['expected']='A'if category=='prose'else'Word!'
        result.append(row)
    return result


def task_events(profile):
    result=[ready(profile,'tasks')]
    for case in cases():
        if profile=='baseline':
            result.append(dict(event='task_started',id=case['id']))
            result.append(dict(event='token',id=case['id'],index=0,token_id=151645,visible=False))
        result.append(dict(event='task_done',id=case['id'],category=case['category'],prompt_tokens=2,
            requested_output_tokens=case['max_output_tokens'],visible_token_ids=[],
            report=dict(reason=dict(kind='stop_token',token_id=151645),sampled_tokens=1,emitted_tokens=0,
                history_tokens=3,evaluated_tokens=2,pending_tokens=1)))
    done=dict(event='done',targets=0,tasks=100)
    if profile=='baseline':done.update(mode='tasks',full_quality_workload=True)
    result.append(done)
    return result


class CompleteCollections(unittest.TestCase):
    def test_full_loss_and_all_targets_are_required_for_both_profiles(self):
        for profile in subject.FORMATS:
            rows=loss_events(profile)
            self.assertEqual(subject.loss(rows,profile,manifest(),[0]*10001)['mean_negative_log_likelihood'],1.0)
            for alteration in ('missing','nan','wrong_target','wrong_total','probe'):
                changed=copy.deepcopy(rows)
                if alteration=='missing':changed.pop(1)
                elif alteration=='nan':changed[1]['negative_log_likelihood']=float('nan')
                elif alteration=='wrong_target':changed[1]['target_token']=1
                elif alteration=='wrong_total':changed[-1]['total_negative_log_likelihood']+=0.01
                else:changed[0]['full_quality_workload']=False
                with self.assertRaises(ValueError):subject.loss(changed,profile,manifest(),[0]*10001)
            with self.assertRaises(ValueError):subject.loss(rows[:-1],profile,manifest(),[0]*10001)
            if profile=='native':
                changed=copy.deepcopy(rows)
                changed.insert(-1,dict(event='chunk_done',block=10,rows=1,evaluated_tokens=1,pending_tokens=0))
                with self.assertRaises(ValueError):subject.loss(changed,profile,manifest(),[0]*10001)

    def test_hidden_eos_pending_and_complete_case_coverage_are_required(self):
        for profile in subject.FORMATS:
            rows=task_events(profile)
            self.assertEqual(len(subject.tasks(rows,profile,manifest(),cases())),100)
            for field,value in [('sampled_tokens',0),('pending_tokens',0),('evaluated_tokens',3),('reason',dict(kind='cancelled'))]:
                changed=copy.deepcopy(rows);done=next(v for v in changed if v['event']=='task_done');done['report'][field]=value
                with self.assertRaises(ValueError):subject.tasks(changed,profile,manifest(),cases())
            with self.assertRaises(ValueError):subject.tasks(rows[:-1],profile,manifest(),cases())
            if profile=='baseline':
                for field,value in [('mode','probe'),('full_quality_workload',False)]:
                    changed=copy.deepcopy(rows);changed[-1][field]=value
                    with self.assertRaises(ValueError):subject.tasks(changed,profile,manifest(),cases())

    def test_literal_existing_grading_keeps_fences_and_exact_text_rules(self):
        case=subject.legacy_case(cases()[0]);text='```python\n    return 1\n```'
        program=subject.task_cases.human_eval_program(case,text)
        self.assertTrue(program.startswith(case['code_prefix']+text))
        self.assertTrue(program.endswith('check(f)\n'))
        self.assertTrue(subject.task_cases.grade_text(subject.legacy_case(cases()[40]),' A \n'))
        self.assertFalse(subject.task_cases.grade_text(subject.legacy_case(cases()[40]),'Answer: A'))
        self.assertTrue(subject.task_cases.grade_text(subject.legacy_case(cases()[70]),'Word! extra'))
        self.assertFalse(subject.task_cases.grade_text(subject.legacy_case(cases()[70]),'word!'))

    def test_exact_existing_quality_limits_and_unresolved_case_rejection(self):
        c=cases()
        def outcomes(passed):return [dict(id=v['id'],category=v['category'],passed=i<passed,
            status='model_pass'if i<passed else'model_fail')for i,v in enumerate(c)]
        value=subject.decision(dict(targets=10000,mean_negative_log_likelihood=0.05),
            dict(targets=10000,mean_negative_log_likelihood=0.0),c,outcomes(78),outcomes(80))
        self.assertEqual(value['decision'],'retain_experimental_candidate')
        value=subject.decision(dict(targets=10000,mean_negative_log_likelihood=math.nextafter(0.05,math.inf)),
            dict(targets=10000,mean_negative_log_likelihood=0.0),c,outcomes(78),outcomes(80))
        self.assertEqual(value['decision'],'reject_precision_candidate')
        value=subject.decision(dict(targets=10000,mean_negative_log_likelihood=0.0),
            dict(targets=10000,mean_negative_log_likelihood=0.0),c,outcomes(77),outcomes(80))
        self.assertFalse(value['tasks_passed'])
        unresolved=outcomes(80);unresolved[0]['status']='infrastructure_failure'
        with self.assertRaises(ValueError):subject.decision(dict(targets=10000,mean_negative_log_likelihood=0.0),
            dict(targets=10000,mean_negative_log_likelihood=0.0),c,unresolved,outcomes(80))


if __name__=='__main__':unittest.main()
