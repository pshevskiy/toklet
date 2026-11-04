"""Repair discards suffixes, handles terminal corrections and never grades cycles."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import gemma4_task_recovery as target


def proposal(ids,stop=False):
    visible=ids[:-1]if stop else ids
    reason=dict(kind='stop_token',token_id=ids[-1])if stop else dict(kind='output_limit')
    return dict(visible_token_ids=visible,stop_reason=reason,
        samples=[dict(index=i,token_id=v,margin=1.,row_sha256='0'*64)for i,v in enumerate(ids)],
        committed_tokens=1+len(ids)-1,history_tokens=1+len(ids),pending_tokens=1)


class Session:
    def __init__(self,model):self.reset()
    def reset(self):self.tokens=[]
    def forward(self,ids):self.tokens.extend(ids)


class Controls(unittest.TestCase):
    def test_corrected_prefix_regenerates_entire_suffix(self):
        case=dict(prompt_tokens=[2],max_output_tokens=4,stop_ids=[6])
        initial=proposal([7,8,9,10])
        bad=dict(verified=False,mismatches=[dict(index=1,stateless_id=3,stateless_margin=.125,stateless_row_sha256='1'*64)])
        good=dict(verified=True,mismatches=[])
        calls=[]
        def generate(session,prompt,maximum,stops,**kwargs):
            calls.append((prompt,maximum))
            # New suffix is [4,stop], never the rejected [9,10].
            session.tokens=prompt+[4]
            tail=proposal([4,6],True);tail.update(committed_tokens=len(session.tokens),history_tokens=len(session.tokens)+1)
            return tail
        with patch.object(target.reference,'Session',Session),patch.object(target.tasks,'generate',side_effect=generate),patch.object(target.tasks,'trajectory',side_effect=[bad,good]):
            result=target.repair(SimpleNamespace(),case,initial=initial)
        self.assertTrue(result['verified']);self.assertEqual(calls,[([2,7,3],2)])
        self.assertEqual(target.sampled_ids(result['generation']),[7,3,4,6])
        self.assertEqual([s['index']for s in result['generation']['samples']],[0,1,2,3])
        self.assertEqual(initial['visible_token_ids'],[7,8,9,10])

    def test_hidden_stop_correction_and_unresolved_bound(self):
        case=dict(prompt_tokens=[2],max_output_tokens=2,stop_ids=[6])
        initial=proposal([7,8]);bad=dict(verified=False,mismatches=[dict(index=0,stateless_id=6,stateless_margin=1.,stateless_row_sha256='2'*64)])
        with patch.object(target.reference,'Session',Session),patch.object(target.tasks,'trajectory',side_effect=[bad,dict(verified=True,mismatches=[])]):
            result=target.repair(SimpleNamespace(),case,initial=initial)
        self.assertEqual(result['generation']['visible_token_ids'],[])
        self.assertEqual(result['generation']['stop_reason'],dict(kind='stop_token',token_id=6))
        with patch.object(target.reference,'Session',Session),patch.object(target.tasks,'trajectory',return_value=bad):
            result=target.repair(SimpleNamespace(),case,initial=initial,max_repairs=0)
        self.assertFalse(result['verified']);self.assertEqual(result['reason'],'repair_bound')

    def test_final_cap_correction_and_cycles_stay_ungraded(self):
        case=dict(prompt_tokens=[2],max_output_tokens=2,stop_ids=[6])
        bad=dict(verified=False,mismatches=[dict(index=1,stateless_id=3,stateless_margin=1.,stateless_row_sha256='3'*64)])
        with patch.object(target.reference,'Session',Session),patch.object(target.tasks,'trajectory',side_effect=[bad,dict(verified=True,mismatches=[])]):
            result=target.repair(SimpleNamespace(),case,initial=proposal([7,8]))
        self.assertEqual(result['generation']['visible_token_ids'],[7,3])
        self.assertEqual(result['generation']['stop_reason'],dict(kind='output_limit'))
        self.assertEqual((result['generation']['committed_tokens'],result['generation']['history_tokens']),(2,3))
        def generate(session,prompt,maximum,stops,**kwargs):
            session.tokens=list(prompt)
            result=proposal([8]);result.update(committed_tokens=len(prompt),history_tokens=len(prompt)+1)
            return result
        mismatches=[dict(verified=False,mismatches=[dict(index=0,stateless_id=v,stateless_margin=1.,stateless_row_sha256='4'*64)])for v in [3,7]]
        with patch.object(target.reference,'Session',Session),patch.object(target.tasks,'generate',side_effect=generate),patch.object(target.tasks,'trajectory',side_effect=mismatches):
            result=target.repair(SimpleNamespace(),case,initial=proposal([7,8]))
        self.assertFalse(result['verified']);self.assertEqual(result['reason'],'proposal_cycle')


if __name__=='__main__':unittest.main()
