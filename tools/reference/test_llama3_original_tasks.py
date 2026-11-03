import unittest
from unittest.mock import patch
import numpy as np
import llama3_original_tasks as subject


class ScriptedSession:
    capacity=566
    def __init__(self, sequence):self.position=0;self.history=[];self.sequence=iter(sequence)
    def forward(self, ids):
        self.position+=len(ids);self.history+=ids
        row=np.zeros(128256,dtype=np.float32);token=next(self.sequence)
        if token is not None:row[token]=1
        return row


class TaskGenerationTests(unittest.TestCase):
    def test_hidden_stops_are_sampled_not_visible_or_committed(self):
        for stop in subject.inputs.STOPS:
            session=ScriptedSession([7,stop])
            result=subject.generate(session,[1,2],4,subject.inputs.STOPS)
            self.assertEqual(result['visible_token_ids'],[7])
            self.assertEqual(result['stop_reason'],dict(kind='stop_token',token_id=stop))
            self.assertEqual((session.position,session.history),(3,[1,2,7]))
            self.assertEqual((result['history_tokens'],result['pending_tokens']),(4,1))
        result=subject.generate(ScriptedSession([128009]),[1],4,subject.inputs.STOPS)
        self.assertEqual(result['visible_token_ids'],[])
    def test_limit_and_tie_do_not_evaluate_final_visible_token(self):
        session=ScriptedSession([7,8]);result=subject.generate(session,[1,2],2,subject.inputs.STOPS)
        self.assertEqual(result['visible_token_ids'],[7,8]);self.assertEqual(session.history,[1,2,7])
        self.assertEqual(result['stop_reason'],dict(kind='output_limit'))
        self.assertEqual(subject.generate(ScriptedSession([None]),[1],1,subject.inputs.STOPS)['visible_token_ids'],[0])
        for maximum in [0,True,566]:
            session=ScriptedSession([7])
            with self.assertRaises(ValueError):subject.generate(session,[1],maximum,subject.inputs.STOPS)
            self.assertEqual(session.position,0)
    def test_trajectory_positions_include_terminal_stop_but_no_extra_limit_prediction(self):
        def distribution(ids):
            rows=np.zeros((len(ids),128256),dtype=np.float32)
            for index,token in enumerate(ids):rows[index,token]=1
            return rows
        result=dict(visible_token_ids=[7,8],stop_reason=dict(kind='output_limit'),samples=[dict(margin=1.),dict(margin=1.)])
        with patch.object(subject.runtime.graph,'quality_forward',return_value=distribution([7,8]))as call:
            replay=subject.trajectory(None,[1,2],result,subject.inputs.STOPS)
        self.assertTrue(replay['verified']);self.assertFalse(replay['terminal_stop_checked'])
        self.assertEqual(call.call_args.args,(None,[1,2,7,8],[1,2]))
        result=dict(visible_token_ids=[7],stop_reason=dict(kind='stop_token',token_id=128009),samples=[dict(margin=1.),dict(margin=1.)])
        with patch.object(subject.runtime.graph,'quality_forward',return_value=distribution([7,128009]))as call:
            replay=subject.trajectory(None,[1,2],result,subject.inputs.STOPS)
        self.assertTrue(replay['verified']);self.assertTrue(replay['terminal_stop_checked'])
        self.assertEqual(call.call_args.args,(None,[1,2,7],[1,2]))
        with patch.object(subject.runtime.graph,'quality_forward',return_value=distribution([7,8])):
            mismatch=subject.trajectory(None,[1,2],result,subject.inputs.STOPS)
        self.assertFalse(mismatch['verified']);self.assertEqual(mismatch['mismatches'][0]['expected_id'],128009)
