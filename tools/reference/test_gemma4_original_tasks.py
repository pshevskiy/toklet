"""Task stops/pending samples, lowest-ID ties and withheld divergent replay."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
import gemma4_original_tasks as target
import gemma4_resident_generation as resident
import test_gemma4_upstream_stream as tiny


class Scripted:
    def __init__(self,sequence):
        self.sequence=iter(sequence);self.tokens=[];self.replay_required=False
        self.cache=SimpleNamespace(get_seq_length=lambda:len(self.tokens))
    def forward(self,ids):
        self.tokens.extend(ids)
        value=torch.zeros((1,1,17),dtype=torch.bfloat16);token=next(self.sequence)
        if token is not None:value[0,0,token]=1
        return value


class Controls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_stops_caps_pending_and_ties(self):
        for stop in [1,6,5]:
            session=Scripted([7,stop]);result=target.generate(session,[2,3],4,[1,6,5])
            self.assertEqual(result['visible_token_ids'],[7])
            self.assertEqual(result['stop_reason'],dict(kind='stop_token',token_id=stop))
            self.assertEqual(session.tokens,[2,3,7])
            self.assertEqual((result['history_tokens'],result['pending_tokens']),(4,1))
        session=Scripted([None,7]);result=target.generate(session,[2],2,[1])
        self.assertEqual(result['visible_token_ids'],[0,7])
        self.assertEqual(session.tokens,[2,0])
        self.assertEqual(result['stop_reason'],dict(kind='output_limit'))
        with self.assertRaises(ValueError):target.generate(Scripted([1]),[0]*567,1,[1])

    def test_hidden_stop_and_divergent_trajectory_withhold_verification(self):
        result=target.generate(Scripted([7,6]),[2,3],4,[6])
        model=SimpleNamespace(config=SimpleNamespace(vocab_size=17))
        def rows(selected):
            for token in selected:
                value=torch.zeros((1,1,17),dtype=torch.bfloat16);value[0,0,token]=1
                yield value
        with patch.object(target,'teacher_rows',return_value=rows([7,6])):
            replay=target.trajectory(model,[2,3],result,[6])
            self.assertTrue(replay['verified']);self.assertTrue(replay['terminal_stop_checked'])
        with patch.object(target,'teacher_rows',return_value=rows([8,6])):
            replay=target.trajectory(model,[2,3],result,[6])
            self.assertFalse(replay['verified']);self.assertEqual(len(replay['mismatches']),1)

    def test_tiny_real_controller_and_single_text_replay(self):
        config,reader,_=tiny.fixture();reader.config=config
        model=resident.Resident(reader);session=resident.Session(model)
        result=target.generate(session,[2,7,13],4,[1])
        replay=target.trajectory(model,[2,7,13],result,[1])
        self.assertTrue(replay['verified'])
        self.assertEqual(len(replay['rows']),len(result['samples']))


if __name__=='__main__':unittest.main()
