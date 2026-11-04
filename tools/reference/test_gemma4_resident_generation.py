"""Tiny resident/streamed and cached/fresh reference qualification."""
import unittest
import torch
import gemma4_upstream_stream as stream
import gemma4_resident_generation as target
import test_gemma4_upstream_stream as tiny


class Controls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_resident_streamed_and_cached_shared_kv(self):
        config,reader,contract=tiny.fixture();reader.config=config
        resident=target.Resident(reader)
        ids=contract['tokens']
        expected=stream.Streamed(config,reader)(torch.tensor([ids],dtype=torch.int64))[:,-1:]
        self.assertEqual(stream.tensor_bytes(resident.stateless(ids)),stream.tensor_bytes(expected))
        self.assertIs(resident.model.lm_head.weight,resident.model.model.embed_tokens.weight)
        self.assertEqual(resident.model.model.embed_tokens_per_layer.weight.device.type,'meta')
        session=target.Session(resident);offset=0
        for end in [3,4,8,9]:
            actual=session.forward(ids[offset:end]);fresh=resident.stateless(ids[:end]);offset=end
            self.assertEqual(actual.argmax(-1).tolist(),fresh.argmax(-1).tolist())
            self.assertEqual(stream.tensor_bytes(actual),stream.tensor_bytes(fresh))
            self.assertEqual(session.tokens,ids[:end])
        previous=session.cache;session.reset()
        self.assertIsNot(session.cache,previous);self.assertEqual(session.tokens,[])
        self.assertEqual(stream.tensor_bytes(session.forward(ids)),stream.tensor_bytes(expected))

    def test_partial_weight_load_and_partial_cache_failure_release(self):
        config,reader,_=tiny.fixture();reader.config=config
        reader.fail_tensor='model.layers.1.mlp.down_proj.weight'
        error=None
        try:target.Resident(reader)
        except RuntimeError as caught:error=caught
        self.assertIsNotNone(error)
        trace=error.__traceback__;failed=None
        while trace is not None:
            instance=trace.tb_frame.f_locals.get('self')
            if isinstance(instance,target.Resident):failed=instance
            trace=trace.tb_next
        self.assertIsNotNone(failed)
        self.assertTrue(all(v.device.type=='meta'for v in failed.model.state_dict().values()))
        reader.fail_tensor=None
        resident=target.Resident(reader);session=target.Session(resident)
        def fail(module,args):raise RuntimeError('injected mid-forward failure')
        hook=resident.model.model.layers[1].register_forward_pre_hook(fail)
        with self.assertRaises(RuntimeError):session.forward([2])
        self.assertTrue(session.replay_required)
        self.assertEqual(session.tokens,[])
        self.assertGreater(session.cache.get_seq_length(),0)
        with self.assertRaises(ValueError):session.forward([2])
        hook.remove();session.reset()
        self.assertEqual(stream.tensor_bytes(session.forward([2])),stream.tensor_bytes(resident.stateless([2])))

    def test_invalid_capacity_and_failed_step_require_replay(self):
        config,reader,contract=tiny.fixture();reader.config=config
        resident=target.Resident(reader);session=target.Session(resident)
        session.forward(contract['tokens'][:3]);before=list(session.tokens)
        for invalid in [[],[True],[17],[0]*568]:
            with self.assertRaises(ValueError):session.forward(invalid)
            self.assertEqual(session.tokens,before)
            self.assertFalse(session.replay_required)
        session.tokens=[0]*567
        with self.assertRaises(ValueError):session.forward([0])
        self.assertFalse(session.replay_required)
        session.reset();reader.fail_tensor=stream.PLE
        with self.assertRaises(RuntimeError):session.forward([2])
        self.assertTrue(session.replay_required)
        with self.assertRaises(ValueError):session.forward([2])
        reader.fail_tensor=None;session.reset()
        self.assertEqual(stream.tensor_bytes(session.forward([2])),stream.tensor_bytes(resident.stateless([2])))


if __name__=='__main__':unittest.main()
