import numpy as np
import unittest
import llama3_original_fp32 as graph
import llama3_original_session as subject
from test_llama3_original_fp32 import TinyReader


class SessionTests(unittest.TestCase):
    def test_offset_rotary_matches_padded_default_byte_for_byte(self):
        values=(np.arange(2*2*64,dtype=np.float32).reshape(2,2,64)-100)/np.float32(64)
        for offset in [0,17,510,511,512,564]:
            padded=np.concatenate((np.zeros((offset,2,64),dtype=np.float32),values))
            np.testing.assert_array_equal(graph.rotary(values,offset),graph.rotary(padded)[offset:])
        for offset in [-1,True,1024]:
            with self.assertRaises(ValueError):graph.rotary(values,offset)

    def test_cached_rows_match_stateless_and_reset_ignores_stale_rows(self):
        reader=TinyReader();session=subject.Session(reader,8);ids=[1,2,3,4,5,6]
        session.forward(ids[:3])
        np.testing.assert_array_equal(session.logits,graph.forward(reader,ids[:3])[-1])
        for count in [4,5,6]:
            row=session.forward([ids[count-1]])
            np.testing.assert_allclose(row,graph.forward(reader,ids[:count])[-1],atol=1e-5,rtol=1e-5)
            self.assertEqual(subject.greedy(row),subject.greedy(graph.forward(reader,ids[:count])[-1]))
        session.reset();row=session.forward([7,8])
        np.testing.assert_array_equal(row,subject.Session(reader,8).forward([7,8]))

    def test_overflow_and_layer_failure_preserve_prefix_and_retry(self):
        class FailingReader(TinyReader):
            fail=False
            def weight(self,name,start=None,stop=None):
                if self.fail and name=='model.layers.1.input_layernorm.weight':
                    raise ValueError('fixture layer failure')
                return super().weight(name,start,stop)
        reader=FailingReader();session=subject.Session(reader,3);session.forward([1,2])
        before=(session.position,list(session.history),session.logits.copy(),session.k[:,:,:2].copy(),session.v[:,:,:2].copy())
        reader.fail=True
        with self.assertRaises(ValueError):session.forward([3])
        self.assertEqual((session.position,session.history),before[:2])
        np.testing.assert_array_equal(session.logits,before[2]);np.testing.assert_array_equal(session.k[:,:,:2],before[3]);np.testing.assert_array_equal(session.v[:,:,:2],before[4])
        reader.fail=False;actual=session.forward([3]);np.testing.assert_allclose(actual,graph.forward(reader,[1,2,3])[-1],atol=1e-5,rtol=1e-5)
        before=(session.position,list(session.history),session.logits.copy())
        with self.assertRaises(ValueError):session.forward([4])
        self.assertEqual((session.position,session.history),before[:2]);np.testing.assert_array_equal(session.logits,before[2])

    def test_history_allocation_failure_precedes_any_cache_write(self):
        class FailingHistory(list):
            def __add__(self,other):raise MemoryError('fixture history allocation')
        session=subject.Session(TinyReader(),3);session.k.fill(42);session.v.fill(43);session.forward([1])
        before=(session.position,list(session.history),session.logits.copy(),session.k.copy(),session.v.copy())
        session.history=FailingHistory(session.history)
        with self.assertRaises(MemoryError):session.forward([2])
        self.assertEqual((session.position,session.history),before[:2])
        np.testing.assert_array_equal(session.logits,before[2])
        np.testing.assert_array_equal(session.k.view(np.uint32),before[3].view(np.uint32))
        np.testing.assert_array_equal(session.v.view(np.uint32),before[4].view(np.uint32))

    def test_lowest_id_tie_and_invalid_inputs(self):
        self.assertEqual(subject.greedy(np.zeros(11,dtype=np.float32)),0)
        with self.assertRaises(ValueError):subject.greedy(np.array([np.nan],dtype=np.float32))
        for capacity in [0,567,True]:
            with self.assertRaises(ValueError):subject.Session(TinyReader(),capacity)
        session=subject.Session(TinyReader(),3)
        for ids in [[],[True],[11],[-1],[1]*4]:
            with self.assertRaises(ValueError):session.forward(ids)
        self.assertEqual(session.position,0)
