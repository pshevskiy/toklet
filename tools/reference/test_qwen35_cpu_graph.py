import copy
import hashlib
import math
import struct
import unittest
from pathlib import Path
import qwen35_cpu_graph as graph


class Controls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw,cls.table,cls.matrices,cls.vectors,cls.backing=graph.artifact()
        cls.staged=graph.forward(cls.matrices,cls.vectors,graph.TOKENS,0,True)

    def test_wire_fixture_is_frozen_and_nonsquare(self):
        root=Path(__file__).resolve().parents[2]/'tests/fixtures/qwen35-cpu-graph-v1'
        self.assertEqual(self.raw,(root/'weights.bin').read_bytes())
        self.assertEqual(sum(v['kind']=='matrix' for v in self.table.values()),32)
        self.assertEqual(sum(v['kind']=='bf16' for v in self.table.values()),23)
        self.assertNotEqual(graph.GEOMETRY['hidden'],graph.GEOMETRY['query_heads']*graph.GEOMETRY['head_dim'])
        for name,spec in self.table.items():
            if 'conv1d' in name:self.assertEqual(spec['shape'][-2:],[4,1])

    def test_per_head_gate_split_is_not_flat_halving(self):
        rows={r['stage']:r['values'] for r in self.staged if r['token']==0 and r['layer']==3}
        raw=rows['attention_q_gate'];dim=graph.GEOMETRY['head_dim'];heads=graph.GEOMETRY['query_heads']
        expected=[v for h in range(heads) for v in raw[2*h*dim:(2*h+1)*dim]]
        self.assertEqual(rows['attention_q'],expected)
        self.assertNotEqual(expected,raw[:len(raw)//2])

    def test_partial_rotary_preserves_inactive_suffix_and_uses_prefix_pairs(self):
        for staged in [False,True]:
            m=graph.Math(staged);x=[float(i-20)/32 for i in range(64)]
            self.assertEqual(m.rotary(x,2,32,8,0),x)
            result=m.rotary(x,2,32,8,11)
            for head in range(2):self.assertEqual(result[head*32+8:(head+1)*32],x[head*32+8:(head+1)*32])
            co=m.r(math.cos(11));si=m.r(math.sin(11))
            self.assertEqual(result[0],m.r(m.r(x[0]*co)-m.r(x[4]*si)))
            self.assertNotEqual(result[0],m.r(m.r(x[0]*co)-m.r(x[16]*si)))

    def test_l2_epsilon_is_on_sum_not_mean(self):
        rows={r['stage']:r['values'] for r in graph.forward(self.matrices,self.vectors,[1],0,False) if r['layer']==0}
        width=graph.GEOMETRY['key_dim'];q=rows['delta_convolved'][:width]
        denom=math.sqrt(math.fsum(v*v for v in q)+1e-6)
        expected=[v/denom/math.sqrt(width) for v in q]
        self.assertEqual(rows['delta_q'][:width],expected)
        old=[v/math.sqrt(math.fsum(x*x for x in q)+width*1e-6)/math.sqrt(width) for v in q]
        self.assertNotEqual(expected,old)

    def test_future_change_preserves_every_earlier_stage(self):
        changed=graph.forward(self.matrices,self.vectors,graph.TOKENS[:-1]+(31,),0,True)
        self.assertEqual([r for r in self.staged if r['token']<16],[r for r in changed if r['token']<16])
        self.assertNotEqual(self.staged[-1]['values'],changed[-1]['values'])

    def test_reference_does_not_mutate_original_weights(self):
        matrices=copy.deepcopy(self.matrices);vectors=copy.deepcopy(self.vectors)
        graph.forward(matrices,vectors,[1,7],4079,True)
        self.assertEqual(matrices,self.matrices);self.assertEqual(vectors,self.vectors)
        self.assertEqual(graph.f32(-0.0).hex(),(-0.0).hex())
        self.assertEqual(struct.pack('<f',graph.bf16(-0.0)[1]),bytes.fromhex('00000080'))
        self.assertEqual(hashlib.sha256(graph.artifact()[0]).digest(),hashlib.sha256(self.raw).digest())


if __name__=='__main__':unittest.main()
