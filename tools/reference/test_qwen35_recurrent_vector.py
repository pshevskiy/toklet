"""Bitwise primitive and complete frozen tied/untied trace controls."""
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import time
import unittest
import numpy as np
import bf16_source
import build_qwen35_vector_graph as builder
import qwen35_cpu_graph as scalar
import qwen35_recurrent_vector as candidate
import qwen35_vector_graph as vector
import qwen35_9b_weights as weights
from test_qwen35_9b_weights import wire

ROOT=Path(__file__).resolve().parents[2]


def scalar_step(q,k,v,decay,beta,state,shape,staged):
    hk,hv,dk,dv=shape;m=scalar.Math(staged);new=[0.]*len(state);out=[]
    for h in range(hv):
        kh=h//(hv//hk);key=k[kh*dk:(kh+1)*dk];query=q[kh*dk:(kh+1)*dk]
        for d in range(dv):
            base=(h*dv+d)*dk;decayed=[m.r(state[base+j]*decay[h]) for j in range(dk)]
            correction=m.r(m.r(v[h*dv+d]-m.dot(decayed,key))*beta[h])
            updated=[m.r(s+m.r(a*correction)) for s,a in zip(decayed,key,strict=True)]
            new[base:base+dk]=updated;out.append(m.dot(updated,query))
    return new,out


def bits(value,staged):return np.asarray(value,dtype='<f4' if staged else '<f8').tobytes()


class Controls(unittest.TestCase):
    def test_generated_source_has_exact_inverse_and_unchanged_original(self):
        proof=builder.validate();self.assertTrue(proof['inverse_patch_exact'])
        original=(ROOT/'tools/reference/qwen35_cpu_graph.py').read_text()
        self.assertEqual(builder.generated(original).encode(),(ROOT/'tools/reference/qwen35_vector_graph.py').read_bytes())
        self.assertEqual(scalar.GATE,vector.GATE)
        for bad in [original.replace(builder.START,'',1),original.replace(builder.END,'',1),original+builder.START]:
            with self.assertRaises(ValueError):builder.generated(bad)

    def test_grouped_real_shapes_and_repeated_steps_match_both_profiles_bitwise(self):
        rng=np.random.default_rng(87221)
        for shape in [(1,1,1,1),(2,4,16,16),(3,6,7,5),(16,32,128,128)]:
            hk,hv,dk,dv=shape
            for staged in [False,True]:
                values=[rng.uniform(-.2,.2,n) for n in [hk*dk,hk*dk,hv*dv,hv,hv,hv*dv*dk]]
                values[3]=rng.uniform(.1,.9,hv);values[4]=rng.uniform(.1,.9,hv)
                dtype=np.float32 if staged else np.float64
                values=[v.astype(dtype).tolist() for v in values]
                for _ in range(2):
                    want=scalar_step(*values,shape,staged);got=candidate.step(*values,shape,staged)
                    for a,b in zip(want,got,strict=True):self.assertEqual(bits(a,staged),bits(b,staged))
                    values[-1]=want[0]

    def test_signed_zero_subnormal_and_cancellation_operands_preserve_bits(self):
        for staged in [False,True]:
            tiny=float(np.nextafter(np.float32(0),np.float32(1))) if staged else math.ulp(0.)
            dtype=np.float32 if staged else np.float64
            for values in [([1.,0.,0.,0.],[1.,0.,0.,0.],[0.],[1.],[0.],[tiny,-tiny,tiny,-tiny]),
                ([1.,-1.,1.,-1.],[-0.,0.,-0.,0.],[0.],[1.],[1.],[-0.,-0.,-0.,-0.]),
                ([1.,1.,1.,1.],[1.,-1.,1.,-1.],[tiny],[.5],[.25],[tiny,-tiny,tiny,-tiny]),
                ([1.,1.,1.,1.],[1.,1.,-1.,-1.],[.125],[1.],[.5],[2.**20,2.**-20,-2.**20,-2.**-20])]:
                arrays=[np.asarray(v,dtype=dtype).tolist() for v in values]
                want=scalar_step(*arrays,(1,1,4,1),staged);got=candidate.step(*arrays,(1,1,4,1),staged)
                for a,b in zip(want,got,strict=True):self.assertEqual(bits(a,staged),bits(b,staged))

    def test_invalid_shapes_operands_profile_and_overflow_reject(self):
        args=[[.25],[.5],[.125],[.75],[.5],[.25]]
        for shape in [(True,1,1,1),(0,1,1,1),(1,129,1,1),(2,3,1,1),(1,1,1),None]:
            with self.assertRaises(ValueError):candidate.step(*args,shape,True)
        for profile in [None,0,1,'fp32']:
            with self.assertRaises(ValueError):candidate.step(*args,(1,1,1,1),profile)
        for changed in [[],[float('nan')],[float('inf')],[[.25]],[True],[1]]:
            broken=copy.deepcopy(args);broken[0]=changed
            with self.assertRaises(ValueError):candidate.step(*broken,(1,1,1,1),True)
        unrounded=copy.deepcopy(args);unrounded[0]=[.1]
        with self.assertRaises(ValueError):candidate.step(*unrounded,(1,1,1,1),True)
        for staged,big in [(False,1e308),(True,float(np.finfo(np.float32).max))]:
            broken=copy.deepcopy(args);broken[-1]=[big];broken[3]=[2.]
            with self.assertRaises(ValueError):candidate.step(*broken,(1,1,1,1),staged)

    def test_tied_frozen_full_traces_match_every_fp32_and_fp64_byte(self):
        _,_,matrices,vectors,_=scalar.artifact()
        for _,tokens,offset in scalar.CASES:
            for staged in [False,True]:
                want=scalar.forward(matrices,vectors,tokens,offset,staged)
                got=vector.forward(matrices,vectors,tokens,offset,staged)
                self.assertEqual(len(want),len(got))
                for a,b in zip(want,got,strict=True):
                    for k in ['token','layer','stage']:self.assertEqual(a[k],b[k])
                    self.assertEqual(bits(a['values'],staged),bits(b['values'],staged))

    def test_untied_dense_fixture_and_packed_paths_stay_bit_exact(self):
        folder=ROOT/'tests/fixtures/qwen35-untied-graph-v1';m=json.loads((folder/'manifest.json').read_text())
        reference=(folder/'reference.bin').read_bytes();raw=b'X'+(folder/'weights.bin').read_bytes()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'weights';path.write_bytes(raw);deadline=time.monotonic()+120
            with bf16_source.Source(path,len(raw),hashlib.sha256(raw).hexdigest(),deadline=deadline) as owner:
                matrices,vectors=weights.bind(owner,wire(m),1,m['geometry'],deadline)
                dense={name:[row for first in range(0,value.rows.rows,128)
                    for row in value.rows.read_rows(first,min(first+128,value.rows.rows)).tolist()]
                    for name,value in matrices.items()}
                embedding=dense[weights.EMBED];head=dense[weights.HEAD]
                class DenseOutputMath(scalar.Math):
                    def linear(self,values,matrix):
                        return super().linear(values,head if matrix is embedding else matrix)
                for case in m['cases']:
                    for profile,staged in [('ideal_fp64',False),('staged_fp32',True)]:
                        got=vector.forward(matrices,vectors,case['tokens'],case['position_offset'],staged,
                            geometry=m['geometry'],math_type=weights.math_type(matrices))
                        old=scalar.forward(matrices,vectors,case['tokens'],case['position_offset'],staged,
                            geometry=m['geometry'],math_type=weights.math_type(matrices))
                        dense_got=vector.forward(dense,vectors,case['tokens'],case['position_offset'],staged,
                            geometry=m['geometry'],math_type=DenseOutputMath)
                        self.assertEqual(len(got),len(old));self.assertEqual(len(dense_got),len(case['profiles'][profile]))
                        for a,b in zip(got,old,strict=True):
                            for k in ['token','layer','stage']:self.assertEqual(a[k],b[k])
                            self.assertEqual(bits(a['values'],staged),bits(b['values'],staged))
                        for a,b in zip(dense_got,case['profiles'][profile],strict=True):
                            for k in ['token','layer','stage']:self.assertEqual(a[k],b[k])
                            self.assertEqual(bits(a['values'],staged),reference[b['offset']:b['offset']+b['bytes']])


if __name__=='__main__':unittest.main()
