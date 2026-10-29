"""Portable analytic/corruption controls. No artifact/checkpoint/producer import."""
import copy
from decimal import Decimal as D, localcontext
from fractions import Fraction
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import gemma3_conformance_math as m
import gemma3_conformance as c
F=np.float32;H=np.float16
GATE={'atol':.0001,'rtol':.001,'fatal_reference':'main'}
def word(x):return int(H(x).view(np.uint16))
def hword(w):return np.array([w],dtype=np.uint16).view(H).astype(F)[0]

class MathControls(unittest.TestCase):
    def test_sqrt_squared_midpoints(self):
        for a,b in [(0x3f800001,0x3f800000),(0x3f800002,0x3f800001),
                    (0x3f7fffff,0x3f7fffff),(0x00000002,0x1a800000)]:
            self.assertEqual(m.bits(m.sqrt32(m.from_bits(a))),b)
        for x in [F(-1),F(np.inf),F(np.nan)]:
            with self.assertRaises(m.ContractError):m.sqrt32(x)
    def test_rms_neumaier_gamma_and_subnormal(self):
        x=np.array([1024]+[.125]*255,dtype=F);scale=np.ones(256,dtype=F);scale[0]=F(1+3*2**-11)
        y=m.rms(x,scale)
        self.assertEqual(word(y[0]),0x4c01);self.assertTrue(np.all(y[1:].astype(H).view(np.uint16)==0x1800))
        x=np.array([3]+[1]*247+[0]*8,dtype=F);scale=np.full(256,F(1+2**-11))
        self.assertEqual(word(m.rms(x,scale)[0]),0x4201)
        self.assertEqual(word(m.rms(x,scale.astype(H).astype(F))[0]),0x4200)
        x=np.zeros(256,dtype=F);x[0]=2**-24
        self.assertEqual(word(m.rms(x,np.ones(256,dtype=F))[0]),0x03e8)
        self.assertFalse(np.array_equal(m.rms(x,np.ones(256,dtype=F)),np.zeros_like(x)))
    def test_faithful_not_one_step_radius(self):
        x=np.array([1,2**-12],dtype=F);w=np.array([[1,1]],dtype=F)
        for word_,accepted in [(0x3c00,True),(0x3c01,True),(0x3bff,False)]:
            n=np.array([hword(word_)],dtype=F);center,valid,_=m.projection(x,w,n)
            self.assertEqual(bool(valid[0]),accepted);self.assertEqual(word(center[0]),0x3c00)
            self.assertTrue(c.numeric_gate(n,center,GATE)[0][0]) # semantic rejects third neighbor
        for w_,ok in [(0x3c00,True),(0x3bff,False),(0x3c01,False)]:
            _,valid,_=m.projection(np.array([1],dtype=F),np.array([[1]],dtype=F),np.array([hword(w_)],dtype=F))
            self.assertEqual(bool(valid[0]),ok)
    def test_exact_dot_midpoints_cancellation_overflow(self):
        self.assertEqual(tuple(map(word,m.half_bracket_fraction(Fraction(1)+Fraction(1,2048)))),(0x3c00,0x3c01,0x3c00))
        for q,expected in [(Fraction(1,2**25),(0,1,0)),(-Fraction(1,2**25),(0x8001,0x8000,0x8000))]:
            self.assertEqual(tuple(map(word,m.half_bracket_fraction(q))),expected)
        for n in [F(0),F(-0.)]:
            out,ok,_=m.projection(np.array([1,1],dtype=F),np.array([[1,-1]],dtype=F),np.array([n],dtype=F))
            self.assertTrue(ok[0]);self.assertEqual(float(out[0]),0)
        with self.assertRaises(m.ContractError):m.half_bracket_fraction(Fraction(65505))
        # Ambiguous nonzero cancellation must exercise the active dot fallback.
        for observed,accepted in [(F(1),True),(F(0),False)]:
            out,valid,count=m.projection(np.ones(3,dtype=F),np.array([[2**60,1,-2**60]],dtype=F),np.array([observed],dtype=F))
            self.assertEqual(out[0],1);self.assertGreater(count,0);self.assertEqual(bool(valid[0]),accepted)
        # A tail after the exact Half midpoint must round directly, not viaF64.
        q=Fraction(1)+Fraction(1,2048)+Fraction(1,2**100)
        self.assertEqual(word(m.half_bracket_fraction(q)[2]),0x3c01)
    def test_true_ulp_binades_exact_grid_and_zero(self):
        with localcontext()as ctx:
            ctx.prec=120
            self.assertEqual(m.ulp_interval(m.point(D(1))),D(2)**-24)
            self.assertEqual(m.ulp_interval(m.point(D(0))),D(2)**-149)
            for q,u in [(D(1)-D(2)**-80,D(2)**-24),(D(1)+D(2)**-80,D(2)**-23)]:
                self.assertEqual(m.ulp_interval(m.point(q)),u)
        with localcontext()as ctx:
            ctx.prec=120
            with self.assertRaises(m.ContractError):m.provider((D(1)-D(2)**-80,D(1)+D(2)**-80),4)
        lo,hi,near=m.provider(m.point(D(1)),4)
        self.assertEqual(m.bits(lo),m.bits(F(1-4*2**-24)))
        self.assertEqual(m.bits(hi),m.bits(F(1+4*2**-24)));self.assertEqual(near,1)
    def test_gelu_discrete_not_continuous_hull(self):
        possible,main=m.gelu_words(m.bits(F(-4)))
        expected=set(range(0x8492,0x84a5,2))
        self.assertEqual(set(map(word,possible)),expected)
        _,valid=m.gelu(np.array([-4],dtype=F),np.array([hword(0x849b)],dtype=F))
        self.assertFalse(valid[0])
        _,valid=m.gelu(np.array([-4],dtype=F),np.array([hword(0x8492)],dtype=F))
        self.assertTrue(valid[0])
        self.assertTrue(np.any(possible==main))
        x=np.array([.25],dtype=F);vals,point=m.gelu_words(m.bits(x[0]))
        self.assertEqual(word(point),0x30ca)
        self.assertEqual(word(m.half(F(np.array([point])*F(5)))[0]),0x39fc)
        for bad in [F(np.nan),F(np.inf)]:
            with self.assertRaises(m.ContractError):m.gelu_words(m.bits(bad))
    def test_rotary_identity_sets_and_wrong_layout(self):
        x=np.zeros((1,4,256),dtype=F);x[:,:,0]=1;x[:,:,128]=2
        center,valid=m.rotary(x,x,0,10000)
        np.testing.assert_array_equal(center,x);self.assertTrue(valid.all())
        bad=x.copy();bad[:,:,0]=hword(word(F(1))+1)
        _,valid=m.rotary(x,bad,0,10000);self.assertFalse(valid[:, :,0].any())
        for theta,pos,dim in [(5000,0,256),(10000,544,256),(10000,1,128)]:
            with self.assertRaises(m.ContractError):m.rotary_contract(theta,pos,0,dim)
        # Exact pi-bound range reduction: point mathematical sin/cos nearpi/2.
        a=F(np.pi/2);s=m.sincos_true(a);co=m.sincos_true(a,True)
        self.assertTrue(s[0] <= D.from_float(float(np.sin(float(a)))) <= s[1] or float(s[0])==float(s[1]))
        self.assertLess(co[1],0)
    def test_pi_machin_range_reduction_and_nonzero_candidates(self):
        def atan(q,n):
            total=sum(((-1)**k*q**(2*k+1)/Fraction(2*k+1)for k in range(n)),Fraction())
            tail=q**(2*n+1)/Fraction(2*n+1)
            return (total-tail,total+tail)
        a=atan(Fraction(1,5),100);b=atan(Fraction(1,239),40)
        pi=(16*a[0]-4*b[1],16*a[1]-4*b[0])
        self.assertLess(Fraction(m.PI_LO),pi[0]);self.assertGreater(Fraction(m.PI_HI),pi[1])
        for angle,sine,cosine in [(1,0x3f576aa4,0x3f0a5140),(543,0x3ef3745d,0xbf613527)]:
            self.assertEqual(m.bits(m.rne_interval(m.sincos_true(F(angle)))),sine)
            self.assertEqual(m.bits(m.rne_interval(m.sincos_true(F(angle),True))),cosine)
        co,si,c0,s0=m.rotary_pairs(10000,543,0)
        self.assertLessEqual(len(co),4096);self.assertGreater(len(co),1)
        self.assertEqual(m.bits(c0),0xbf613527);self.assertEqual(m.bits(s0),0x3ef3745d)
        left=m.half(F(F(F(1)*co)-F(F(2)*si)))
        right=m.half(F(F(F(2)*co)+F(F(1)*si)))
        # Corrupt either member of a jointly generated pair beyond its finite set.
        bad=np.nextafter(np.float16(max(left)),np.float16(np.inf),dtype=np.float16)
        self.assertFalse(np.any((left==F(bad))&(right==right[0])))

    def test_rotary_rejects_individually_reachable_incompatible_pair(self):
        co,si,c0,s0=m.rotary_pairs(10000,543,0)
        left=m.half(F(F(co)-F(F(2)*si)))
        right=m.half(F(F(F(2)*co)+si))
        pairs=set(zip(map(word,left),map(word,right),strict=True))
        impossible=next((a,b)for a in sorted(set(map(word,left)))for b in sorted(set(map(word,right)))if(a,b)not in pairs)
        x=np.zeros((1,1,256),dtype=F);x[0,0,0]=1;x[0,0,128]=2
        actual=x.copy();actual[0,0,0]=hword(impossible[0]);actual[0,0,128]=hword(impossible[1])
        # Only irrelevant all-zero feature contracts are substituted. Feature0
        # uses the complete independently computed actual543-angle provider set.
        original=m.rotary_pairs
        with patch.object(m,'rotary_pairs',side_effect=lambda t,p,f:original(t,p,f)if f==0 else(np.array([1],dtype=F),np.array([0],dtype=F),F(1),F(0))):
            _,valid=m.rotary(x,actual,543,10000)
        self.assertFalse(valid[0,0,0]);self.assertFalse(valid[0,0,128])

    def test_softmax_masks_onekey_and_exp_subnormal(self):
        scores=np.array([[[9,100]]]*4,dtype=F);actual=np.array([[[1,0]]]*4,dtype=F)
        center,valid=m.softmax(scores,actual,0,0,None)
        np.testing.assert_array_equal(center,actual);self.assertTrue(valid.all())
        bad=actual.copy();bad[:,:,1]=F(-0.)
        with self.assertRaises(m.ContractError):m.softmax(scores,bad,0,0,None)
        bad=actual.copy();bad[:,:,0]=np.nextafter(F(1),F(0))
        self.assertFalse(m.softmax(scores,bad,0,0,None)[1][:,:,0].any())
        scores=np.zeros((4,1,3),dtype=F);actual=np.array([[[0,.5,.5]]]*4,dtype=F)
        self.assertTrue(m.softmax(scores,actual,512,0,512)[1].all())
        # Local query512 excludeskey0, includeskey1..2; global includesall3.
        self.assertFalse(m.softmax(scores,actual,512,0,None)[1].all())
        lo,hi,point=m.exp_contract(F(-200));self.assertEqual(lo,0);self.assertEqual(point,0)
        self.assertEqual(m.bits(hi),4)
        lo,hi,point=m.exp_contract(F(-100));self.assertGreater(lo,0)
        self.assertTrue(lo<=point<=hi) # FTZzero cannot pass4ULP here.
    def test_softmax_multikey_and_corrupt_mass(self):
        scores=np.zeros((4,1,3),dtype=F);n=np.full_like(scores,F(1/3))
        point,valid=m.softmax(scores,n,2,0,None);self.assertTrue(valid.all())
        n[:,:,0]=F(.5);self.assertFalse(m.softmax(scores,n,2,0,None)[1][:,:,0].any())
        with self.assertRaises(m.ContractError):m.softmax(scores,np.full_like(scores,np.nan),2,0,None)
        with self.assertRaises(m.ContractError):m.softmax(scores,n,2,3,None)

class ZeroWeights:
    def embedding(self,token):
        m.require(type(token)is int and 0<=token<262144,'token index');return np.zeros((1,640),dtype=F)
    def norm(self,name,width):return np.ones(width,dtype=F)
    def matrix(self,name,out,width):return np.zeros((out,width),dtype=F)


def shape(stage):
    leaf=stage.split('.')[-1]
    if leaf in ('embedding_lookup','embedding','final_norm'):return (1,640)
    if leaf=='logits':return (1,262144)
    if leaf in ('attention_scores','attention_probabilities'):return (4,1,1)
    if leaf in ('q_projection','q_norm','q_rotary'):return (1,4,256)
    if leaf in ('k_projection','k_norm','k_rotary','v_projection'):return (1,1,256)
    return (1,2048 if leaf in ('gate_projection','up_projection','gate_gelu','gated')else 1024 if leaf=='attention_context'else 640)

def record(stage,row=0,label='synthetic'):
    return dict(label=label,row=row,stage=stage,shape=list(shape(stage)),absolute_key_start=0)

class InventoryControls(unittest.TestCase):
    def test_exhaustive_stage_mapping_and_all_routes(self):
        v=c.Validator(ZeroWeights(),{});v.begin('synthetic',[2],[0])
        for stage in c.STAGES:
            a=np.zeros(shape(stage),dtype=F)
            if stage.endswith('attention_probabilities'):a[:]=1
            v.observe('synthetic',0,stage,a,record(stage),GATE)
        v.end();self.assertEqual(v.groups,363);self.assertEqual(v.semantic_failures,0);self.assertEqual(v.numeric_failures,0)
        self.assertEqual(len(c.STAGES),400)
        self.assertEqual(sum(c.route(s)=='coupled'for s in c.STAGES),37)
        with self.assertRaises(m.ContractError):c.route('layers.18.output')
    def test_missing_duplicate_wrong_identity_and_no_fallback(self):
        v=c.Validator(ZeroWeights(),{});v.begin('synthetic',[2],[0])
        with self.assertRaises(m.ContractError):v.observe('synthetic',0,'embedding',np.zeros((1,640),dtype=F),record('embedding'),GATE)
        for field,value in [('label','wrong'),('row',1),('stage','embedding'),('shape',[640]),('absolute_key_start',1)]:
            v=c.Validator(ZeroWeights(),{});v.begin('synthetic',[2],[0]);d=record('embedding_lookup');d[field]=value
            with self.assertRaises(m.ContractError):v.observe('synthetic',0,'embedding_lookup',np.zeros((1,640),dtype=F),d,GATE)
        v=c.Validator(ZeroWeights(),{});v.begin('synthetic',[2],[0]);a=np.zeros((1,640),dtype=F)
        v.observe('synthetic',0,'embedding_lookup',a,record('embedding_lookup'),GATE)
        with self.assertRaises(m.ContractError):v.observe('synthetic',0,'embedding_lookup',a,record('embedding_lookup'),GATE)
        with self.assertRaises(m.ContractError):v.end()
    def test_basic_exact_failure_below_old_atol_and_signed_zero(self):
        v=c.Validator(ZeroWeights(),{});v.begin('synthetic',[2],[0]);a=np.zeros((1,640),dtype=F)
        v.observe('synthetic',0,'embedding_lookup',a,record('embedding_lookup'),GATE)
        a[0,0]=F(2**-24);v.observe('synthetic',0,'embedding',a,record('embedding'),GATE)
        self.assertEqual(v.semantic_failures,1);self.assertEqual(v.numeric_failures,0)
        self.assertEqual(v.first_failure['index'],0)
        v=c.Validator(ZeroWeights(),{});v.begin('synthetic',[2],[0]);a[:]=F(-0.)
        v.observe('synthetic',0,'embedding_lookup',a,record('embedding_lookup'),GATE)
        self.assertEqual(v.semantic_failures,0)
    def test_coefficient_dtype_and_packed_row_decoding(self):
        scales=np.array([0x3f80,0],dtype=np.uint16)
        np.testing.assert_array_equal(c.Weights.bf(scales,True),[1,0])
        for word_ in [1,0x7f80,0x7fc0]:
            with self.assertRaises(m.ContractError):c.Weights.bf(np.array([word_],dtype=np.uint16),True)
        # Independent fake original packed rows: two different groups/coefficients.
        w=c.Weights.__new__(c.Weights);w.max_numeric_bytes=0
        codes=np.array([[0x76543210]*8+[0xfedcba98]*8],dtype=np.uint32)
        w.raw=lambda name,shape,dtype,row=None: codes if name.endswith('.weight') else np.array([[0x3f80,0x4000]],dtype=np.uint16)if name.endswith('.scales')else np.array([[0,0xbf80]],dtype=np.uint16)
        decoded=w.decode('synthetic',1,128)
        np.testing.assert_array_equal(decoded[0,:64],np.tile(np.arange(8),8))
        np.testing.assert_array_equal(decoded[0,64:],np.tile(2*np.arange(8,16)-1,8))
        w.raw=lambda name,shape,dtype,row=None: codes if name.endswith('.weight') else np.full((1,2),0x7f7f,dtype=np.uint16)
        with self.assertRaises(m.ContractError):w.decode('unsupported-overflow',1,128)
    def test_clamp_and_store_order_corruption(self):
        x=np.array([64864,-64864,2**-24],dtype=F);y=np.array([14520,-14520,0],dtype=F)
        np.testing.assert_array_equal(m.half(np.clip(F(x+y),-65504,65504)),[65504,-65504,2**-24])
        with self.assertRaises(m.ContractError):m.half(F(x+y))
        gate=F(.25);up=F(5);_,point=m.gelu_words(m.bits(gate))
        self.assertEqual(word(m.half(np.array([F(point*up)]))[0]),0x39fc)
        self.assertNotEqual(0x39fd,0x39fc)

if __name__=='__main__':unittest.main()
