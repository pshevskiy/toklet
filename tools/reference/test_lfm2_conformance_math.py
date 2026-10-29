"""Synthetic controls only: no checkpoint, captured target or graph imports."""
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lfm2_conformance_math as m
F = np.float32; H = np.float16
GATE = {'atol': 1e-4, 'rtol': .001}


def word(x): return int(H(x).view(np.uint16))
def hword(x): return np.array([x], dtype=np.uint16).view(H).astype(F)[0]
def row(values, width=64):
    out = np.zeros((1, width), dtype=F); out[0, :len(values)] = values; return out

def projection(w, x, observed):
    return m.projection(row(x, len(w)), np.asarray([w], dtype=F), np.array([[observed]], dtype=F))


def rne_fraction(value):
    seed = F(float(value))
    before = np.nextafter(seed, F(-np.inf), dtype=F)
    after = np.nextafter(seed, F(np.inf), dtype=F)
    choices = [before, seed, after]
    return min(choices, key=lambda x: (abs(Fraction.from_float(float(x)) - value), m.bits(x) & 1))


def trig_small(exact, cosine):
    total = Fraction(1) if cosine else exact
    term = total
    for k in range(1, 40):
        term *= -exact * exact / ((2*k-1)*(2*k) if cosine else (2*k)*(2*k+1))
        total += term
    k = 40
    tail = abs(term * exact * exact / ((2*k-1)*(2*k) if cosine else (2*k)*(2*k+1)))
    lo, hi = rne_fraction(total-tail), rne_fraction(total+tail)
    if lo != hi: raise AssertionError('independent rational trig point inconclusive')
    return lo


class LocalContracts(unittest.TestCase):
    def test_direct_rms_epsilon_gain_and_neumaier(self):
        for width in (64, 1024):
            x = np.zeros(width, dtype=F); x[0] = hword(1)
            gains = np.ones(width, dtype=F)
            self.assertEqual(word(m.rms(x, gains)[0]), 0x013c)
            x[0] = hword(0x8001)
            self.assertEqual(word(m.rms(x, gains)[0]), 0x813c)
            self.assertTrue(np.all(m.rms(np.ones(width, dtype=F), np.zeros(width, dtype=F)) == 0))
            gains.fill(F(2**-25)); x.fill(1); x[0] = 8
            # This checks preservation of a raw BF16 gain below the Half grid;
            # premature Half coefficient storage makes the entire result zero.
            self.assertGreater(word(m.rms(x, gains)[0]), 0)
            self.assertTrue(np.all(m.rms(x, gains.astype(H).astype(F)) == 0))
        x = np.array([1024] + [.125]*1023, dtype=F)
        self.assertEqual(m.square_sum(x), F(1048592))
        naive = F(0)
        for v in x: naive = F(naive + F(v*v))
        self.assertEqual(naive, F(1048576))
        for bad in (np.ones(256, dtype=F), np.ones(63, dtype=F)):
            with self.assertRaises(m.ContractError): m.rms(bad, bad)
        gains = np.ones(64, dtype=F); gains[0] = np.nextafter(F(1), F(2), dtype=F)
        with self.assertRaises(m.ContractError): m.rms(np.ones(64, dtype=F), gains)

    def test_sqrt_and_stored_operand_types_fail_closed(self):
        for x, expected in [(0x3f800001, 0x3f800000), (0x3f800002, 0x3f800001), (2, 0x1a800000)]:
            value = np.array([x], dtype=np.uint32).view(F)[0]
            self.assertEqual(m.bits(m.sqrt32(value)), expected)
        for x in (np.ones(64, dtype=np.float64), np.ones(64, dtype=np.uint32),
                  np.full(64, np.nan, dtype=F), np.full(64, .1, dtype=F)):
            with self.assertRaises(m.ContractError): m.rms(x, np.ones(64, dtype=F))

    def test_basic_stores_signed_zero_subnormal_and_no_clamp(self):
        a = np.array([hword(x) for x in (0, 0x8000, 1, 1, 3, 0x7bff)], dtype=F)
        b = np.array([1, 1, 1, .5, .5, 1], dtype=F)
        result = m.binary(a, b, 'product')
        self.assertEqual(list(result.astype(H).view(np.uint16)), [0, 0x8000, 1, 0, 2, 0x7bff])
        with self.assertRaises(m.ContractError): m.binary(np.array([65504], dtype=F), np.array([2], dtype=F), 'product')
        with self.assertRaises(m.ContractError): m.binary(np.array([65504], dtype=F), np.array([65504], dtype=F), 'residual')
        with self.assertRaises(m.ContractError): m.binary(np.array([np.inf], dtype=F), np.array([0], dtype=F), 'product')
        with self.assertRaises(m.ContractError): m.binary(a, b, 'clamped_residual')

    def test_original_gather_and_bcx_views(self):
        original = np.ones(1024, dtype=F); original[0] = F(1 + 2**-11)
        self.assertEqual(word(m.gather_row(original)[0]), 0x3c00)
        parent = np.concatenate([np.full(1024, x, dtype=F) for x in (2, 3, 5)])
        self.assertTrue(np.shares_memory(m.bcx_view(parent, 'c'), parent))
        np.testing.assert_array_equal(m.binary(m.bcx_view(parent, 'b'), m.bcx_view(parent, 'x'), 'product'), np.full(1024, 10, dtype=F))
        self.assertEqual(m.bcx_view(parent, 'c')[0], 3)
        for wrong in ('z', 'B', None):
            with self.assertRaises(m.ContractError): m.bcx_view(parent, wrong)
        with self.assertRaises(m.ContractError): m.bcx_view(parent[:-1], 'x')

    def test_conv_actual_predecessors_all_boundary_modes_and_suffix(self):
        old = np.zeros((2, 1024), dtype=F); old[:, 0] = [7, 11]
        u = np.zeros((3, 1024), dtype=F); u[:, 0] = [2, 6, 12]
        coeff = np.zeros((1024, 3), dtype=F); coeff[0] = [1, 10, 100]
        self.assertEqual([m.conv_row(u, old, coeff, r)[0] for r in range(3)], [317, 631, 1262])
        np.testing.assert_array_equal(m.history_suffix(old, u[:1])[:, 0], [11, 2])
        np.testing.assert_array_equal(m.history_suffix(old, u)[:, 0], [6, 12])
        zero = np.zeros_like(coeff)
        self.assertTrue(np.all(m.conv_row(u, old, zero, 2) == 0))
        self.assertEqual(m.history_suffix(old, u)[1, 0], 12)  # Not filtered z.
        u[-1, 1] = -0.
        self.assertEqual(m.bits(m.history_suffix(old, u)[1, 1]), 0x80000000)
        reversed_coeff = coeff[:, ::-1].copy()
        self.assertNotEqual(m.conv_row(u, old, reversed_coeff, 0)[0], 317)
        with self.assertRaises(m.ContractError): m.conv_row(u[1:], old, coeff, 2)
        u[0, 0] = np.inf
        with self.assertRaises(m.ContractError): m.conv_row(u, old, coeff, 0)

    def test_conv_one_store_and_finite_intermediate_domain(self):
        old = np.zeros((2, 1024), dtype=F); u = np.zeros((1, 1024), dtype=F)
        old[0, 0] = 1; old[1, 0] = 1; u[0, 0] = 1
        coeff = np.zeros((1024, 3), dtype=F); coeff[0] = [1, 2**-11, 2**-12]
        self.assertEqual(word(m.conv_row(u, old, coeff, 0)[0]), 0x3c01)
        # A premature Half store after p0+p1 loses the ties-even increment.
        self.assertEqual(word(F(H(F(1 + 2**-11))) + F(2**-12)), 0x3c00)
        old[0, 0] = 2; coeff[0] = [F(2**127), 0, 0]
        with self.assertRaises(m.ContractError): m.conv_row(u, old, coeff, 0)

    def test_faithful_projection_singleton_midpoint_and_third_neighbor(self):
        x = [1, 2**-12] + [0]*62; w = [1, 1] + [0]*62
        for candidate, allowed in [(0x3c00, True), (0x3c01, True), (0x3bff, False), (0x3c02, False)]:
            center, valid, stats = projection(w, x, hword(candidate))
            self.assertEqual(word(center[0, 0]), 0x3c00)
            self.assertEqual(bool(valid[0, 0]), allowed)
            if candidate == 0x3c00: self.assertEqual(stats['equal_center_certificates'], 1)
            if candidate in (0x3c01, 0x3bff): self.assertEqual(stats['exact_bracket_checks'], 1)
        for candidate, allowed in [(0x3c00, True), (0x3c01, False), (0x3bff, False)]:
            _, valid, _ = projection([1]+[0]*63, [1]+[0]*63, hword(candidate))
            self.assertEqual(bool(valid[0, 0]), allowed)

    def test_projection_cancellation_double_rounding_zero_and_finite_edge(self):
        center, valid, stats = projection([2**100, 1, -2**100]+[0]*61, [1, 1, 1]+[0]*61, 1)
        self.assertEqual(center[0, 0], 1); self.assertTrue(valid[0, 0])
        self.assertGreater(stats['center']['blocked_outputs'], 0)
        center, valid, _ = projection([1, 2**-11, 2**-100]+[0]*61, [1, 1, 1]+[0]*61, hword(0x3c01))
        self.assertEqual(word(center[0, 0]), 0x3c01); self.assertTrue(valid[0, 0])
        for zero in (F(0), F(-0.)):
            center, valid, _ = projection([np.finfo(F).max]+[0]*63, [0]*64, zero)
            self.assertEqual(m.bits(center[0, 0]), 0); self.assertEqual(bool(valid[0, 0]), m.bits(zero) == 0)
        for candidate, allowed in [(0x8000, True), (0x8001, True), (0, False), (1, False)]:
            center, valid, _ = projection([-2**-25]+[0]*63, [1]+[0]*63, hword(candidate))
            self.assertEqual(word(center[0, 0]), 0x8000); self.assertEqual(bool(valid[0, 0]), allowed)
        for candidate, allowed in [(0, True), (1, True), (0x8000, False), (0x8001, False)]:
            center, valid, _ = projection([2**-25]+[0]*63, [1]+[0]*63, hword(candidate))
            self.assertEqual(word(center[0, 0]), 0); self.assertEqual(bool(valid[0, 0]), allowed)
        for candidate, allowed in [(0, True), (0x8000, False), (1, False), (0x8001, False)]:
            center, valid, _ = projection([1, -1]+[0]*62, [1, 1]+[0]*62, hword(candidate))
            self.assertEqual(word(center[0, 0]), 0); self.assertEqual(bool(valid[0, 0]), allowed)
        for weight in (65505, -65505):
            with self.assertRaises(m.ContractError): projection([weight]+[0]*63, [1]+[0]*63, F(np.sign(weight)*65504))
        for value in (-65504, 65504):
            _, valid, stats = projection([value]+[0]*63, [1]+[0]*63, value)
            self.assertTrue(valid[0, 0]); self.assertEqual(stats['exact_bracket_checks'], 1)

    def test_projection_real_column_boundary_and_wrong_shapes(self):
        x = np.zeros((4, 4608), dtype=F); x[:, -1] = [1, 2, 3, 4]
        weights = np.zeros((128, 4608), dtype=F); weights[:, -1] = np.arange(128, dtype=F) / F(128)
        actual = (x[:, -1, None] * weights[:, -1]).astype(H).astype(F)
        center, valid, _ = m.projection(x, weights, actual)
        np.testing.assert_array_equal(center, actual); self.assertTrue(valid.all())
        for badx, badw, bada in [(x[:, :-1].copy(), weights[:, :-1].copy(), actual),
                                 (x[:1], weights, actual), (x, weights.astype(np.float64), actual),
                                 (x, np.zeros((129, 4608), dtype=F), np.zeros((4, 129), dtype=F))]:
            with self.assertRaises(m.ContractError): m.projection(badx, badw, bada)

    def test_provider_literal_ulp_grid_domain_and_no_ftz(self):
        self.assertEqual(m.PROVIDER_ULP, (('exp', 4), ('pow', 16), ('sin', 4), ('cos', 4)))
        with localcontext() as ctx:
            ctx.prec = 120
            self.assertEqual(m.provider(m.point(Decimal(1)), 4)[0], F(1 - 4*2**-24))
            with self.assertRaises(m.ContractError):
                m.provider((Decimal(1)-Decimal(2)**-80, Decimal(1)+Decimal(2)**-80), 4)
        lo, hi, near = m.exp_contract(F(-200))
        self.assertEqual((m.bits(lo), m.bits(hi), m.bits(near)), (0, 4, 0))
        lo, _, _ = m.exp_contract(F(-90)); self.assertGreater(lo, 0)
        with self.assertRaises(m.ContractError): m.exp_contract(F(1))
        zero = m.signed_trig_members(F(0), F(0), 16)
        self.assertEqual(set(map(int, zero.view(np.uint32))), {0, 0x80000000})
        tiny = np.array([1], dtype=np.uint32).view(F)[0]
        members = m.signed_trig_members(-tiny, tiny, 16)
        self.assertEqual(set(map(int, members.view(np.uint32))), {0x80000001, 0x80000000, 0, 1})
        with self.assertRaises(m.ContractError): m.signed_trig_members(F(0), F(0), 1)

    def test_silu_exact_words_reachable_members_and_corruption(self):
        for input_word, expected in [(0, 0), (0x8000, 0x8000), (1, 0), (3, 2), (0x7bff, 0x7bff)]:
            candidates, center = m.silu_words(input_word)
            self.assertEqual(word(center), expected)
            self.assertTrue(np.any(candidates == center))
        for input_word in (0, 0x8000):
            x = np.array([hword(input_word)], dtype=F)
            _, valid = m.silu(x, x.copy()); self.assertTrue(valid[0])
            wrong = np.array([hword(input_word ^ 0x8000)], dtype=F)
            _, valid = m.silu(x, wrong); self.assertFalse(valid[0])
        # Fixed, target-independent synthetic domain; stops at the first legal
        # stored drift. No target word chooses provider width or acceptance.
        witness = None
        for input_word in range(0xbc00, 0xcc00):
            candidates, center = m.silu_words(input_word)
            if len(candidates) > 1:
                witness = (input_word, candidates, center); break
        self.assertIsNotNone(witness, 'synthetic Half exp-bound drift witness')
        input_word, candidates, _ = witness
        x = np.array([hword(input_word)], dtype=F)
        for candidate in candidates:
            _, valid = m.silu(x, np.array([candidate], dtype=F)); self.assertTrue(valid[0])
        bad = F(np.nextafter(H(min(candidates)), H(-np.inf), dtype=H))
        _, valid = m.silu(x, np.array([bad], dtype=F)); self.assertFalse(valid[0])
        for bad in (0x7c00, 0xfc00, 0x7e00):
            with self.assertRaises(m.ContractError): m.silu_words(bad)

    def test_rotary_fixed_domain_high_feature_and_joint_membership(self):
        co, si, c, s = m.rotary_pairs(1000000, 64, 63, 16)
        angle = F(F(63) * F(.001))  # Exact true pow(theta,-1/2) = 1/1000.
        exact = Fraction.from_float(float(angle))
        self.assertEqual(m.bits(c), m.bits(trig_small(exact, True)))
        self.assertEqual(m.bits(s), m.bits(trig_small(exact, False)))
        self.assertEqual(len(co), len(si)); self.assertLessEqual(len(co), 4096)
        m.rotary_pairs(1000000, 64, 63, 31)
        for args in [(10000,64,0,0),(1000000,256,0,0),(1000000,64,64,0),(1000000,64,0,32)]:
            with self.assertRaises(m.ContractError): m.rotary_pairs(*args)
        x = np.zeros((8,64),dtype=F); x[:,0] = 1; x[:,32] = 2
        center, good = m.rotary(x,x,0); np.testing.assert_array_equal(center,x); self.assertTrue(good.all())
        # Two coordinates each separately reachable, but no common pair emits
        # (1,1) from(1,2). Test the public joint checker, not only helper arrays.
        pairs = (np.array([1,0],dtype=F),np.array([0,1],dtype=F),F(1),F(0))
        wrong=x.copy();wrong[:,32]=1
        with patch.object(m,'rotary_pairs',return_value=pairs):
            _, good=m.rotary(x,wrong,31)
        self.assertFalse(good[:,0].any());self.assertFalse(good[:,32].any())

        # The intrinsic remaining stores use exact words: with co=1,si=+0,
        # (-0,+0) maps to (-0,+0). Numeric equality would hide the wrong sign.
        x = np.zeros((8,64), dtype=F); x[:,0] = F(-0.)
        pairs = (np.array([1],dtype=F),np.array([0],dtype=F),F(1),F(0))
        with patch.object(m,'rotary_pairs',return_value=pairs):
            _, good = m.rotary(x,x.copy(),0); self.assertTrue(good.all())
            wrong = x.copy(); wrong[:,0] = F(0.)
            _, good = m.rotary(x,wrong,0)
            self.assertFalse(good[:,0].any()); self.assertFalse(good[:,32].any())

        # Both trig zero signs are legal under a numeric provider interval.
        # From (-0,-0), shared si=+0/-0 yields (+0,-0)/(-0,+0), never (+0,+0).
        x = np.full((8,64), F(-0.), dtype=F)
        si = m.signed_trig_members(F(0), F(0), 16)
        pairs = (np.ones(len(si), dtype=F), si, F(1), F(0))
        with patch.object(m, 'rotary_pairs', return_value=pairs):
            for a, b in ((F(0.), F(-0.)), (F(-0.), F(0.))):
                observed = np.empty_like(x); observed[:,:32] = a; observed[:,32:] = b
                _, good = m.rotary(x,observed,0); self.assertTrue(good.all())
            _, good = m.rotary(x,np.zeros_like(x),0); self.assertFalse(good.any())

    def test_q16_probabilities_onekey_mask_and_serial_mass(self):
        scores=np.zeros((16,1,3),dtype=F);scores[:,:,1:]=[123,-7]
        actual=np.zeros_like(scores);actual[:,:,0]=1
        center,good=m.probabilities(scores,actual,0)
        np.testing.assert_array_equal(center,actual);self.assertTrue(good.all())
        actual[:,:,0]=F(np.nextafter(F(1),F(0)))
        _,good=m.probabilities(scores,actual,0);self.assertFalse(good[:,:,0].any())
        actual[:,:,0]=1;actual[:,:,2]=-0.
        with self.assertRaises(m.ContractError):m.probabilities(scores,actual,0)
        scores=np.zeros((16,1,4),dtype=F);actual=np.full_like(scores,F(.25))
        center,good=m.probabilities(scores,actual,3)
        np.testing.assert_array_equal(center,actual);self.assertTrue(good.all())
        actual[15,0,3]=F(.5)
        _,good=m.probabilities(scores,actual,3);self.assertFalse(good[15,0,3])
        for q,start in [(4,0),(0,1),(-1,0)]:
            with self.assertRaises(m.ContractError):m.probabilities(scores,actual,q,start)
        with self.assertRaises(m.ContractError):m.probabilities(scores[:4],actual[:4],3)

    def test_semantic_and_old_local_gate_both_required(self):
        actual=np.array([1],dtype=F);center=np.array([1],dtype=F)
        valid,report=m.conjoined(actual,center,np.array([False]),GATE)
        self.assertFalse(valid[0]);self.assertEqual(report['numeric_failures'],0)
        valid,report=m.conjoined(np.array([2],dtype=F),center,np.array([True]),GATE)
        self.assertFalse(valid[0]);self.assertEqual(report['semantic_failures'],0)
        self.assertEqual(report['numeric_failures'],1)
        for bad in ({'atol':1e-4,'rtol':.01},{'atol':0,'rtol':.001},{'atol':1e-4,'rtol':.001,'fatal_reference':'main'}):
            with self.assertRaises(m.ContractError):m.numeric_gate(actual,center,bad)
        valid,_=m.conjoined(np.array([-0.],dtype=F),np.array([0.],dtype=F),np.array([True]),GATE)
        self.assertTrue(valid[0])

if __name__ == '__main__': unittest.main()
