"""Portable prospective controls, no source export or GPU/model execution."""
import importlib.util
import math
from pathlib import Path
import unittest
spec=importlib.util.spec_from_file_location('lfm2_half_rms',Path(__file__).with_name('lfm2_half_rms.py'))
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)

class Controls(unittest.TestCase):
    def test_frozen_geometry_policy_and_last_head_row(self):
        self.assertEqual(s.POLICY,dict(atol_floor=1e-4,envelope_factor=2,rtol=1e-3,fatal_reference='main'))
        self.assertEqual(sum(r*w for _,r,w,_ in s.CASES),75008)
        for _,r,w,p in s.CASES:
            words,gamma,gain,stride=s.operands(r,w,p)
            self.assertEqual((len(gamma),len(gain),stride),(w,w,w+3))
            for row in range(r):self.assertEqual(words[row*stride+w:(row+1)*stride],[s.INPUT_PADDING]*3)
            self.assertEqual(words[-7:],[s.INPUT_PADDING]*7)
        words,_,_,stride=s.operands(512,64,'dyadic')
        self.assertTrue(any(words[511*stride:511*stride+64]))
        for r,w in [(0,64),(513,64),(33,256),(33,1024),(512,1024),(1,32),(1,640),(True,64)]:
            with self.assertRaises(ValueError):s.operands(r,w,'dyadic')

    def test_direct_gain_not_unit_offset(self):
        self.assertEqual(s.direct([0,0x3f80,0xbf80,0x3980,0x3300]),[0.,1.,-1.,2**-12,2**-25])
        words,gamma,gain,stride=s.operands(1,64,'zero_gain')
        for profile in ['main','wide']:
            self.assertEqual(s.rms(words,gain,1,64,stride,profile),[0]*64)
            self.assertEqual(s.rms(words,[1.]*64,1,64,stride,profile),[0x3c00]*64)
        for word in [0x7fc0,0x7f80,0xff80]:
            with self.assertRaises(ValueError):s.direct([word])

    def test_direct_gain_narrowing_exact_witness(self):
        words,_,gain,stride=s.operands(1,64,'coefficient')
        original=words[:],gain[:]
        wrong=[s.half_value(s.half_word(v)) for v in gain]
        for profile in ['main','wide']:
            self.assertEqual(s.rms(words,gain,1,64,stride,profile)[0],4)
            self.assertEqual(s.rms(words,wrong,1,64,stride,profile)[0],0)
        self.assertEqual((words,gain),original)

    def test_new_epsilon_signed_subnormal_is_not_old_epsilon(self):
        words,_,gain,stride=s.operands(2,64,'subnormal')
        for profile in ['main','wide']:
            out=s.rms(words,gain,2,64,stride,profile)
            self.assertEqual((out[0],out[64]),(0x013c,0x813c))
            self.assertTrue(all(v==0 for i,v in enumerate(out) if i not in (0,64)))
            old=s.helper.rms(words,gain,2,64,stride,profile)
            self.assertEqual((old[0],old[64]),(0x03e8,0x83e8))

    def test_compensated_reduction_and_large_input(self):
        self.assertEqual(s.neumaier([2**24,1,-2**24]),1)
        squares=[1024.**2]+[.125**2]*1023
        self.assertEqual(s.neumaier(squares),s.f32(math.fsum(squares)))
        naive=0.
        for v in squares:naive=s.f32(naive+v)
        self.assertNotEqual(naive,s.neumaier(squares))
        words,_,gain,stride=s.operands(1,1024,'large')
        for profile in ['main','wide']:
            self.assertTrue(all(math.isfinite(s.half_value(w)) for w in s.rms(words,gain,1,1024,stride,profile)))

    def test_nonfinite_overflow_extent_and_padding(self):
        self.assertEqual(len(s.errors()),5)
        for e in s.errors():
            for profile in ['main','wide']:
                with self.assertRaises((ValueError,OverflowError)):
                    s.rms(e['input_words'],s.direct(e['gamma_words']),e['rows'],e['width'],e['input_stride'],profile)
        for words,scales,stride in [([0]*63,[1.]*64,64),([0]*64,[1.]*63,64),([0]*64,[math.nan]*64,64),([0]*64,[1.]*64,63)]:
            with self.assertRaises(ValueError):s.rms(words,scales,1,64,stride,'main')
        words,_,gain,stride=s.operands(1,64,'dyadic')
        a=s.rms(words,gain,1,64,stride,'main');words[64:]=[0x7c00]*(len(words)-64)
        self.assertEqual(a,s.rms(words,gain,1,64,stride,'main'))

if __name__=='__main__':unittest.main()
