"""Portable analytic controls; no GPU, model, graph or source fixture export."""
import importlib.util
from pathlib import Path
import unittest
spec=importlib.util.spec_from_file_location('causal_conv3_half',Path(__file__).with_name('causal_conv3_half.py'))
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)


class Controls(unittest.TestCase):
    def test_fixed_shapes_policy_and_original_bf16(self):
        self.assertEqual(sum(r*h for _,r,h,_ in s.CASES),46571)
        self.assertEqual(len(s.CASES),12)
        self.assertEqual(s.POLICY,dict(atol_floor=1e-4,envelope_factor=2,rtol=1e-3,fatal_reference='main'))
        self.assertEqual(s.coefficients([0,0x8000,0x3f80,0xbf80,0x3300]),[0.,-0.,1.,-1.,2**-25])
        for rows,width in [(0,5),(33,5),(1,6),(1,64),(True,5),(1,True)]:
            with self.assertRaises(ValueError):s.shape(rows,width)
        for _,r,h,p in s.CASES:
            u,old,gamma,scales,stride=s.operands(r,h,p)
            self.assertEqual((len(u),len(old),len(gamma),len(scales)),(r*(h+3)+7,2*h+7,3*h,3*h))
            for row in range(r):self.assertEqual(u[row*stride+h:(row+1)*stride],[s.INPUT_PADDING]*3)
            self.assertEqual(u[-7:],[s.INPUT_PADDING]*7)
            self.assertEqual(old[-7:],[s.CANARY]*7)

    def test_asymmetric_hand_oracle_nonzero_history_and_tap_order(self):
        u,old,_,scales,stride=s.operands(5,5,'hand')
        for profile in ('main','wide'):
            out,candidate=s.conv(u,old,scales,5,5,stride,profile)
            self.assertEqual(out,list(map(s.half_word,s.HAND_OUTPUT)))
            self.assertEqual(candidate,list(map(s.half_word,[20,-4,4,-4,8,30,-6,5,-5,10])))
            swapped=[v for c in range(5) for v in scales[3*c:3*c+3][::-1]]
            wrong,_=s.conv(u,old,swapped,5,5,stride,profile)
            self.assertNotEqual(out,wrong)
            zero,_=s.conv(u,[0]*10,scales,5,5,stride,profile)
            self.assertNotEqual(out[:10],zero[:10])

    def test_exact_half_store_coefficient_subnormal_and_signed_zero(self):
        u,old,_,scales,stride=s.operands(1,5,'stores')
        for profile in ('main','wide'):
            out,candidate=s.conv(u,old,scales,1,5,stride,profile)
            self.assertEqual(out,s.STORE_WORDS)
            self.assertEqual(candidate,old[5:10]+u[:5])
            self.assertEqual((candidate[4],candidate[9]),(0x8000,0x8000))
            narrowed=[s.half_value(s.half_word(v)) for v in scales]
            wrong,_=s.conv(u,old,narrowed,1,5,stride,profile)
            self.assertEqual(wrong[1],0)
        first=s.half_value(s.half_word(1.+2**-11))
        self.assertEqual(s.half_word(first+2**-11),0x3c00)
        self.assertEqual(s.half_word(1.+2**-11+2**-11),0x3c01)
        self.assertEqual(s.float_word(s.half_value(0x8000)),0x80000000)

    def test_raw_suffix_all_boundaries_and_partition_words(self):
        for _,rows,width,pattern in s.CASES:
            u,old,_,scales,stride=s.operands(rows,width,pattern)
            out,candidate=s.conv(u,old,scales,rows,width,stride,'main')
            prefix=old[:2*width]
            for r in range(rows):prefix+=u[r*stride:r*stride+width]
            self.assertEqual(candidate,prefix[-2*width:])
            records=s.partition_checks(u,old,scales,rows,width,stride,out)
            self.assertTrue(records)
            for item in records:self.assertEqual(sum(item['chunks']),rows)
        # T1 has H outputs but must return BOTH history rows, including signbits.
        u,old,_,scales,stride=s.operands(1,5,'stores')
        _,candidate=s.conv(u,old,scales,1,5,stride,'main')
        self.assertEqual(len(candidate),10)
        self.assertEqual(candidate[:5],old[5:10]);self.assertEqual(candidate[5:],u[:5])

    def test_zero_coefficients_do_not_erase_raw_candidate(self):
        u,old,_,scales,stride=s.operands(3,5,'zero')
        for profile in ('main','wide'):
            out,candidate=s.conv(u,old,scales,3,5,stride,profile)
            self.assertTrue(all(w&0x7fff==0 for w in out))
            self.assertTrue(any(w&0x7fff for w in candidate))
            self.assertNotEqual(candidate,out[-10:])

    def test_causality_padding_and_immutable_inputs(self):
        u,old,_,scales,stride=s.operands(3,5,'seed')
        originals=u[:],old[:],scales[:]
        for profile in ('main','wide'):
            baseline,_=s.conv(u,old,scales,3,5,stride,profile)
            changed=u[:];changed[2*stride:2*stride+5]=[s.half_word(8.)]*5
            future,_=s.conv(changed,old,scales,3,5,stride,profile)
            self.assertEqual(baseline[:10],future[:10])
            self.assertNotEqual(baseline[10:],future[10:])
            padded=u[:]
            for r in range(3):padded[r*stride+5:(r+1)*stride]=[0x7c00,0xfc00,0x7e00]
            padded[-7:]=[0x7c00]*7
            poisoned_old=old[:10]+[0x7e00]*7
            self.assertEqual(s.conv(padded,poisoned_old,scales,3,5,stride,profile)[0],baseline)
        self.assertEqual((u,old,scales),originals)

    def test_nonfinite_intermediate_and_half_overflow_fail_closed(self):
        self.assertEqual(len(s.errors()),9)
        for item in s.errors():
            for profile in ('main','wide'):
                with self.assertRaises((ValueError,OverflowError),msg=item['name']):
                    s.conv(item['u_words'],item['old_words'],s.coefficients(item['gamma_words']),1,5,item['input_stride'],profile)
        item=next(v for v in s.errors() if v['name']=='partial_fp32_overflow_before_cancellation')
        k=s.coefficients(item['gamma_words'])[:3]
        self.assertTrue(all(v==s.f32(v) for v in k))
        # Correct native domain rejects first FP32 addition even though the
        # sequential FP64 reduction ends finite after the third term.
        self.assertEqual((k[0]+k[1])+k[2],2**127)
        for word in (0x7fc0,0x7f80,0xff80):
            with self.assertRaises(ValueError):s.coefficients([word])

    def test_extent_profile_and_last_active_value_validation(self):
        u,old,_,scales,stride=s.operands(3,5,'seed')
        bad=[(u[:20],old,scales,stride,'main'),(u,old[:9],scales,stride,'main'),
             (u,old,scales[:14],stride,'main'),(u,old,scales,4,'main'),(u,old,scales,stride,'unknown')]
        for args in bad:
            with self.assertRaises(ValueError):s.conv(args[0],args[1],args[2],3,5,args[3],args[4])
        invalid=u[:];invalid[2*stride+4]=0x7e00
        for profile in ('main','wide'):
            with self.assertRaises(ValueError):s.conv(invalid,old,scales,3,5,stride,profile)
        self.assertEqual(old[-7:],[s.CANARY]*7)


if __name__=='__main__':unittest.main()
