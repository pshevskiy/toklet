"""Portable view/arithmetic counterexamples; no source export or GPU."""
import importlib.util
from pathlib import Path
import unittest
spec=importlib.util.spec_from_file_location('strided_half_product',Path(__file__).with_name('strided_half_product.py'))
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)


class Controls(unittest.TestCase):
    def test_frozen_shapes_policy_owner_counts_and_extents(self):
        self.assertEqual(sum(r*h for _,r,h,*_ in s.CASES),67072)
        self.assertEqual(s.POLICY,dict(atol_floor=1e-4,envelope_factor=2,rtol=1e-3,fatal_reference='main'))
        arrays=raw=0
        for case in s.CASES:
            _,rows,width,kind,_,_=case;p,a,b,o=s.operands(case)
            self.assertEqual(a['owner']==b['owner'],kind!='cz')
            self.assertEqual(len(p),2 if kind=='cz' else 1)
            for view in (a,b):self.assertGreaterEqual(len(p[view['owner']]),s.extent(rows,width,view['offset'],view['stride']))
            arrays+=len(p)+2;raw+=sum(2*len(v)for v in p.values())+4*rows*width
            self.assertTrue(all(v==s.PAD for v in p['bcx'][-7:]))
        self.assertEqual((arrays,raw),(17,738248))
        for rows,width in [(0,256),(33,256),(1,5),(1,64),(True,256),(1,True)]:
            with self.assertRaises(ValueError):s.extent(rows,width,0,width)

    def test_exact_store_words_and_finite_product_identity(self):
        p,a,b,o=s.operands(s.CASES[2])
        for profile in ('main','wide'):
            out=s.multiply(p,a,b,o,2,256,profile)
            self.assertEqual(out[:10],s.WITNESS)
        for x,y,expected in zip(s.WITNESS_A,s.WITNESS_B,s.WITNESS,strict=True):
            product=s.half_value(x)*s.half_value(y)
            self.assertEqual(s.f32(product),product)
            self.assertEqual(s.half_word(product),expected)
        self.assertEqual(s.WITNESS[0],0x8000)
        self.assertEqual(s.WITNESS[1],0)
        self.assertEqual(s.WITNESS[2:7],[1,0,2,0x8001,0x8000])
        # Deliberate FTZ/toward-zero substitutes do not satisfy these words.
        self.assertNotEqual(s.WITNESS[2],0);self.assertNotEqual(s.WITNESS[4],1)

    def test_full_main_wide_and_partition_profiles(self):
        for case in s.CASES:
            _,rows,width,*_=case;p,a,b,o=s.operands(case)
            before={key:v[:]for key,v in p.items()}
            main=s.multiply(p,a,b,o,rows,width,'main');wide=s.multiply(p,a,b,o,rows,width,'wide')
            self.assertEqual(main,wide);self.assertEqual(p,before)
            schedules=s.partitions(p,a,b,o,rows,width,main)
            for schedule in schedules:self.assertEqual(sum(schedule),rows)
            self.assertEqual(len(main),rows*width)

    def test_parent_view_offsets_and_order_counterexamples(self):
        for case in [s.CASES[0],s.CASES[1]]:
            _,rows,width,*_=case;p,a,b,o=s.operands(case)
            correct=s.multiply(p,a,b,o,rows,width,'main')
            # B*x mistaken for B*C, or C*z mistaken for B*z, must differ.
            wrong_a=dict(a,offset=0 if a['offset']else width)
            wrong=s.multiply(p,wrong_a,b,o,rows,width,'main')
            self.assertNotEqual(correct,wrong)
            manual=[]
            for r in range(rows):
                for c in range(width):
                    av=p[a['owner']][a['offset']+r*a['stride']+c]
                    bv=p[b['owner']][b['offset']+r*b['stride']+c]
                    manual.append(s.half_word(s.half_value(av)*s.half_value(bv)))
            self.assertEqual(correct,manual)

    def test_inactive_poison_causality_and_output_prefix_tails(self):
        for case in [s.CASES[1],s.CASES[2]]:
            _,rows,width,kind,*_=case;p,a,b,o=s.operands(case)
            main=s.multiply(p,a,b,o,rows,width,'main')
            active={name:set()for name in p}
            for view in (a,b):
                for row in range(rows):active[view['owner']].update(range(view['offset']+row*view['stride'],view['offset']+row*view['stride']+width))
            poison={name:[word if i in active[name]else 0x7c00 for i,word in enumerate(v)]for name,v in p.items()}
            self.assertEqual(s.multiply(poison,a,b,o,rows,width,'main'),main)
            changed={name:v[:]for name,v in p.items()}
            start=a['offset']+(rows-1)*a['stride'];changed[a['owner']][start:start+width]=[s.half_word(8.)]*width
            future=s.multiply(changed,a,b,o,rows,width,'main')
            self.assertEqual(future[:(rows-1)*width],main[:(rows-1)*width]);self.assertNotEqual(future[(rows-1)*width:],main[(rows-1)*width:])
            output=s.output_words(main,rows,width,o);occupied=set()
            for row in range(rows):
                start=o['offset']+row*o['stride'];occupied.update(range(start,start+width))
                self.assertEqual(output[start:start+width],main[row*width:(row+1)*width])
            self.assertTrue(all(word==s.CANARY for i,word in enumerate(output)if i not in occupied))
            self.assertGreater(o['offset'],0)

    def test_nonfinite_zero_counterpart_and_half_overflow(self):
        c=s.controls();self.assertEqual(len(c['invalid']),8)
        for item in c['invalid']:
            p,a,b,o=s.apply_control(item)
            for profile in ('main','wide'):
                with self.assertRaises((ValueError,OverflowError),msg=item['name']):s.multiply(p,a,b,o,2,256,profile)
        self.assertEqual(s.half_word(65504.*1.),0x7bff)
        with self.assertRaises(OverflowError):s.half_word(65504.*2.)

    def test_checked_extent_truncation_integer_and_offset_domains(self):
        p,a,b,o=s.operands(s.CASES[1]);rows,width=3,256
        for view in (a,b):
            needed=s.extent(rows,width,view['offset'],view['stride'])
            for length in [needed-1,needed-s.GUARD-1]:
                short={name:v[:]for name,v in p.items()};short[view['owner']]=short[view['owner']][:length]
                with self.assertRaises(ValueError):s.multiply(short,a,b,o,rows,width,'main')
        needed=s.extent(rows,width,o['offset'],o['stride'],True)
        with self.assertRaises(ValueError):s.multiply(p,a,b,o,rows,width,'main',needed-1)
        for offset,stride in [(-1,256),(True,256),(0,255),(s.U32,256),(0,s.U32),(0,True)]:
            with self.assertRaises(ValueError):s.extent(rows,width,offset,stride)
        with self.assertRaises(ValueError):s.multiply(p,a,b,o,rows,width,'other')
        with self.assertRaises(ValueError):s.output_words([0],rows,width,o)

    def test_read_read_alias_and_writable_alias_rejections(self):
        s.aliases('bcx','bcx','out','status')
        s.aliases('bcx','z','out','status')
        for owners in [('bcx','bcx','bcx','status'),('bcx','z','z','status'),('bcx','z','status','status'),('bcx','z','out','bcx'),('bcx','z','out','z')]:
            with self.assertRaises(ValueError):s.aliases(*owners)
        p,a,b,o=s.operands(s.CASES[0]);self.assertIs(p[a['owner']],p[b['owner']])
        with self.assertRaises(ValueError):s.multiply(p,a,b,dict(o,owner='bcx'),1,256,'main')


if __name__=='__main__':unittest.main()
