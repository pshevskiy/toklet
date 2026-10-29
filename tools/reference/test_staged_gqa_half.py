"""Portable analytic GQA/stride/copy controls; no target/model or fixture export."""
import importlib.util
import math
from pathlib import Path
import struct
import unittest

spec=importlib.util.spec_from_file_location('staged_gqa_half',Path(__file__).with_name('staged_gqa_half.py'))
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)


class Controls(unittest.TestCase):
    def test_profile_all_queries_counts_and_shape_rejections(self):
        self.assertEqual(s.POLICY,dict(fatal_reference='main',atol_floor=1e-4,envelope_factor=2,rtol=1e-3))
        self.assertEqual(len(s.CASES),11)
        context=sum(c['tokens']*c['query_heads']*64 for c in s.CASES)
        scores=sum(c['tokens']*c['query_heads']*(s.view(c)['end']-s.view(c)['start']) for c in s.CASES)
        physical=sum(32*c['query_heads']*64+7+4*(c['kv_heads']*s.view(c)['retention']*64+7)+2*(c['kv_heads']*32*64+7) for c in s.CASES)
        self.assertEqual((context,scores,physical),(84736,334348,8358427))
        self.assertEqual(2*physical+4*context+16*scores,22405366)
        deep=s.CASES[9];self.assertEqual((deep['committed'],deep['tokens'],deep['query_heads']),(512,32,16))
        for change in [dict(query_heads=3),dict(kv_heads=0),dict(kv_heads=3),dict(dim=32),dict(capacity=2**64),dict(tokens=0),dict(tokens=33),dict(committed=-1),dict(committed=64),dict(window=0),dict(query_heads=True),dict(name='unknown')]:
            with self.assertRaises(ValueError):s.shape(dict(s.CASES[0],**change))
        self.assertEqual(s.view(s.CASES[3])['old_spans'],[dict(absolute_start=2,slot=2,rows=2),dict(absolute_start=4,slot=0,rows=1)])
        self.assertEqual(s.view(s.CASES[10])['old_spans'],[dict(absolute_start=2,slot=2,rows=510),dict(absolute_start=512,slot=0,rows=1)])

    def test_all_head_copy_intervals_absolute_rebuild_and_chain(self):
        chain=None
        for c in s.CASES:
            old,staged,q=s.operands(c);before={k:v[:] for k,v in old.items()};new_before={k:v[:] for k,v in staged.items()}
            after=s.commit_reference(c,old,staged)
            for kind in ('k','v'):
                self.assertEqual(after[kind],s.physical(c,c['committed']+c['tokens'],kind))
                self.assertEqual(after[kind][-7:],[s.PAD]*7)
            if c['name']=='real_512':chain=after
            if c['name']=='real_513':self.assertEqual(old,chain)
            self.assertEqual((old,staged),(before,new_before))
            intervals=s.intervals(c);ret=s.view(c)['retention'];n=c['tokens'];kv=c['kv_heads']
            self.assertEqual(len(intervals['active_stage_fill_and_validation']),2*kv)
            for kind in ('k','v'):
                active=set()
                for head in range(kv):
                    descriptor=next(x for x in intervals['active_stage_fill_and_validation'] if x['kind']==kind and x['head']==head)
                    self.assertEqual(descriptor['byte_offset'],head*32*64*2)
                    self.assertEqual(descriptor['bytes'],n*64*2)
                    active.update(range(head*32*64,(head*32+n)*64))
                expected={i for h in range(kv) for t in range(n) for i in range((h*32+t)*64,(h*32+t+1)*64)}
                self.assertEqual(active,expected)
                if n<32:self.assertNotEqual(active,set(range(kv*n*64)))
                # Independent byte-descriptor executor must reproduce every word.
                copied=old[kind][:]
                for item in intervals['phase_b_copy']:
                    if item['kind']!=kind:continue
                    source=item['source_byte_offset']//2;target=item['destination_byte_offset']//2;count=item['bytes']//2
                    self.assertLessEqual(source+count,(item['head']+1)*32*64)
                    self.assertLessEqual(target+count,(item['head']+1)*ret*64)
                    copied[target:target+count]=staged[kind][source:source+count]
                self.assertEqual(copied,after[kind])
            self.assertTrue(all(w==s.PAD for w in q[c['tokens']*c['query_heads']*64:]))

    def test_one_key_group_mapping_and_signed_subnormal_exact_words(self):
        c=s.CASES[0];old,staged,q=s.operands(c)
        for profile in ('main','wide'):
            out=s.attention(c,old,staged,q,profile)
            s.numerical_checks(c,out)
            self.assertEqual(out['probabilities'],[1.]*4)
            for head in range(4):self.assertEqual(out['context'][head*64:(head+1)*64],s.row(c,0,head//2,'v'))
            self.assertEqual([out['context'][h*64] for h in range(4)],[1,1,0x8001,0x8001])
            self.assertNotEqual(out['context'][:64],out['context'][128:192])
            broadcast={k:v[:] for k,v in staged.items()}
            broadcast['v'][32*64:33*64]=broadcast['v'][:64]
            wrong=s.attention(c,old,broadcast,q,profile)
            self.assertNotEqual(out['context'],wrong['context'])
            with self.assertRaises(ValueError):s.numerical_checks(c,wrong)

    def test_two_key_uniform_analytic_average(self):
        c=dict(s.CASES[1],tokens=1);old,staged,q=s.operands(c)
        q[:4*64]=[0]*(4*64)
        for head in range(2):
            old['k'][head*64*64:head*64*64+64]=[0]*64
            staged['k'][head*32*64:head*32*64+64]=[0]*64
            old['v'][head*64*64:head*64*64+64]=[s.half_word(head+.5)]*64
            staged['v'][head*32*64:head*32*64+64]=[s.half_word(head+1.5)]*64
        for profile in ('main','wide'):
            out=s.attention(c,old,staged,q,profile)
            self.assertEqual(out['scores'],[0.]*8)
            self.assertEqual(out['probabilities'],[.5]*8)
            for head in range(4):self.assertEqual(out['context'][head*64:(head+1)*64],[s.half_word(head//2+1.)]*64)

    def test_raw_future_scores_exact_probability_mask_and_causality(self):
        c=s.CASES[3];old,staged,q=s.operands(c);keys=s.view(c)['end']-s.view(c)['start']
        original=({k:v[:] for k,v in old.items()},{k:v[:] for k,v in staged.items()},q[:])
        for profile in ('main','wide'):
            out=s.attention(c,old,staged,q,profile);s.numerical_checks(c,out)
            self.assertTrue(any(x!=0 for x in out['scores'][4:keys]))
            self.assertTrue(all(struct.pack('<f',x)==b'\0'*4 for x in out['probabilities'][4:keys]))
            changed={k:v[:] for k,v in staged.items()}
            for head in range(2):
                start=(head*32+6)*64
                changed['v'][start:start+64]=[s.half_word(8.)]*64
                for feature in range(64):
                    qvalue=s.half_value(q[(head*2)*64+feature])
                    delta=4. if qvalue>0 else (-4. if qvalue<0 else 0.)
                    changed['k'][start+feature]=s.half_word(s.half_value(staged['k'][start+feature])+delta)
            other=s.attention(c,old,changed,q,profile)
            self.assertEqual(out['context'][:6*4*64],other['context'][:6*4*64])
            self.assertNotEqual(out['context'][6*4*64:],other['context'][6*4*64:])
            self.assertNotEqual(out['scores'][keys-1],other['scores'][keys-1])
            s.numerical_checks(c,other)
        self.assertEqual((old,staged,q),original)

    def test_late_head_active_poison_and_inactive_strided_padding(self):
        c=s.CASES[5];old,staged,q=s.operands(c)
        baseline=s.decoded_view(c,old,staged,q)
        for kind in ('k','v'):
            for word in (0x7e00,0x7c00):
                bad={k:v[:] for k,v in staged.items()};bad[kind][(7*32+2)*64+63]=word
                with self.assertRaises(ValueError):s.decoded_view(c,old,bad,q)
            omitted={k:v[:] for k,v in staged.items()};omitted[kind][7*32*64:(7*32+3)*64]=[s.PAD]*(3*64)
            with self.assertRaises(ValueError):s.decoded_view(c,old,omitted,q)
            ignored={k:v[:] for k,v in staged.items()};ignored[kind][(7*32+3)*64:8*32*64]=[0x7c00]*(29*64)
            self.assertEqual(s.decoded_view(c,old,ignored,q),baseline)
            bad_old={k:v[:] for k,v in old.items()};bad_old[kind][(7*544+31)*64]=0x7c00
            with self.assertRaises(ValueError):s.decoded_view(c,bad_old,staged,q)
        bad_q=q[:];bad_q[(2*16+15)*64+63]=0x7e00
        with self.assertRaises(ValueError):s.decoded_view(c,old,staged,bad_q)
        # The physically retained older row is outside a local minimal W-1 view.
        c=s.CASES[3];old,staged,q=s.operands(c);baseline=s.decoded_view(c,old,staged,q)
        for kind in ('k','v'):
            ignored={k:v[:] for k,v in old.items()};ignored[kind][(1*4+1)*64:(1*4+2)*64]=[0x7e00]*64
            self.assertEqual(s.decoded_view(c,ignored,staged,q),baseline)

    def test_serial_main_and_fsum_wide_store_contract(self):
        c=s.CASES[0];old,staged,q=s.operands(c)
        q[:4*64]=[0]*(4*64);q[:3]=list(map(s.half_word,[32768.,1.,-32768.]))
        for head in range(2):staged['k'][head*32*64:head*32*64+64]=list(map(s.half_word,[32768.,1.,32768.]))+[0]*61
        main=s.attention(c,old,staged,q,'main');wide=s.attention(c,old,staged,q,'wide')
        self.assertEqual(main['scores'][0],0.)
        self.assertEqual(wide['scores'][0],.125)
        self.assertEqual(main['probabilities'],wide['probabilities'])
        self.assertEqual(main['context'],wide['context'])
        self.assertEqual(s.f32(math.fsum([2**30,1.,-2**30])),1.)

    def test_extent_and_unknown_profile_rejections(self):
        c=s.CASES[0];old,staged,q=s.operands(c)
        with self.assertRaises(ValueError):s.attention(c,old,staged,q,'unknown')
        with self.assertRaises(ValueError):s.decoded_view(c,old,staged,q[:-1])
        for kind in ('k','v'):
            bad={k:v[:] for k,v in old.items()};bad[kind].pop()
            with self.assertRaises(ValueError):s.decoded_view(c,bad,staged,q)
            bad={k:v[:] for k,v in staged.items()};bad[kind].pop()
            with self.assertRaises(ValueError):s.decoded_view(c,old,bad,q)
        for committed in (-1,65,True):
            with self.assertRaises(ValueError):s.physical(c,committed,'v')
        with self.assertRaises(ValueError):s.row(c,0,0,'unknown')


if __name__=='__main__':unittest.main()
