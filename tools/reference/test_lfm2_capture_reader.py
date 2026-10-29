"""Private authored negative controls; no captured target/model files required.

These tests have not been imported or executed. Full event/runner-schema controls
will be added before the complete reader proposal is frozen.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np
import lfm2_capture_contract as c
import lfm2_capture_bytes as b


def encoded(value):
    return (json.dumps(value, separators=(',', ':'), allow_nan=False) + '\n').encode()


def array_record(raw, seq=1, offset=0, query=None):
    return dict(kind='array', seq=seq, offset=offset, bytes=len(raw), sha256=b.digest(raw),
                dtype='f16le', id=[1, 1, 0, 0], shape=[len(raw) // 2, 1, 1], rank=1,
                query=[0, 1] if query is None else query, keys=[0, 0])


class Saved:
    def __init__(self, directory, raw=None, records=None):
        self.raw = struct.pack('<3H', 0, 0x8000, 0x7e55) if raw is None else raw
        self.records = ([dict(seq=0, kind='start', body={}), array_record(self.raw),
                         dict(seq=2, kind='end', body={})] if records is None else records)
        self.journal = b''.join(encoded(r) for r in self.records)
        self.raw_path = Path(directory) / 'capture.bin'
        self.journal_path = Path(directory) / 'journal.jsonl'
        self.raw_path.write_bytes(self.raw); self.journal_path.write_bytes(self.journal)

    def summary(self):
        return dict(raw_bytes=len(self.raw), raw_sha256=b.digest(self.raw),
                    journal_bytes=len(self.journal), journal_sha256=b.digest(self.journal),
                    records=len(self.records), arrays=sum(r['kind'] == 'array' for r in self.records),
                    boundaries=sum(r['kind'] in ('boundary', 'session_owners') for r in self.records),
                    other=sum(r['kind'] not in ('start', 'array', 'boundary', 'session_owners') for r in self.records))

    def open(self):
        return b.Capture(self.raw_path, self.journal_path, self.summary())


class FakeCapture:
    """Test-only recorded operands, never an oracle fallback used by reader."""
    def __init__(self):
        self.items = {}

    def put(self, kind, index, words, query=(0, 0)):
        words = np.ascontiguousarray(words, dtype='<u2')
        shape = list(words.shape) + [1] * (3 - words.ndim)
        self.items[(kind, index, query)] = dict(dtype='f16le', shape=shape, rank=words.ndim,
                                               query=list(query), keys=[0, 0], words=words)

    def get(self, session, attempt, kind, index, query=(0, 0)):
        return self.items[(kind, index, tuple(query))]

    @staticmethod
    def words(d):
        return d['words']


def physical(fault=None, start=3, n=2):
    capture = FakeCapture()
    state = b.PhysicalState(capture, 1, True, True)
    state.offset = start
    for id_ in range(24):
        state.kv[id_] = np.full(state.sizes[id_] // 2, c.HALF_PAD, dtype='<u2')
    for owner, layer in enumerate(c.CONVOLUTION):
        old = np.full(2055, c.HALF_PAD, dtype='<u2'); old[:2048] = 0
        stale = old.copy(); stale[:2048] = 13
        u = (np.arange(n * 1024, dtype=np.uint16).reshape(n, 1024) % 97 + layer + 1)
        candidate = old.copy(); candidate[:2048] = u[-2:].reshape(-1)
        if fault == 'OmitHistory9' and owner == 9:
            candidate[:2048] = 0x7e7e
        capture.put(1, layer, u, (start, n))
        capture.put(4, 2 * owner, old); capture.put(4, 2 * owner + 1, stale)
        capture.put(5, 2 * owner, candidate if fault is None else old)
        capture.put(5, 2 * owner + 1, old if fault is None else candidate)
    for layer in c.ATTENTION:
        for kind in (2, 3):
            values = (np.arange(n * 512, dtype=np.uint16).reshape(n, 512) % 997 + 0x2000 + layer + kind)
            capture.put(kind, layer, values, (start, n))
    return state, capture


class ReaderControls(unittest.TestCase):
    def test_concrete_stage_inventory_and_all_routes(self):
        self.assertEqual(len(c.STAGES), 293)
        self.assertEqual(len(c.SLOTS), 293)
        self.assertEqual(sum(s.coupled for s in c.STAGES), 13)
        self.assertEqual(sum(not s.coupled for s in c.STAGES), 280)
        self.assertEqual([s.name for s in c.STAGES if s.operation == 'conv'],
                         [f'layers.{i}.convolution' for i in c.CONVOLUTION])
        for layer in range(16):
            self.assertEqual(sum(s.layer == layer for s in c.STAGES), 20 if layer in c.ATTENTION else 17)
        self.assertEqual(c.STAGES[c.SLOTS['layers.15.operator_residual']].inputs,
                         ('layers.14.output', 'layers.15.operator_projection'))
        self.assertEqual(c.STAGES[-2].inputs, ('layers.15.output',))

    def test_json_duplicate_nonfinite_and_boolean_identity_rejected(self):
        for raw in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}'):
            with self.assertRaises(ValueError): c.parse_json(raw)
        for value in (True, 1.0, -1, 513):
            with self.assertRaises(ValueError): c.integer(value, 0, 512, 'test')

    def test_file_regular_cap_symlink_fifo_and_endpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'file'; p.write_bytes(b'abc')
            with self.assertRaises(ValueError): c.File(p, 2)
            link = Path(directory) / 'link'; link.symlink_to(p)
            with self.assertRaises(OSError): c.File(link, 3)
            fifo = Path(directory) / 'fifo'; os.mkfifo(fifo)
            with self.assertRaises(ValueError): c.File(fifo, 3)
            f = c.File(p, 3)
            try:
                p.write_bytes(b'xyz')
                with self.assertRaises(ValueError): f.endpoint()
            finally: f.close()

    def test_raw_nan_words_retained_numerical_view_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            saved = Saved(directory); capture = saved.open()
            try:
                self.assertEqual(capture.words(capture.by_seq[1]).tolist(), [0, 0x8000, 0x7e55])
                with self.assertRaises(ValueError): capture.values(capture.by_seq[1])
                with self.assertRaises(ValueError): capture.get(1, 1, 0, 1, (0, 1))
            finally: capture.close()

    def test_descriptor_corruption_and_trailing_bytes_fail(self):
        raw = struct.pack('<3H', 1, 2, 3)
        for change in ({'offset': 1}, {'bytes': 4}, {'rank': 2}, {'dtype': 'f64le'},
                       {'shape': [2, 1, 1]}, {'sha256': '0' * 64}, {'query': [512, 1]},
                       {'extra': 0}):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                d = array_record(raw); d.update(change)
                # Rank2 alone is valid for [3,1,1]; make its extent inconsistent.
                if change == {'rank': 2}: d['shape'] = [3, 2, 1]
                saved = Saved(directory, raw, [dict(seq=0, kind='start', body={}), d])
                with self.assertRaises(ValueError): saved.open()
        with tempfile.TemporaryDirectory() as directory:
            saved = Saved(directory, raw + b'\x00\x00',
                          [dict(seq=0, kind='start', body={}), array_record(raw)])
            with self.assertRaises(ValueError): saved.open()

    def test_sequence_duplicate_operand_and_complete_summary(self):
        raw = struct.pack('<H', 1)
        for mode in ('sequence', 'duplicate', 'summary'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                records = [dict(seq=0, kind='start', body={}), array_record(raw)]
                if mode == 'sequence': records[1]['seq'] = 2
                if mode == 'duplicate': records.append(array_record(raw, 2, len(raw)))
                saved = Saved(directory, raw * (2 if mode == 'duplicate' else 1), records)
                if mode == 'summary':
                    summary = saved.summary(); summary['arrays'] += 1
                    with self.assertRaises(ValueError): b.Capture(saved.raw_path, saved.journal_path, summary)
                else:
                    with self.assertRaises(ValueError): saved.open()

    def test_successful_selected_zero_and_raw_hash_distinction(self):
        plus = np.zeros(65536, dtype='<u4'); minus = plus.copy(); minus[19] = 0x80000000
        self.assertNotEqual(b.hash_words(plus), b.hash_words(minus))
        self.assertEqual(b.numeric_head_hash(plus), b.numeric_head_hash(minus))
        minus[19] = 0x7fc00000
        with self.assertRaises(ValueError): b.numeric_head_hash(minus)

    def test_physical_owner_sizes_and_guard_inventory(self):
        diagnostic = b.expected_owner_bytes(True, True)
        ordinary = b.expected_owner_bytes(False, False)
        self.assertEqual((len(diagnostic), len(ordinary)), (357, 64))
        self.assertEqual(sum(diagnostic[i] for i in range(24)), 6685008)
        self.assertEqual(sum(diagnostic[i] for i in range(24, 44)), 82200)
        self.assertEqual(sum(diagnostic[i] for i in range(64, 357)), 54530796)
        self.assertEqual(sum(len(b.expected_guard(i)) for i in diagnostic), 5824)
        self.assertEqual(sum(len(b.expected_guard(i)) for i in ordinary), 980)
        with self.assertRaises(ValueError): b.expected_owner_bytes(True, False)

    def test_duplicate_physical_owner_and_wrong_size_rejected(self):
        state, _ = physical()
        rows = [[i, (i + 1) * 16 * 1024**2, size, 1] for i, size in state.sizes.items()]
        bad = copy.deepcopy(rows); bad[1][1] = bad[0][1]
        with self.assertRaises(ValueError): state.owner_table(bad)
        state, _ = physical(); bad = copy.deepcopy(rows); bad[17][2] -= 2
        with self.assertRaises(ValueError): state.owner_table(bad)

    def test_all_kv_heads_keep_inactive_gap_and_publish_actual_suffix(self):
        state, capture = physical()
        state.reconstruct(1, 3, [1, 2])
        self.assertEqual(state.offset, 5)
        for owner, layer in enumerate(c.ATTENTION):
            for role, kind in ((0, 2), (1, 3)):
                actual = capture.words(capture.get(1, 1, kind, layer, (3, 2))).reshape(2, 8, 64)
                old = state.kv[4 * owner + role][:-7].reshape(8, 512, 64)
                stage = state.kv[4 * owner + role + 2][:-7].reshape(8, 32, 64)
                np.testing.assert_array_equal(old[:, 3:5], actual.transpose(1, 0, 2))
                np.testing.assert_array_equal(stage[:, :2], actual.transpose(1, 0, 2))
                self.assertTrue(np.all(old[:, :3] == c.HALF_PAD))
                self.assertTrue(np.all(old[:, 5:] == c.HALF_PAD))
                self.assertTrue(np.all(stage[:, 2:] == c.HALF_PAD))

    def test_corrupt_actual_last_history_owner_rejected(self):
        state, capture = physical()
        capture.items[(5, 18, (0, 0))]['words'][2047] ^= 1
        with self.assertRaises(ValueError): state.reconstruct(1, 3, [1, 2])

    def test_first_copy_only128_head0K_words_poison_prefix_unchanged(self):
        state, _ = physical('FirstCopy')
        state.reconstruct(1, 3, [1, 2], 'FirstCopy')
        self.assertEqual(state.offset, 3)
        self.assertEqual(state.mutated_copy_words, 128)
        self.assertEqual(np.count_nonzero(state.kv[0] != c.HALF_PAD), 128)
        for id_ in range(1, 24):
            if id_ % 4 < 2:
                self.assertTrue(np.all(state.kv[id_] == c.HALF_PAD))
        self.assertTrue(np.all(state.tokens == 0))

    def test_omitted_history_stamp_retained_and_old_preserved(self):
        state, _ = physical('OmitHistory9')
        state.reconstruct(1, 3, [1, 2], 'OmitHistory9')
        self.assertTrue(np.all(state.history[18][:2048] == 0))
        self.assertTrue(np.all(state.history[19][:2048] == 0x7e7e))
        for id_ in range(24):
            if id_ % 4 < 2:
                self.assertTrue(np.all(state.kv[id_] == c.HALF_PAD))

    def test_reset_clears_history_but_preserves_dirty_KV_tokens(self):
        state, _ = physical()
        state.reconstruct(1, 3, [1, 2])
        kv = [b.hash_words(state.kv[i]) for i in range(24)]
        tokens = state.tokens.copy()
        state.reset()
        self.assertEqual(state.offset, 0)
        self.assertEqual(kv, [b.hash_words(state.kv[i]) for i in range(24)])
        np.testing.assert_array_equal(tokens, state.tokens)
        for words in state.history.values():
            self.assertTrue(np.all(words[:2048] == 0))
            self.assertTrue(np.all(words[-7:] == c.HALF_PAD))


class EventAndDispatchControls(unittest.TestCase):
    def test_15_literal_workload_and_typed_terminal(self):
        import lfm2_capture_workload as w
        self.assertEqual((len(w.PRIMARY),sum(map(len,w.PRIMARY)),
                          sum(len(x.tokens) for r in w.PRIMARY for x in r)),(8,78,231))
        self.assertEqual([x.start for x in w.TEACHER],[0,8,9,10])
        chunk=w.PRIMARY[0][0]
        event=dict(session=1,attempt=1,run=0,start=0,tokens=[1],selected=[0],prior_boundary=17)
        w.validate_begin(event,chunk,1,17)
        for key,value in (('tokens',[2]),('selected',[]),('start',1),('attempt',True),('prior_boundary',16)):
            bad=copy.deepcopy(event);bad[key]=value
            with self.assertRaises(ValueError):w.validate_begin(bad,chunk,1,17)
        terminal=dict(session=1,attempt=1,result='ok',error=None,phase_a=[True,True],
                      cache_completion=[True,True],poisoned=False,offset=1)
        w.validate_terminal(terminal,chunk,1)
        for key,value in (('phase_a',[1,1]),('cache_completion',None),('poisoned',True),('offset',0)):
            bad=copy.deepcopy(terminal);bad[key]=value
            with self.assertRaises(ValueError):w.validate_terminal(bad,chunk,1)

    def test_16_free_argmax_stop_unmapped_and_repeat(self):
        import lfm2_capture_workload as w
        def sample(f,word_ids):
            values=np.zeros(65536,dtype='<f4')
            for i in word_ids:values[i]=2
            token=min(word_ids);index=len(f.sampled)
            e=dict(run=9+f.repeat,repeat=f.repeat,sample=index,distribution_ref=100+index,
                   id=token,stop=token==7,unmapped=token>=64400,pending=token!=7,
                   committed_count=len(f.committed)+(8 if index==0 else 1))
            f.sample(values.view('<u4'),e,100+index)
        a=w.FreeRepeat(0);d=w.FreeRepeat(1)
        for x in (a,d):sample(x,[7,8])
        w.compare_free_repeats(a,d)
        self.assertEqual(a.finish()['committed'],list(w.PLAIN))
        self.assertIsNone(a.finish()['pending'])
        with self.assertRaises(ValueError):a.next_chunk()
        bad=w.FreeRepeat(0);sample(bad,[64400,65535])
        self.assertEqual(bad.finish()['pending'],64400)
        self.assertIsNotNone(bad.finish()['codec_failure'])
        with self.assertRaises(ValueError):bad.next_chunk()
        f=w.FreeRepeat(0)
        nan=np.zeros(65536,dtype='<f4');nan[0]=np.nan
        with self.assertRaises(ValueError):f.sample(nan.view('<u4'),{},0)

    def test_17_file_truncation_growth_consumed_json_replacement(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'input';p.write_bytes(b'{"x":1}')
            f=c.File(p,1024)
            try:
                p.write_bytes(b'{"x":2}')
                with self.assertRaises(ValueError):f.json(1024)
                with p.open('ab') as out:out.write(b'x'*2048)
                with self.assertRaises(ValueError):f.hash()
                p.write_bytes(b'x')
                with self.assertRaises(ValueError):f.hash()
                p.unlink();p.write_bytes(b'{"x":1}')
                with self.assertRaises(ValueError):f.endpoint()
            finally:f.close()

    def test_18_summary_marker_prefix_and_missing_positive_harness(self):
        import check_lfm2_capture as r
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'stdout'
            for content,valid in ((b'test abc ... LFM2_REAL_CAPTURE_SUMMARY {"x":1}\n'
                                  b'test result: ok. 1 passed; 0 failed;\n',True),
                                 (b'LFM2_REAL_CAPTURE_SUMMARY {}\n',False),
                                 (b'LFM2_REAL_CAPTURE_SUMMARY {}\nLFM2_REAL_CAPTURE_SUMMARY {}\n'
                                  b'test result: ok. 1 passed; 0 failed;\n',False)):
                p.write_bytes(content);f=c.File(p,1024)
                try:
                    if valid:self.assertEqual(r.summary_from_stdout(f),{'x':1})
                    else:
                        with self.assertRaises(ValueError):r.summary_from_stdout(f)
                finally:f.close()

    def test_19_footer_prefix_is_not_complete_journal_hash(self):
        import check_lfm2_capture as r
        with tempfile.TemporaryDirectory() as tmp:
            raw=b'\0\0';first=[dict(seq=0,kind='start',body={}),array_record(raw)]
            prefix_raw=b''.join(encoded(v) for v in first)
            prefix=dict(raw_bytes=2,raw_sha256=b.digest(raw),journal_bytes=len(prefix_raw),
                        journal_sha256=b.digest(prefix_raw),records=2,arrays=1,boundaries=0,other=0)
            records=first+[dict(seq=2,kind='end',body={'prefix':prefix})]
            saved=Saved(tmp,raw,records);cap=saved.open()
            try:
                self.assertEqual(r.footer(cap)['prefix'],prefix)
                cap.records[-1]['body']['prefix']['journal_sha256']=cap.journal_file.sha256
                with self.assertRaises(ValueError):r.footer(cap)
            finally:cap.close()

    def test_20_exact_event_order_boolean_and_extra_array(self):
        import check_lfm2_capture as r
        from types import SimpleNamespace
        fake=SimpleNamespace(records=[dict(seq=0,kind='run_begin',body={'run':1})])
        reader=r.Reader(fake,None,None,{}, {})
        with self.assertRaises(ValueError):reader.event('session_begin')
        self.assertEqual(reader.cursor,0)
        with self.assertRaises(ValueError):reader.exact_event('run_begin',{'run':True})
        self.assertFalse(c.strict_equal({'x':[1,0]}, {'x':[True,False]}))
        fake.records=[dict(seq=0,kind='array',id=[1,1,0,4],dtype='f16le',query=[0,1],keys=[0,0],
                           rank=1,shape=[1024,1,1])]
        reader.cursor=0;reader.state=SimpleNamespace(session=1);reader.current_attempt=1
        with self.assertRaises(ValueError):reader.array(0,3,shape=(1024,),query=(0,1))

    def test_21_gather_and_view_dispatch_reject_wrong_signed_zero(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        gate={'atol':1e-4,'rtol':.001}
        zero=np.zeros(1024,dtype='<f4');negative=zero.copy();negative[17]=-0.0
        class Gold:
            groups={'cases.synthetic.row_0.embedding_lookup':{'gate':gate},
                    'cases.synthetic.row_0.layers.0.b':{'gate':gate}}
            def values(self,name,profile,keys=None):return zero.copy()
        weights=SimpleNamespace(gather=lambda token:zero.copy())
        gather=c.Stage('embedding_lookup',1024,'gather')
        with patch.object(c,'STAGES',(gather,)):
            d=c.Dispatcher(weights,Gold())
            d.selected('synthetic',0,1,0,1,lambda slot:negative.copy(),None,None)
            self.assertEqual(d.local_semantic_failures,1)
            self.assertEqual(d.local_numeric_failures,0)
        view=c.Stage('layers.0.b',1024,'view',('layers.0.bcx_projection',))
        parent=np.zeros(3072,dtype='<f4')
        with patch.object(c,'STAGES',(view,)),patch.dict(c.SLOTS,{'layers.0.bcx_projection':7}):
            d=c.Dispatcher(weights,Gold())
            d.selected('synthetic',0,1,0,1,lambda slot:parent if slot==7 else negative.copy(),None,None)
            self.assertEqual(d.local_semantic_failures,1)
            d2=c.Dispatcher(weights,Gold())
            d2.selected('synthetic',0,1,0,1,lambda slot:parent if slot==7 else zero.copy(),None,None)
            self.assertEqual(d2.local.failures,0)

    def test_22_numeric_zero_exception_and_unchanged_hard_guard(self):
        from unittest.mock import patch
        gate={'atol':1e-4,'rtol':.001}
        center=np.zeros(1024,dtype='<f4')
        class Gold:
            groups={'cases.synthetic.row_0.layers.0.u':{'gate':gate}}
            def values(self,name,profile,keys=None):return center.copy()
        stage=c.Stage('layers.0.u',1024,'product',('left','right'))
        wrong=center.copy();wrong[0]=-0.;wrong[1]=1.
        with patch.object(c,'STAGES',(stage,)),patch.dict(c.SLOTS,{'left':8,'right':9}):
            d=c.Dispatcher(None,Gold())
            d.selected('synthetic',0,1,0,1,lambda slot:center.copy() if slot in (8,9) else wrong,None,None)
            self.assertEqual(d.local_semantic_failures,1)
            self.assertEqual(d.local_numeric_failures,1)
            self.assertEqual(d.main.failures,1)
            self.assertEqual(d.local.first['index'],1)

    def test_23_missing_operand_and_unknown_route_hard_fail(self):
        from unittest.mock import patch
        class Gold:
            groups={'cases.synthetic.row_0.unknown':{'gate':{'atol':1e-4,'rtol':.001}}}
            def values(self,*args):return np.zeros(1,dtype='<f4')
        with patch.object(c,'STAGES',(c.Stage('unknown',1,'unimplemented'),)):
            with self.assertRaises(ValueError):
                c.Dispatcher(None,Gold()).selected('synthetic',0,1,0,1,
                    lambda slot:np.zeros(1,dtype='<f4'),None,None)
        fake=FakeCapture()
        with self.assertRaises(KeyError):fake.get(1,1,1,0)

    def test_24_model_owner_original_words_and_guard_contract(self):
        import check_lfm2_capture as r
        expected=dict(index=279,name='norm.weight',role='direct_gain',bytes=32,sha256='a'*64)
        actual={**expected,'address':4096,'device':77,'tail':[c.WORD_PAD]*7}
        r.Reader.validate_model_owner(actual,expected)
        for key,value in (('name','other.weight'),('sha256','b'*64),('tail',[0]*7),('bytes',28)):
            bad=copy.deepcopy(actual);bad[key]=value
            with self.assertRaises(ValueError):r.Reader.validate_model_owner(bad,expected)
        zero={**expected,'index':334,'name':'zero_bias','role':'zero_bias'}
        r.Reader.validate_model_owner({**zero,'address':8192,'device':77,'tail':[0]*7},zero)

    def test_25_eligibility_waits_for_all_endpoints(self):
        import check_lfm2_capture as r
        good=dict(structural_complete=True,codec_passed=True,numerical={'fatal_numerical_passed':True},error=None)
        self.assertTrue(r.finalize(copy.deepcopy(good),[])['eligible_for_bounded_native_conformance'])
        for key,value in (('structural_complete',False),('codec_passed',False),('error','read failure'),
                          ('numerical',{'fatal_numerical_passed':False})):
            bad=copy.deepcopy(good);bad[key]=value
            self.assertFalse(r.finalize(bad,[])['eligible_for_bounded_native_conformance'])
        out=r.finalize(copy.deepcopy(good),['source changed'])
        self.assertFalse(out['eligible_for_bounded_native_conformance'])
        self.assertFalse(out['numerical_qualification'])

    def test_26_complete_coverage_not_inferred_from_zero_failures(self):
        d=c.Dispatcher(None,None)
        with self.assertRaises(ValueError):d.finish()
        d.coupled.groups=303;d.coupled.values=1965024
        d.local.groups=6440;d.local.values=11971552
        d.main.groups=d.wide.groups=6743
        d.main.values=d.wide.values=13936576
        self.assertTrue(d.finish()['fatal_numerical_passed'])
        d.local_semantic_failures=1;d.local.failures=1
        self.assertFalse(d.finish()['fatal_numerical_passed'])
        d.local.values-=1
        with self.assertRaises(ValueError):d.finish()


    def test_27_main_admission_failure_retains_ineligible_result(self):
        import check_lfm2_capture as r
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            control=Path(tmp)/'control.json';control.write_text('{"format":"synthetic"}')
            output=Path(tmp)/'result.json'
            argv=['reader','--control',str(control),'--control-sha256','0'*64,'--output',str(output)]
            with patch('sys.argv',argv):
                self.assertEqual(r.main(),1)
            actual=json.loads(output.read_bytes())
            self.assertIn('reader consumed invocation identity',actual['error'])
            self.assertFalse(actual['passed'])
            self.assertFalse(actual['eligible_for_bounded_native_conformance'])
            self.assertFalse(actual['numerical_qualification'])
            self.assertEqual(actual['endpoint_errors'],[])
            with patch('sys.argv',argv),self.assertRaises(FileExistsError):r.main()

    def test_28_original_rank3_tap_axis_and_descriptor_guard(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        weights=c.Weights.__new__(c.Weights)
        name='model.layers.0.conv.conv.weight'
        weights.header={name:dict(dtype='BF16',shape=[1024,3,1],data_offsets=[0,6144])}
        self.assertEqual(weights.descriptor(name,(1024,3,1),'BF16')['shape'],[1024,3,1])
        with self.assertRaises(ValueError):weights.descriptor(name,(1024,1,3),'BF16')
        stage=c.Stage('layers.0.convolution',1024,'conv',tensor=name,layer=0)
        seen=[]
        class W:
            def coefficient(self,tensor,shape):
                seen.append((tensor,shape))
                if shape!=(1024,3,1):raise ValueError('wrong original axes')
                return np.ones(shape,dtype='<f4')
        class Gold:
            groups={'cases.synthetic.row_0.layers.0.convolution':{'gate':{'atol':1e-4,'rtol':.001}}}
            def values(self,*args):return np.zeros(1024,dtype='<f4')
        with patch.object(c,'STAGES',(stage,)):
            d=c.Dispatcher(W(),Gold())
            d.selected('synthetic',0,1,0,1,lambda slot:np.zeros(1024,dtype='<f4'),
                       lambda layer:np.zeros((1,1024),dtype='<f4'),
                       lambda layer:np.zeros((2,1024),dtype='<f4'))
            self.assertEqual(d.local.failures,0)
            self.assertEqual(seen,[(name,(1024,3,1))])


    def test_29_source_state_drift_is_reported_separately_from_publication(self):
        import check_lfm2_capture as r
        from lfm2_capture_workload import Chunk
        class C(FakeCapture):
            def values(self,d):return self.words(d).view('<f2').astype('<f4')
        capture=C();groups={}
        class Gold:
            def values(self,name,profile):
                shape=(1,2,1024) if 'history_' in name else ((1,1024) if 'u_producer' in name else (1,8,64))
                return np.zeros(shape,dtype='<f4')
        source=Gold();source.groups=groups
        def group(layer,name):
            groups[f'cases.synthetic.layers.{layer}.{name}']=dict(route='state_producer_sideband',
                gate_usage='diagnostic sideband',gate={'atol':1e-4,'rtol':.001})
        for owner,layer in enumerate(c.CONVOLUTION):
            capture.put(1,layer,np.zeros((1,1024),dtype='<u2'),(0,1));group(layer,'u_producer')
            for kind,name in ((4,'history_before'),(5,'history_after')):
                capture.put(kind,2*owner,np.zeros(2055,dtype='<u2'));group(layer,name)
        for owner,layer in enumerate(c.ATTENTION):
            for role,kind,name in ((0,2,'kv_keys_producer'),(1,3,'kv_values_producer')):
                capture.put(kind,layer,np.zeros((1,512),dtype='<u2'),(0,1))
                capture.put(7,2*owner+role,np.zeros((8,1,64),dtype='<u2'));group(layer,name)
        capture.items[(1,0,(0,1))]['words'][0,0]=0x3c00
        reader=r.Reader(capture,source,None,{},{});reader.current_attempt=1
        reader.source_state_diagnostics(Chunk(0,1,0,(1,),(),'synthetic'))
        self.assertEqual(reader.source_state_main.groups,54)
        self.assertEqual(reader.source_state_main.values,63488)
        self.assertEqual(reader.source_state_main.failures,1)
        self.assertEqual(reader.source_state_wide.failures,1)
        self.assertEqual(len(reader.source_state_identities),42)
        self.assertEqual(reader.dispatch.local.groups,0)
        self.assertEqual(reader.dispatch.coupled.failures,0)


if __name__ == '__main__':
    unittest.main()
