"""Cold same-input controls, with frozen primitive and coherent corruptions."""
import hashlib
import json
import struct
import unittest
from unittest.mock import patch
import numpy as np
import check_gemma_v_boundary as check
import test_gemma4_text_math as fixture


def capture():
    events = []; body = bytearray(); number = 0
    def emit(name, value):
        events.append((name, value))
    def array(owner, position, name, values):
        nonlocal number
        raw = np.asarray(values, dtype='<f4').tobytes()
        fields = dict(record=number, name=f'vboundary.{owner}.{name}', offset=len(body),
                      bytes=len(raw), before=position, count=1, values=len(values))
        emit('ARRAY_ATTEMPT', fields)
        emit('ARRAY', dict(**fields, sha256=hashlib.sha256(raw).hexdigest()))
        body.extend(raw); number += 1
    emit('REAL_VBOUNDARY_CONTRACT', dict(cases=[[a,b] for a,b,_ in check.CASES], input_tokens=651, capacity=1025, chunk_rows=32))
    outputs = {}
    for owner, position, dim in check.CASES:
        start = position//32*32
        emit('REAL_VBOUNDARY_CASE', dict(owner=owner, position=position, before=start,
             count=min(32,651-start), head_dim=dim, fault_preserved_prefix=True))
        x = np.linspace(-.5,.5,1536,dtype='<f4')
        v = x[:dim]
        # Independent vector reduction, rather than the checker's scalar fsum.
        denominator = np.sqrt(np.mean(v.astype('<f8')**2)+float(np.float32(1e-6)))
        normalized = (v.astype('<f8')/denominator).astype('<f4')
        outputs[owner] = normalized
        for name, values in [('input',x), ('raw',v), ('normalized',normalized)]:
            array(owner, position, name, values)
    for owner, position, _ in check.CASES:
        array(owner, position, 'retained', outputs[owner])
    emit('CAPTURE', dict(records=12, bytes=len(body), sha256=hashlib.sha256(body).hexdigest(), whole_e2b_qualified=False))
    emit('REAL_VBOUNDARY_COMPLETE', dict(committed=651, all3_stage_retained_exact=True,
         allocated_bytes=271117548, numerical_decision='separate same-input replay',
         earlier_failures_waived=False, app_qualified=False, phone_qualified=False))
    return events, bytes(body)


def transcript(events):
    lines = ['running 1 test']
    for index,(name,value) in enumerate(events):
        prefix = f'test {check.TEST} ... ' if index==0 else ''
        lines.append(prefix+'GEMMA_HYBRID_'+name+' '+json.dumps(value))
    return ('\n'.join(lines)+'\nok\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 999 filtered out; finished in 1.00s\n').encode()


class SparseProjection:
    """Known rectangular identity selects the first D activation components."""
    def linear(self, name, x, dtype):
        owner = int(name.split('.')[1])
        assert name==f'layers.{owner}.self_attn.v_proj.weight'
        dim = next(d for o,_,d in check.CASES if o==owner)
        return x[:,:dim].astype(dtype)


class BoundaryControls(unittest.TestCase):
    def test_norm_matches_frozen_staged_primitive(self):
        weights,_,_ = fixture.fixture(); states = {}
        fixture.graph.evaluate(weights,[2,7,13,7],True,sink=lambda n,v:states.update({n:v.copy()}))
        for owner in range(2):
            raw = states[f'layer.{owner}.self_attn.v_proj']
            expected = states[f'layer.{owner}.self_attn.v_norm'].reshape(-1,raw.shape[-1])
            for x,y in zip(raw.reshape(expected.shape),expected,strict=True):
                np.testing.assert_array_equal(check.normalized(x,True), y)
                np.testing.assert_array_equal(check.normalized(x,False).astype('<f4'), y)
        with self.assertRaises(ValueError):
            check.normalized([float('nan')],True)

    def test_complete_replay_and_local_numerical_rejection(self):
        events,body = capture(); rows = check.native_rows(transcript(events),body)
        result = check.compare(rows,SparseProjection())
        self.assertTrue(result['local_operations_passed'])
        broken = dict(rows)
        value = np.frombuffer(rows[13,'raw'],dtype='<f4').copy(); value[0] += .1
        broken[13,'raw'] = value.tobytes()
        self.assertFalse(check.compare(broken,SparseProjection())['local_operations_passed'])
        old = {}
        for owner,position,dim in check.CASES:
            committed = 512 if position==433 else 1024
            first = 0 if owner%5==4 else max(0,committed-512)
            data = bytearray((committed-first)*dim*4)
            start = (position-first)*dim*4
            data[start:start+dim*4] = rows[owner,'normalized']
            old[committed,owner,'v'] = bytes(data)
        with patch.object(check.prior,'native_capture',return_value=({},old,[])):
            check.earlier_rows(rows,b'',b'')
            corrupted = dict(rows); corrupted[12,'normalized'] = bytes(256*4)
            with self.assertRaises(ValueError):
                check.earlier_rows(corrupted,b'',b'')

    def test_coherent_cache_change_and_bad_wrapper_reject(self):
        events,body = capture(); changed = bytearray(body)
        item = next(v for n,v in events if n=='ARRAY' and v['name']=='vboundary.13.retained')
        struct.pack_into('<f',changed,item['offset'],.25)
        item['sha256'] = hashlib.sha256(changed[item['offset']:item['offset']+item['bytes']]).hexdigest()
        next(v for n,v in events if n=='CAPTURE')['sha256'] = hashlib.sha256(changed).hexdigest()
        with self.assertRaisesRegex(ValueError,'transfer'):
            check.native_rows(transcript(events),changed)
        events,body = capture()
        with self.assertRaises(ValueError):
            check.native_rows(transcript(events).replace(b'1 passed; 0 failed',b'0 passed; 1 failed'),body)
        with self.assertRaises(ValueError):
            check.native_rows(transcript(events),body[:-4])


if __name__=='__main__':
    unittest.main()
