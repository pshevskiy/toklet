"""Cold replay controls: retained-value alignment, coherent failures and framing."""
from pathlib import Path
import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import check_gemma_hybrid1025 as check
import test_gemma4_e2b_wrap_reference as source_controls


def fixture():
    events = []; body = bytearray(); number = 0
    def emit(name, value):
        events.append((name, value))
    def array(name, before, count, value):
        nonlocal number
        raw = np.asarray(value, dtype='<f4').tobytes()
        fields = dict(record=number, name=name, offset=len(body), bytes=len(raw),
                      before=before, count=count, values=len(raw)//4)
        emit('ARRAY_ATTEMPT', fields)
        emit('ARRAY', dict(**fields, sha256=hashlib.sha256(raw).hexdigest()))
        body.extend(raw); number += 1
    for index, partition in enumerate(check.PARTITIONS):
        emit('REAL_WRAP_SCHEDULE', dict(index=index, chunks=partition))
        n = 0
        for count in partition:
            if n == 0:
                emit('REAL_WRAP_PREFIX', dict(schedule=index, sha256='a'*64))
            for position in check.POSITIONS:
                if n <= position < n+count:
                    array('wrap.logits', position, 1, np.zeros(262144, dtype='<f4'))
            n += count
            if index == 0 and n in check.MILESTONES:
                for owner in range(15):
                    full = owner % 5 == 4
                    capacity = 1025 if full else 512
                    dim = 512 if full else 256
                    start = 0 if full else max(0, n-512)
                    slots = [None]*capacity
                    for pos in range(start, n):
                        slots[pos % capacity] = pos
                    emit('REAL_WRAP_SLOTS', dict(committed=n, owner=owner, slots=slots))
                    values = np.broadcast_to(np.arange(start, n, dtype='<f4')[:, None], (n-start, dim))
                    for kind in ['k', 'v']:
                        array(f'wrap.layer.{owner}.{kind}', n, n-start, values)
        emit('REAL_WRAP_STATE', dict(schedule=index, committed=1025, overflow_preserved=True,
             reset_all32_rows_exact=True, reset_all15_valid_kv_prefixes_exact=True,
             allocated_bytes=271117548, cross_schedule_endpoints=[256, 1025]))
    emit('CAPTURE', dict(records=number, bytes=len(body), sha256=hashlib.sha256(body).hexdigest(), whole_e2b_qualified=False))
    emit('REAL_WRAP_COMPLETE', dict(state_passed=True, numerical_decision='separate reference replay',
         positions=1025, selected_positions=check.POSITIONS, retained_milestones=check.MILESTONES,
         prior_precision_failures_waived=False, app_qualified=False, phone_qualified=False))
    return events, bytes(body)


def transcript(events):
    lines = ['running 1 test']
    for index, (name, value) in enumerate(events):
        prefix = f'test {check.TEST} ... ' if index == 0 else ''
        lines.append(prefix+'GEMMA_HYBRID_'+name+' '+json.dumps(value))
    lines.extend(['ok', 'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 999 filtered out; finished in 1.00s'])
    return ('\n'.join(lines)+'\n').encode()


class WrapCaptureControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.events, cls.body = fixture()

    def test_complete_replay_and_coherent_reference_rejection(self):
        with tempfile.TemporaryDirectory(dir=check.ROOT / 'artifacts') as name, \
                patch.object(check, 'qualified_prefix', return_value='a'*64), \
                patch.object(check.oracle.source_graph.source, 'PINS', {}):
            root = Path(name)
            helper = source_controls.WrapReferenceControls()
            produced = helper.run_fake(root, helper.fake_graph)
            self.assertTrue(produced['completed'])
            result = check.replay(transcript(self.events), self.body, root/'output')
            self.assertTrue(result['state_passed'] and result['overflow_reset_passed'] and result['numerical_passed'])
            record = json.loads((root/'output/result.json').read_text())
            raw = bytearray((root/'output/rows.bin').read_bytes())
            for item in record['arrays']:
                if item['kind'] == 'k' and item['owner'] == 0:
                    dtype = '<f8' if item['contract'] == 'math_fp64' else '<f4'
                    values = np.frombuffer(raw, dtype=dtype, count=1025*256, offset=item['offset'])
                    values[1024*256] = 1025.
                    item['sha256'] = hashlib.sha256(raw[item['offset']:item['offset']+item['bytes']]).hexdigest()
                    del values
            record['rows_sha256'] = hashlib.sha256(raw).hexdigest()
            (root/'output/rows.bin').write_bytes(raw)
            (root/'output/result.json').write_text(json.dumps(record))
            rejected = check.replay(transcript(self.events), self.body, root/'output')
            self.assertTrue(rejected['state_passed'] and rejected['overflow_reset_passed'])
            self.assertFalse(rejected['numerical_passed'])
            for kind in check.CONTRACTS:
                self.assertEqual(rejected['numerical'][kind]['failed_values'], 1)
            failed = next(v for v in rejected['rows'] if v['kind']=='k' and v.get('owner')==0 and v.get('committed')==1025)
            self.assertEqual(failed['contracts']['math_fp64']['examples'][0]['absolute_position'], 1024)

    def test_slots_state_wrapper_and_truncation_reject(self):
        with patch.object(check, 'qualified_prefix', return_value='a'*64):
            for mutation in ['slots', 'reset', 'wrapper', 'truncate']:
                events = json.loads(json.dumps(self.events))
                if mutation == 'slots':
                    next(v for k, v in events if k=='REAL_WRAP_SLOTS' and v['committed']==513 and v['owner']==0)['slots'][0] = 0
                if mutation == 'reset':
                    next(v for k, v in events if k=='REAL_WRAP_STATE')['reset_all32_rows_exact'] = False
                text = transcript(events)
                if mutation == 'wrapper':
                    text = text.replace(b'1 passed; 0 failed', b'0 passed; 1 failed')
                with self.assertRaises(ValueError, msg=mutation):
                    check.native_capture(text, self.body[:-4] if mutation=='truncate' else self.body)


if __name__ == '__main__':
    unittest.main()
