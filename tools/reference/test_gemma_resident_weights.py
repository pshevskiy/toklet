"""Verifier controls only; synthetic packets never qualify native execution."""
from pathlib import Path
import copy
import sys
import unittest
import check_gemma_resident_weights as check


class Replay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = check.inventory(Path(sys.argv[-1]))

    def packet(self):
        reference = self.reference
        report = dict(context_dropped=True, matrices=278, vectors=262, identity_components=1096,
            resident_bytes=2604665212, policy_bytes=2821589444, source_bytes=82575432,
            matrix_slots=278, vector_slots=262, bundle_abi=6728, vector_abi=24, weights_abi=112,
            process_policy_bytes=6*1024**3, diagnostic_policy_bytes=32*1024**2,
            available_bytes=6*1024**3-32*1024**2, container_identity=reference['model']['sha256'],
            container_bytes=reference['model']['bytes'])
        for key, name in [('config_digest', 'config.json'), ('index_digest', 'index.json'), ('header_digest', 'header.json')]:
            report[key] = list(bytes.fromhex(reference['metadata'][name]['sha256']))
        events = [('BUDGET', dict(rejected=True, required=2821589444, limit=2821589443,
            source_bytes_read=318228, error='Metal admission requires 2821589444 bytes, budget 2821589443')), ('LOAD', dict(ok=True, error=None)), ('REPORT', report)]
        identity = 0
        def buffer(expected):
            nonlocal identity
            identity += 1
            return dict(allocation=hex(4096+identity*4096), bytes=expected['guarded_bytes'], sha256=expected['guarded_sha256'])
        for number, expected in enumerate(reference['matrices']):
            chunks=[]; owned=112+len(expected['chunks'])*112
            for part in expected['chunks']:
                child=dict(first=part['first'], rows=part['rows'])
                for role in ('codes','scales','biases'):
                    child[role]=buffer(part['buffers'][role]);owned+=child[role]['bytes']
                chunks.append(child)
            events.append(('MATRIX', dict(index=number, rows=expected['rows'], columns=expected['columns'],
                chunk_slots=len(chunks), chunk_abi=112, device='0xabc', owned_bytes=owned, chunks=chunks)))
        for number, expected in enumerate(reference['vectors']):
            events.append(('VECTOR', dict(index=number, values=expected['count'], buffer=buffer(expected))))
        events.append(('COMPLETE', dict(passed=True, context_dropped=True, model_matrices=278,
            model_vectors=262, source_path_and_descriptor_unchanged=True, whole_graph_qualified=False,
            performance_qualified=False, phone_qualified=False, source_or_precision_failures_waived=False)))
        return events

    def rejects(self, mutate):
        data=self.packet();mutate(data)
        with self.assertRaises((ValueError, KeyError, StopIteration, TypeError)):
            check.real(data,self.reference)

    def test_complete_synthetic_verifier_packet(self):
        result=check.real(self.packet(),self.reference)
        self.assertEqual(result['guarded_allocations'],1186)
        self.assertFalse(result['whole_graph_qualified'])

    def test_missing_reordered_duplicate_and_partial_records(self):
        for mutate in [lambda d:d.pop(4),lambda d:d.append(d[-1]),lambda d:d.reverse(),lambda d:d.pop()]:
            self.rejects(mutate)

    def test_complete_buffer_hash_and_padding_rejection(self):
        self.rejects(lambda d:d[3][1]['chunks'][0]['scales'].update(sha256='0'*64))
        self.rejects(lambda d:d[-2][1]['buffer'].update(bytes=4))

    def test_alias_and_multidevice_rejection(self):
        self.rejects(lambda d:d[4][1]['chunks'][0]['codes'].update(allocation=d[3][1]['chunks'][0]['codes']['allocation']))
        self.rejects(lambda d:d[4][1].update(device='0xdef'))

    def test_accounting_source_identity_and_early_io(self):
        for mutate in [lambda d:d[2][1].update(resident_bytes=1),lambda d:d[2][1].update(source_bytes=1),
            lambda d:d[2][1].update(container_identity='0'*64),lambda d:d[0][1].update(source_bytes_read=318229),
            lambda d:d[2][1].update(matrix_slots=277),lambda d:d[3][1].update(chunk_slots=0)]:
            self.rejects(mutate)

    def test_scope_and_unknown_fields_reject(self):
        for mutate in [lambda d:d[-1][1].update(phone_qualified=True),lambda d:d[2][1].update(whole_graph_qualified=True),
            lambda d:d[3][1].update(unqualified_note=True),lambda d:d[1][1].update(error='load failed')]:
            self.rejects(mutate)

    def test_wrapper_partial_duplicate_json_and_wrong_exact_test(self):
        for body in [b'', b'running 1 test\n', b'GEMMA_RESIDENT_REAL_LOAD {"ok":true,"ok":false}\n',
                     b'test wrong ... GEMMA_RESIDENT_REAL_LOAD {}\n']:
            with self.assertRaises((ValueError,KeyError)):
                check.records(body,'real')

    def tiny_packet(self):
        cases=check.frozen.fixture(Path(sys.argv[-2]))
        matrices,vectors,resident=check.tiny_expected(cases)
        minimum=resident+check.CONTEXT+check.COLD+check.STACK+1440
        identity=0
        def buffer(words):
            nonlocal identity
            identity+=1
            return dict(allocation=hex(4096*identity),words=len(words),data=words)
        def storage():
            result=dict(resident_bytes=resident,matrix_slots=2,vector_slots=2,matrices=[],vectors=[])
            for matrix in matrices:
                value={key:value for key,value in matrix.items() if key!='chunks'}
                child=dict(rows=matrix['rows'])
                for role in ('codes','scales','biases'):
                    child[role]=buffer(matrix['chunks'][0][role])
                value['chunks']=[child];result['matrices'].append(value)
            for vector in vectors:
                result['vectors'].append(dict(values=len(vector),buffer=buffer(vector+[check.S]*7)))
            return result
        old=storage()
        plan=dict(bundle_struct_bytes=6728,vector_struct_bytes=24,weights_struct_bytes=112,
            minimum_policy_bytes=minimum,base=7000,payload=resident-7000,source=1440,
            cold_policy=check.COLD,context_policy=check.CONTEXT,hash_stack_policy=check.STACK,
            real_e2b_model=False,whole_graph_qualified=False,process_allowance_bytes=512*1024**2,
            diagnostic_policy_bytes=32*1024**2,reader_backing_bytes=4096,
            initial_available_bytes=512*1024**2-32*1024**2-4096)
        events=[('PLAN',plan),('INITIAL',dict(source_context_dropped=True,identity_components=8,
            resident_bytes=resident,policy_bytes=minimum,source_bytes=1440,storage=old))]
        for number,case in enumerate(cases):
            events.append(('PROJECTION',dict(matrix=number,ok=True,bits=case['expected'])))
        errors={'budget':f'Metal admission requires {minimum} bytes, budget {minimum-1}',
            'changed':'Gemma source matrix changed/incomplete between passes',
            'late-read':'source IO: resident original late read failure',
            'late-vector':'Gemma resident vector nonfinite',
            'late-codebook':'Gemma affine metadata domain'}
        for mode in ['budget','changed','late-read','late-vector','late-codebook','valid-replacement']:
            outcome=dict(key=mode,ok=mode=='valid-replacement')
            if mode=='valid-replacement':
                outcome.update(resident_bytes=resident,policy_bytes=minimum,source_bytes=1440,
                    matrix_slots=2,vector_slots=2,candidate=storage())
            else:outcome.update(error=errors[mode],failed_candidate_buffers_captured=False)
            events.append(('OUTCOME',dict(outcome=outcome,previous=copy.deepcopy(old))))
            state=dict(position=10,bytes_read=0,seeks=0,historical_eof_observed=False) if mode=='budget' else dict(position=100,bytes_read=200,seeks=2,historical_eof_observed=True)
            events.append(('READER',dict(key=mode,state=state)))
        events.append(('COMPLETE',dict(passed=True,steps=6,context_dropped=True,projection_values=90,
            failed_candidate_buffers_captured=False,real_e2b_model=False,whole_graph_qualified=False,
            source_or_precision_failures_waived=False,performance_qualified=False)))
        return events,cases

    def test_complete_synthetic_tiny_packet(self):
        events,cases=self.tiny_packet()
        self.assertEqual(check.tiny(events,cases)['projection_values'],90)

    def test_tiny_signed_zero_previous_state_and_replacement_alias(self):
        for change in range(4):
            events,cases=self.tiny_packet()
            if change==0:events[1][1]['storage']['vectors'][0]['buffer']['data'][0]=0
            if change==1:events[4][1]['previous']['matrix_slots']=3
            if change==2:
                events[-3][1]['outcome']['candidate']['matrices'][0]['chunks'][0]['codes']['allocation']=events[1][1]['storage']['matrices'][0]['chunks'][0]['codes']['allocation']
            if change==3:events[-1][1]['failed_candidate_buffers_captured']=True
            with self.assertRaises(ValueError):check.tiny(events,cases)

    def test_tiny_fixture_shapes_and_signed_zero(self):
        directory=Path(sys.argv[-2])
        matrices,vectors,resident=check.tiny_expected(check.frozen.fixture(directory))
        self.assertEqual([(v['rows'],v['columns']) for v in matrices],[(19,128),(11,64)])
        self.assertEqual(vectors[0][0],0x80000000)
        self.assertGreater(resident,6728)
        with self.assertRaises(StopIteration):
            check.tiny([],check.frozen.fixture(directory))


if __name__=='__main__':
    unittest.main(argv=[sys.argv[0]],exit=True)
