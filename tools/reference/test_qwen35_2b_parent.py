"""Portable byte/stream controls; no model tensors or numerical imports."""
import copy
import hashlib
import io
import struct
import unittest

import qwen35_2b_parent as parent
from test_qwen35_2b_layout import fixture


class Parent(unittest.TestCase):
    def test_complete_certificate_requires_all_tensors_and_both_original_rate_gates(self):
        protocol=parent.protocol();rate=dict(passed=True,values=16,failed_coordinates=0,
                                             first_failed_coordinates=[],atol=1e-5,rtol=1e-4)
        rows=[]
        for target,spec in parent.source_mapping().items():
            count=__import__('math').prod(spec['shape'])
            rows.append(dict(name=spec['name'],parent_name=target,kind=spec['kind'],passed=True,
                source_bytes=count*(4 if spec['dtype']=='F32' else 2),parent_bytes=count*(2 if spec['kind']=='matrix' else 4),
                source_sha256='1'*64,parent_sha256='2'*64,
                functional_rate_profiles={'staged_fp32':dict(rate),'ideal_fp64':dict(rate)} if spec['kind']=='A_log' else None))
        value=dict(format='toklet-qwen35-2b-parent-lineage-v1',passed=True,protocol_sha256='3'*64,protocol=protocol,
            transport=dict(https_only=True,certificate_verification='CERT_REQUIRED',hostname_verification=True,
                TLS_verified_for_completed_stream=True,ca=dict(path='/synthetic/ca.pem',bytes=4,sha256='4'*64,
                    before_identity=[1,2,4,3,4],after_identity=[1,2,4,3,4])),
            parent=dict(parent.PARENT,whole_SHA256_authenticated=True,EOF_checked=True,retained_payload_bytes=0),
            selected_source=dict(parent.layout.SOURCE,whole_SHA256_authenticated_before=True,whole_SHA256_authenticated_after=True,
                before_identity=[1,2,parent.layout.SOURCE_BYTES,3,4],after_identity=[1,2,parent.layout.SOURCE_BYTES,3,4]),
            tensors=rows,historical_publisher_lineage_proved=False,A_log_bitwise_reconstruction_claimed=False,
            model_numerics_quality_fit_performance_adoption_qualified=False)
        self.assertIs(parent.validate_certificate(value,'3'*64),value)
        bad=copy.deepcopy(value);bad['tensors']=bad['tensors'][:-1]
        with self.assertRaises(ValueError):parent.validate_certificate(bad,'3'*64)
        bad=copy.deepcopy(value);bad['transport']['hostname_verification']=False
        with self.assertRaises(ValueError):parent.validate_certificate(bad,'3'*64)
        bad=copy.deepcopy(value);bad['transport']['ca']['sha256']='0'*64
        with self.assertRaises(ValueError):parent.validate_certificate(bad,'3'*64)
        bad=copy.deepcopy(value)
        next(row for row in bad['tensors'] if row['kind']=='A_log')['functional_rate_profiles']['ideal_fp64']['atol']=1e-3
        with self.assertRaises(ValueError):parent.validate_certificate(bad,'3'*64)

    def test_complete_frozen_metadata_directory_and_source_mapping(self):
        rows = parent.directory(fixture('parent-directory.bin'))
        self.assertEqual(len(rows), 320)
        self.assertEqual(sum(r['kind'] == 'matrix' for r in rows), 187)
        self.assertEqual(sum(r['kind'] == 'A_log' for r in rows), 18)
        self.assertEqual(sum(r['kind'] == 'narrowed_BF16' for r in rows), 115)
        self.assertEqual(rows[-1]['parent_offset']+rows[-1]['parent_bytes'], parent.PARENT['bytes']-parent.DATA_START)
        self.assertEqual(sum(r['parent_bytes']//2 for r in rows if r['kind'] == 'matrix'), 1881276416)
        for raw in (fixture('parent-directory.bin')[:-1], fixture('parent-directory.bin')+b' ', bytes(18837)):
            with self.assertRaises(ValueError):
                parent.directory(raw)

    def test_primary_rules_and_original_gates_are_explicit_before_payload(self):
        p = parent.protocol()
        self.assertEqual((p['rules']['atol'], p['rules']['rtol']), (1e-5, 1e-4))
        self.assertEqual(p['rules']['profiles'], ['staged_fp32', 'ideal_fp64'])
        self.assertEqual(len(p['rules']['source_sha256']), 9)
        self.assertFalse(p['historical_publisher_lineage_proved'])
        self.assertFalse(p['model_numerics_quality_fit_performance_adoption_qualified'])

    def test_RNE_ties_signed_zero_and_subnormal_literals(self):
        words = [0x3f808000, 0x3f818000, 0xbf808000, 0xbf818000, 0x00008000, 0x00018000, 0x80000000]
        expected = [0x3f80, 0x3f82, 0xbf80, 0xbf82, 0, 2, 0x8000]
        raw = struct.pack('<7I', *words)
        self.assertEqual(parent.narrow_f32(raw), struct.pack('<7H', *expected))
        for word in (0x7f800000, 0xff800000, 0x7fc00001, 0x7f7fffff):
            with self.assertRaises(ValueError):
                parent.narrow_f32(struct.pack('<I', word))
        with self.assertRaises(ValueError):
            parent.narrow_f32(b'bad')

    def test_functional_rate_profiles_are_separate_and_each_gate_must_pass(self):
        raw = struct.pack('<2f', -1, -2)
        source = struct.pack('<2f', 0, 1)
        profiles = dict(staged_fp32=raw, ideal_fp64=[-1.0, -2.0])
        result = parent.rate_parity(raw, source, lambda _:profiles)
        self.assertTrue(all(item['passed'] for item in result.values()))
        bad = dict(profiles, ideal_fp64=[-1.01, -2.0])
        result = parent.rate_parity(raw, source, lambda _:bad)
        self.assertTrue(result['staged_fp32']['passed'])
        self.assertFalse(result['ideal_fp64']['passed'])
        self.assertEqual(result['ideal_fp64']['first_failed_coordinates'], [0])

    def test_nonfinite_positive_or_missing_rate_oracles_reject(self):
        for values in ((0, -1), (1, -1), (float('nan'), -1), (float('-inf'), -1)):
            with self.assertRaises(ValueError):
                parent.rate_parity(struct.pack('<2f', *values), bytes(8), lambda _:None)
        with self.assertRaises(ValueError):
            parent.rate_parity(struct.pack('<f', -1), struct.pack('<f', float('nan')), lambda _:None)
        with self.assertRaises(ValueError):
            parent.rate_parity(struct.pack('<f', -1), bytes(4), lambda _:dict(ideal_fp64=[-1.0]))

    def test_stream_short_reads_are_completed_and_digest_covers_every_byte(self):
        raw = b'framed whole input'
        class Short:
            def __init__(self):
                self.stream = io.BytesIO(raw)
            def read(self, count):
                return self.stream.read(min(3, count))
        owner = parent.HashedStream(Short())
        self.assertEqual(owner.read(4), raw[:4])
        owner.skip_to(len(raw))
        self.assertEqual(owner.digest.hexdigest(), hashlib.sha256(raw).hexdigest())
        with self.assertRaises(ValueError):
            owner.read(1)
        with self.assertRaises(ValueError):
            owner.skip_to(0)

    def test_stream_byte_comparison_rejection_retains_whole_remaining_body(self):
        matrix = struct.pack('<4H', 0, 0x3f80, 0x4000, 0x4040)
        narrowed = struct.pack('<4f', 1, 2, 3, 4)
        entries = [dict(name='matrix', parent_name='m', kind='matrix', parent_bytes=8),
                   dict(name='small', parent_name='s', kind='narrowed_BF16', parent_bytes=16)]
        source = matrix[:-2]+b'\0\0'+struct.pack('<4H', 0x3f80, 0x4000, 0x4040, 0x4080)
        spans = {'matrix':dict(start=0, end=8), 'small':dict(start=8, end=16)}
        stream, events = io.BytesIO(matrix+narrowed), []
        reports = parent.compare_payload(stream.read, entries, lambda n,o:source[o:o+n], spans,
                                         lambda _:True, None, events.append)
        self.assertFalse(reports[0]['passed'])
        self.assertTrue(reports[1]['passed'])
        self.assertEqual(stream.tell(), 24)
        self.assertEqual(len(events), 4)
        self.assertTrue(all(e['whole_parent_authenticated'] is False for e in events))

    def test_incomplete_or_changed_certificate_cannot_open_conversion_gate(self):
        for value in ({}, dict(format='toklet-qwen35-2b-parent-lineage-v1', passed=False),
                      dict(format='toklet-qwen35-2b-parent-lineage-v1', passed=True, protocol_sha256='1'*64,
                           protocol=dict(parent.protocol(), historical_publisher_lineage_proved=True))):
            with self.assertRaises(ValueError):
                parent.validate_certificate(value, '1'*64)


if __name__ == '__main__':
    unittest.main()
