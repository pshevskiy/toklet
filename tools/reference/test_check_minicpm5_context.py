"""Portable interval/coverage/owner controls; no model or numerical imports."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import check_minicpm5_context as reader
import test_minicpm5_context_reference as fixtures


def source_fixture(folder):
    contract,old,quality,tokens=fixtures.fixture()
    _,inventory,_=reader.producer.plan(contract,old,quality,tokens)
    arrays={};files={}
    for profile in ['fp32','fp64']:
        cursor=0
        for name,spec in inventory.items():
            size=spec['shape'][0]*spec['shape'][1]*(4 if spec['dtype']=='<f4' else 2)
            arrays[profile+'.'+name]=dict(**spec,file=profile+'.bin',offset=cursor,bytes=size,sha256='a'*64)
            cursor+=size
        files[profile+'.bin']=dict(bytes=cursor,sha256='a'*64)
    value=dict(format='toklet-minicpm5-context1025-reference-v1',completed=True,
        native_outcomes_seen=False,quality_scored=False,managed_adoption_qualified=False,gates_recomputed_or_relaxed=False,
        gates=contract['gates'],columns=contract['columns'],cases=contract['cases'],
        contract_sha256=reader.producer.CONTRACT_SHA,quality_manifest_sha256=reader.producer.QUALITY_SHA,
        original_reference_manifest_sha256=reader.producer.OLD_SHA,
        elapsed_ns=6,profile_records=[dict(case=f'prefix{n}',profile=p,elapsed_ns=1)
            for n in reader.COUNTS for p in ['fp32','fp64']],
        model=dict(repository=reader.producer.model.REPOSITORY,revision=reader.producer.model.REVISION,files=reader.producer.model.PINS),
        memory_admission=reader.producer.admission(),graph_identity=reader.producer.graph_identity(
            Path(reader.producer.model.__file__).read_bytes(),Path(reader.producer.graph.__file__).read_bytes()),arrays=arrays,files=files)
    return value,contract,old,quality,tokens


class Reader(unittest.TestCase):
    def test_import_does_not_execute_numpy_mlx_or_torch(self):
        for name in ['numpy','mlx.core','torch']:self.assertNotIn(name,sys.modules)

    def test_exact_frame_source_row_storage_and_typed_offsets(self):
        row=dict(event='completed',kind='prefill',case='prefix128',chunk=32,stage='logits',row_start=64,
                 shape=[32,130560],dtype='<f4',offset=128,bytes=32*130560*4,sha256='a'*64)
        reader.validate_frame(row,'prefill',128,32,'logits',64,32,130560,128)
        for key,value in [('chunk',True),('case','prefix512'),('kind','corrected_last'),('row_start',63),
            ('shape',[True,130560]),('offset',0),('bytes',1),('dtype','<f2'),('sha256','x'*64),('extra',True)]:
            with self.assertRaises(ValueError):reader.validate_frame(dict(row,**{key:value}),'prefill',128,32,'logits',64,32,130560,128)

    def test_reference_layout_inventory_flags_and_holes_are_checked_before_raw_files(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);value,contract,old,quality,tokens=source_fixture(folder)
            def check(v):
                raw=json.dumps(v).encode();(folder/'manifest.json').write_bytes(raw)
                return reader.load_reference(folder,hashlib.sha256(raw).hexdigest(),contract,old,quality,tokens)
            self.assertEqual(check(value),value)
            for change in [dict(completed=1),dict(quality_scored=True),dict(gates_recomputed_or_relaxed=True)]:
                with self.assertRaises(ValueError):check(dict(value,**change))
            for field,got in [('offset',True),('offset',1),('bytes',1),('dtype','<f8'),('shape',[128,1])]:
                v=copy.deepcopy(value);v['arrays']['fp32.prefix128.embedding'][field]=got
                with self.assertRaises(ValueError):check(v)
            v=copy.deepcopy(value);del v['arrays']['fp64.prefix1025.logits']
            with self.assertRaises(ValueError):check(v)

    def test_consumed_regular_binary_descriptor_and_path_are_rehashed(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'raw';p.write_bytes(b'abcd');pin=dict(bytes=4,sha256=hashlib.sha256(b'abcd').hexdigest())
            raw=reader.Raw(p,pin,4)
            try:
                self.assertEqual(raw.read(1,2),b'bc')
                with self.assertRaises(ValueError):raw.read(3,2)
                p.write_bytes(b'zzzz')
                with self.assertRaises(ValueError):raw.verify()
            finally:raw.close()
            link=Path(d)/'link';link.symlink_to(p)
            with self.assertRaises(OSError):reader.Raw(link,pin,4)

    def test_journal_truncation_and_coverage_caps_reject(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'events';p.write_bytes(b'{"a":1}\n')
            self.assertEqual(list(reader.lines(p,1)),[{'a':1}])
            p.write_bytes(b'{"a":1}')
            with self.assertRaises(ValueError):list(reader.lines(p,1))
            p.write_bytes(b'{}\n{}\n')
            with self.assertRaises(ValueError):list(reader.lines(p,1))
            p.write_bytes(b' '*65537)
            with self.assertRaises(ValueError):list(reader.lines(p,1))

    def test_owner_closure_limits_and_frozen_inputs_are_not_assumed(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'source';p.write_bytes(b'fixture')
            freeze=dict(source_sha256={str(p):hashlib.sha256(b'fixture').hexdigest()},
                guard_bytes=6*1024**3,disk_floor_bytes=2*1024**3)
            owner=dict(passed=True,owned_group_absent=True,source_inputs_unchanged=True,exit_code=0,failure=None,
                native_capture_completed=True,
                guard_bytes=6*1024**3,quality_scored=False,model_adoption_qualified=False,source_files=1,
                sample_errors=[],final_physical_peak_available=False,started_ns=1,completed_ns=2,
                minimum_observed_free_bytes=3*1024**3,final_wait4_peak_rss_bytes=1,
                max_sampled_aggregate_rss_bytes=1,max_sampled_aggregate_physical_bytes=1,
                max_sampled_aggregate_lifetime_physical_bytes=1)
            reader.validate_owner(owner,freeze)
            for change in [dict(passed=1),dict(exit_code=False),dict(owned_group_absent=False),dict(started_ns=True),
                dict(sample_errors=['missing']),dict(native_capture_completed=False),dict(minimum_observed_free_bytes=0),dict(final_wait4_peak_rss_bytes=7*1024**3)]:
                with self.assertRaises(ValueError):reader.validate_owner(dict(owner,**change),freeze)
            p.write_bytes(b'changed')
            with self.assertRaises(ValueError):reader.validate_owner(owner,freeze)

    def test_actual_samples_must_reconcile_with_pid_lifetime_and_receipt(self):
        record=dict(pid=42,started_ns=10,command=['fixture'])
        freeze=dict(command=['fixture'])
        samples=[dict(observed_ns=t,processes=[dict(pid=42,resident_size=r,phys_footprint=r-1,
                    lifetime_max_phys_footprint=r)]) for t,r in [(11,5),(12,9)]]
        receipt=dict(started_ns=10,completed_ns=20,sampled_observations=2,final_wait4_peak_rss_bytes=10,
            max_sampled_aggregate_rss_bytes=9,max_sampled_aggregate_physical_bytes=8,
            max_sampled_aggregate_lifetime_physical_bytes=9)
        reader.validate_samples(receipt,record,samples,freeze)
        mutations=[]
        for key,value in [('pid',True),('pid',43),('resident_size',True),('resident_size',11),
                          ('phys_footprint',12),('lifetime_max_phys_footprint',7)]:
            got=copy.deepcopy(samples);got[-1]['processes'][0][key]=value;mutations.append(got)
        for time in [10,11,20,True]:
            got=copy.deepcopy(samples);got[-1]['observed_ns']=time;mutations.append(got)
        mutations.extend([samples[:1],list(reversed(samples))])
        for got in mutations:
            with self.assertRaises(ValueError):reader.validate_samples(receipt,record,got,freeze)
        with self.assertRaises(ValueError):reader.validate_samples(receipt,dict(record,command=['other']),samples,freeze)
        with self.assertRaises(ValueError):reader.validate_samples(dict(receipt,final_wait4_peak_rss_bytes=8),record,samples,freeze)

    def test_consumed_journal_rechecks_its_descriptor_and_path(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'events';p.write_bytes(b'{}\n{}\n')
            pin=dict(bytes=6,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
            self.assertEqual(list(reader.lines(p,2,pin)),[{},{}])
            rows=reader.lines(p,2,pin)
            try:
                self.assertEqual(next(rows),{})
                p.write_bytes(b'{}\n[]\n')
                with self.assertRaises(ValueError):list(rows)
            finally:rows.close()

    def test_evidence_requires_external_digest_and_exact_roles(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'evidence';one=Path(d)/'one';one.write_bytes(b'{}')
            pin=dict(path=str(one),bytes=2,sha256=hashlib.sha256(b'{}').hexdigest())
            names=['native_receipt','native_owner','native_samples','native_freeze','native_capture',
                   'reference_receipt','reference_owner','reference_samples','reference_freeze']
            value=dict(format='toklet-minicpm5-context1025-evidence-v1',files={k:pin for k in names})
            p.write_text(json.dumps(value));digest=hashlib.sha256(p.read_bytes()).hexdigest()
            self.assertEqual(set(reader.evidence(p,digest)[1]),set(names))
            with self.assertRaises(ValueError):reader.evidence(p,'a'*64)
            one.write_bytes(b'[]')
            with self.assertRaises(ValueError):reader.evidence(p,digest)

    def test_fixed_arm64_memory_components_cannot_be_zeroed_and_rebalanced(self):
        contract=fixtures.fixture()[0]
        for chunk,scratch,metadata,total in [(1,2851694,1464,2789313128),(32,83009386,5184,2869474540)]:
            expected=reader.expected_memory(chunk,contract['columns'])
            self.assertEqual(expected,dict(weights_bytes=2674218150,kv_bytes=44084376,scratch_bytes=scratch,
                history_bytes=4100,logits_bytes=1044480,metadata_bytes=metadata,
                metadata_allowance_bytes=67108864,admitted_bytes=total))
        with self.assertRaises(ValueError):reader.expected_memory(True,contract['columns'])

    def test_serial_load_host_gpu_spans_are_disjoint(self):
        cap=dict(load_wall_ns=100,elapsed_ns=1000)
        reader.serial_spans(cap,110,900,500,1.,1.0000005)
        for args in [(99,900,500,1.,1.0000005),(110,1001,950,1.,1.0000005),
                     (110,900,500,1.,1.000002)]:
            with self.assertRaises(ValueError):reader.serial_spans(cap,*args)


if __name__=='__main__':unittest.main()
