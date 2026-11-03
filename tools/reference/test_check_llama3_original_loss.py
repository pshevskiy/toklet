import copy
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest
import check_llama3_original_loss as subject

INPUTS=subject.inputs.ROOT/'benchmarks/llama3-original-quality-v1'
SHA=subject.inputs.parent.digest((INPUTS/'manifest.json').read_bytes())


class FullLossTests(unittest.TestCase):
    def records(self):
        manifest,ids=subject.inputs.load(INPUTS,SHA)
        rows=[]
        for b in manifest['blocks']:
            for i in range(*b['global_target_indices']):
                rows.append(dict(block=b['block'],local_row=i-b['input_start']-1,global_target_index=i,
                                 target=ids[i],target_fp32_bits=0,vocab=128256,greedy_lowest_id=0,
                                 maximum=0.,exp_mass=128256.,nll=math.log(128256),row_sha256='0'*64))
        return manifest,ids,rows

    def test_target_coverage_rejects_duplicate_missing_misaligned_or_nonfinite(self):
        manifest,ids,rows=self.records()
        self.assertEqual(len(subject.collect(rows,ids,manifest['blocks'])),10000)
        for changed in [rows[:-1],rows+[rows[0]]]:
            with self.assertRaises(ValueError):subject.collect(changed,ids,manifest['blocks'])
        for field,value in [('local_row',5000),('target',128256),('nll',float('nan')),('exp_mass',float('inf'))]:
            changed=copy.deepcopy(rows);changed[1024][field]=value
            with self.assertRaises(ValueError):subject.collect(changed,ids,manifest['blocks'])

    def test_alignment_uses_only_first_block_positions_and_pinned_reports(self):
        records=[[dict(global_target_index=i,nll=1.,numeric_row_sha256='0'*64)for i in [512,513,1024,1025]]for _ in range(2)]
        source=dict(format='toklet-llama3-original-long-source-v1',passed=True,cases=[
            dict(prefix_tokens=i,consistency=dict(prefix=dict(nll=999. if i==1025 else 1.)))for i in [512,513,1024,1025]])
        native=dict(format='toklet-llama3-quality-context-v1',ok=True,observations=[dict(kind='independent',prefix=i,numeric_sha256='0'*64)for i in [17,512,513,1024,1025]])
        with tempfile.TemporaryDirectory() as name:
            folder=Path(name);(folder/'source.json').write_text(json.dumps(source));(folder/'result.json').write_text(json.dumps(native))
            source_sha=subject.inputs.parent.digest((folder/'source.json').read_bytes())
            native_sha=subject.inputs.parent.digest((folder/'result.json').read_bytes())
            self.assertEqual(subject.alignment_crosscheck(records,folder,source_sha,folder,native_sha),[512,513,1024])
            records[0][0]['nll']+=.0002
            with self.assertRaises(ValueError):subject.alignment_crosscheck(records,folder,source_sha,folder,native_sha)
            records[0][0]['nll']=1.;records[1][0]['numeric_row_sha256']='1'*64
            with self.assertRaises(ValueError):subject.alignment_crosscheck(records,folder,source_sha,folder,native_sha)
            with self.assertRaises(ValueError):subject.alignment_crosscheck(records,folder,'0'*64,folder,native_sha)

    def test_pinned_reports_and_frozen_gate_without_model_execution(self):
        manifest,ids,rows=self.records()
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);source=root/'source';native=root/'native';source.mkdir();native.mkdir()
            def write(folder,report,records):
                raw=''.join(json.dumps(r,separators=(',',':'))+'\n'for r in records).encode()
                (folder/'targets.jsonl').write_bytes(raw)
                report['targets_sha256']=subject.inputs.parent.digest(raw)
                raw=json.dumps(report).encode();(folder/'result.json').write_bytes(raw)
                return subject.inputs.parent.digest(raw)
            shared=dict(completed=True,full_quality_qualified=False,manifest_sha256=SHA,targets=10000,mean_nll=math.log(128256))
            source_report=dict(shared,format='toklet-llama3-quality-loss-original-v1',
                               model=dict(repository=subject.original.META,revision=subject.original.META_REVISION,weight_sha256=subject.original.WEIGHT_SHA),
                               peak_rss_bytes=1000,blocks=[dict(block=b['block'],targets=b['score_rows'][1],input_positions=b['input_end']-b['input_start'])for b in manifest['blocks']])
            native_report=dict(shared,format='toklet-llama3-quality-loss-native-v1',repository=subject.inputs.parent.REPOSITORY,revision=subject.inputs.parent.REVISION,
                               admitted_bytes_including_app=1000,profile=dict(projection='MlxAffine',normalization='Compensated',attention='Reference',chunk=32,capacity=1025))
            source_sha=write(source,source_report,rows);native_sha=write(native,native_report,rows)
            result=subject.check(INPUTS,SHA,source,source_sha,native,native_sha)
            self.assertTrue(result['loss_gate_passed']);self.assertFalse(result['full_quality_qualified'])
            with self.assertRaises(ValueError):subject.check(INPUTS,SHA,source,'0'*64,native,native_sha)
            # A valid synthetic native distribution can fail the gate; retain a completed rejection.
            changed=copy.deepcopy(rows);target_value=struct.unpack('<f',struct.pack('<f',-0.1))[0]
            for row in changed:
                row['target_fp32_bits']=struct.unpack('<I',struct.pack('<f',target_value))[0]
                row['nll']=math.log(128256)-target_value
            native_report['mean_nll']=math.fsum(r['nll']for r in changed)/10000
            native_sha=write(native,native_report,changed)
            rejected=subject.check(INPUTS,SHA,source,source_sha,native,native_sha)
            self.assertTrue(rejected['completed']);self.assertFalse(rejected['loss_gate_passed'])
            (native/'targets.jsonl').write_bytes(b'changed')
            with self.assertRaises(ValueError):subject.check(INPUTS,SHA,source,source_sha,native,native_sha)
