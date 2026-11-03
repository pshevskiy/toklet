import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import llama3_original_long as producer
import check_llama3_original_long as checker

INPUTS = producer.graph.ROOT/'benchmarks/llama3-original-quality-v1'
SHA = producer.short.sha(INPUTS/'manifest.json')


class LongTests(unittest.TestCase):
    def test_eight_rows_and_native_scores_replay_with_fail_closed_alignment(self):
        _, ids = producer.inputs.load(INPUTS, SHA)
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory); cases=[]; row=np.zeros(128256,dtype=np.float32); raw=row.tobytes()
            digest=hashlib.sha256(raw).hexdigest()
            for n in producer.inputs.PREFIXES:
                case=dict(prefix_tokens=n,logit_row=n-1,target_position=n,target_id=ids[n],
                          consistency=producer.short.comparison(row,row,ids[n]),arrays={})
                for label in ['batch','prefix']:
                    name=f'prefix-{n}-{label}.f32le';(folder/name).write_bytes(raw)
                    case['arrays'][label]=dict(file=name,bytes=len(raw),sha256=digest)
                cases.append(case)
            report=dict(format='toklet-llama3-original-long-source-v1',passed=True,cases=cases,
                        full_quality_qualified=False,input_manifest_sha256=SHA,peak_rss_bytes=1000)
            source=folder/'source.json';source.write_text(json.dumps(report))
            self.assertTrue(checker.check(folder,INPUTS,SHA)['passed'])
            native=folder/'native';native.mkdir();observations=[]
            for n in [17,512,513,1024,1025]:
                (native/f'prefix-{n}.f32le').write_bytes(raw)
                observations.append(dict(kind='independent',prefix=n,raw_sha256=digest))
            native_report=dict(format='toklet-llama3-quality-context-v1',ok=True,capacity=1025,
                               full_quality_qualified=False,revision=producer.inputs.parent.REVISION,
                               observations=observations)
            path=native/'result.json';path.write_text(json.dumps(native_report))
            with self.assertRaises(ValueError): checker.check(folder,INPUTS,SHA,native,'0'*64)
            self.assertEqual(checker.check(folder,INPUTS,SHA,native,producer.short.sha(path))['long_mean_nll_difference'],0)
            native_report['observations'].append(copy.deepcopy(observations[0]));path.write_text(json.dumps(native_report))
            with self.assertRaises(ValueError): checker.check(folder,INPUTS,SHA,native,producer.short.sha(path))
            report['cases'][0]['target_position'] += 1;source.write_text(json.dumps(report))
            with self.assertRaises(ValueError): checker.check(folder,INPUTS,SHA)
            report['cases'][0]['target_position'] -= 1;source.write_text(json.dumps(report))
            (folder/cases[-1]['arrays']['prefix']['file']).write_bytes(b'x')
            with self.assertRaises(ValueError): checker.check(folder,INPUTS,SHA)

    def test_failed_model_open_retains_nonpassing_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); model=root/'model';model.mkdir();(model/'acquisition.json').write_text('{}')
            with patch.object(producer.graph,'Reader',side_effect=ValueError('fixture reader failure')):
                report=producer.produce(model,INPUTS,SHA,root/'output')
            self.assertFalse(report['passed']);self.assertIn('fixture reader failure',report['error'])
            self.assertFalse(json.loads((root/'output/source.json').read_text())['passed'])
