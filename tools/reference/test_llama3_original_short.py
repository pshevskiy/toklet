import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import llama3_original_short as producer
import check_llama3_original_short as checker

ROOT = Path(__file__).resolve().parents[2]
INPUTS = ROOT/'benchmarks/llama3-original-short-v1'
SHA = hashlib.sha256((INPUTS/'manifest.json').read_bytes()).hexdigest()


class ShortTests(unittest.TestCase):
    def test_frozen_positions_and_separate_consistency_gates(self):
        manifest, ids = producer.inputs(INPUTS, SHA)
        self.assertEqual([c['target_id'] for c in manifest['cases']], [ids[n] for n in [1,4,8,17]])
        prefix = np.zeros(3, dtype=np.float32)
        self.assertTrue(producer.comparison(prefix+np.float32(.0009),prefix,0)['passed'])
        shifted = prefix.copy(); shifted[0] = np.float32(.0005)
        self.assertFalse(producer.comparison(shifted,prefix,0)['passed'])
        self.assertFalse(producer.comparison(prefix+np.float32(.002),prefix,0)['passed'])
        with self.assertRaises(ValueError):
            producer.comparison(np.array([np.nan],dtype=np.float32),np.zeros(1,dtype=np.float32),0)

    def test_saved_rows_replay_and_reject_corruption_or_retargeting(self):
        manifest, ids = producer.inputs(INPUTS, SHA)
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory); cases=[]
            row=np.zeros(128256,dtype=np.float32)
            for case in manifest['cases']:
                entry=dict(case,consistency=producer.comparison(row,row,case['target_id']),arrays={})
                for label in ['batch','prefix']:
                    raw=row.tobytes(); name=f"prefix-{case['prefix_tokens']}-{label}.f32le"
                    (folder/name).write_bytes(raw)
                    entry['arrays'][label]={'file':name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
                cases.append(entry)
            report={'format':'toklet-llama3-original-short-source-v1','full_quality_qualified':False,'passed':True,'cases':cases,'input_manifest_sha256':SHA}
            path=folder/'source.json';path.write_text(json.dumps(report))
            self.assertTrue(checker.check(folder,INPUTS,SHA)['passed'])
            import math
            native={'format':'toklet-llama3-original-short-native-v1','passed':True,'full_quality_qualified':False,'input_manifest_sha256':SHA,'cases':[
                {'prefix_tokens':c['prefix_tokens'],'target_position':c['target_position'],'target_id':c['target_id'],
                 'score':{'target':c['target_id'],'evaluated':c['prefix_tokens'],'vocabulary':128256,'greedy':0,'targetLogit':0.0,'greedyLogit':0.0,'negativeLogLikelihood':math.log(128256)}}for c in cases]}
            native_path=folder/'native.json';native_path.write_text(json.dumps(native))
            self.assertEqual(checker.compare_native(folder,INPUTS,SHA,native_path)['short_mean_nll_difference'],0)
            native['cases'][0]['score']['negativeLogLikelihood']=float('nan');native_path.write_text(json.dumps(native))
            with self.assertRaises(ValueError):checker.compare_native(folder,INPUTS,SHA,native_path)
            native['cases'][0]['score']['negativeLogLikelihood']=math.log(128256)
            native['cases'][0]['target_position']=2;native_path.write_text(json.dumps(native))
            with self.assertRaises(ValueError):checker.compare_native(folder,INPUTS,SHA,native_path)

            report['cases'][0]['consistency']['batch']['nll']=float('nan');path.write_text(json.dumps(report))
            with self.assertRaises(ValueError):checker.check(folder,INPUTS,SHA)
            report['cases'][0]['consistency']['batch']['nll']=producer.comparison(row,row,ids[1])['batch']['nll']
            original=report['cases'][0]['arrays']['prefix']['file']
            report['cases'][0]['arrays']['prefix']['file']=report['cases'][0]['arrays']['batch']['file'];path.write_text(json.dumps(report))
            with self.assertRaises(ValueError):checker.check(folder,INPUTS,SHA)
            report['cases'][0]['arrays']['prefix']['file']=original

            report['cases'][0]['target_id'] = (ids[1]+1)%128256;path.write_text(json.dumps(report))
            with self.assertRaises(ValueError):checker.check(folder,INPUTS,SHA)
            report['cases'][0]['target_id']=ids[1];path.write_text(json.dumps(report))
            (folder/cases[0]['arrays']['batch']['file']).write_bytes(b'x')
            with self.assertRaises(ValueError):checker.check(folder,INPUTS,SHA)

    def test_reader_failure_retains_nonpassing_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);model=root/'model';model.mkdir();(model/'acquisition.json').write_text('{}')
            with patch.object(producer.graph,'Reader',side_effect=ValueError('fixture reader failure')):
                with self.assertRaises(ValueError):producer.produce(model,INPUTS,SHA,root/'output')
            report=json.loads((root/'output/source.json').read_text())
            self.assertFalse(report['passed']);self.assertIn('fixture reader failure',report['error'])
