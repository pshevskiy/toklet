import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import check_llama3_original_session as checker
import llama3_original_short as producer

ROOT=producer.graph.ROOT
CONTRACT=ROOT/'benchmarks/llama3-original-incremental-v1/manifest.json'
SHA=producer.sha(CONTRACT)


class IncrementalReplayTests(unittest.TestCase):
    def test_saved_rows_margins_state_flags_and_probe_inventory(self):
        policy=json.loads(CONTRACT.read_bytes())
        _,ids=checker.inputs.load(ROOT/'benchmarks/llama3-original-quality-v1',policy['quality_manifest_sha256'])
        with tempfile.TemporaryDirectory() as name:
            folder=Path(name);cases=[];row=np.zeros(128256,dtype=np.float32);raw=row.tobytes();digest=hashlib.sha256(raw).hexdigest()
            for case in policy['cases']:
                for n in range(case['prefill_tokens'],case['continuation_end']+1):
                    record=dict(prefix_tokens=n,logit_row=n-1,target_position=n,target_id=ids[n],
                                consistency=producer.comparison(row,row,ids[n]),greedy_matches=True,arrays={})
                    for label in ['batch','prefix']:
                        filename=f'prefix-{n}-{label}.f32le';(folder/filename).write_bytes(raw)
                        record['arrays'][label]=dict(file=filename,bytes=len(raw),sha256=digest)
                    cases.append(record)
            report=dict(format='toklet-llama3-original-incremental-source-v1',passed=True,contract_sha256=SHA,
                        full_quality_qualified=False,cases=cases,kv_bytes=37093376,peak_rss_bytes=1000,
                        overflow_preserves_prefix=True,reset_matches_fresh=True,all_tied_lowest_id=0,
                        decode_probe=dict(warmups=3,repetitions=5,weight_cache=False,head_tile=256,publish_throughput_claim=False,
                                          samples=[dict(repetition=r,context_before=n,elapsed_ns=1)for r in range(5)for n in range(17,21)]))
            path=folder/'source.json'
            def check():
                path.write_text(json.dumps(report));return checker.check(folder,CONTRACT,SHA)
            self.assertEqual(check()['rows'],22)
            report['cases'][0]['consistency']['prefix_greedy_margin']=1.
            with self.assertRaises(ValueError):check()
            report['cases'][0]['consistency']['prefix_greedy_margin']=0.
            report['overflow_preserves_prefix']='yes'
            with self.assertRaises(ValueError):check()
            report['overflow_preserves_prefix']=True
            report['decode_probe']['samples'][-1]['context_before']=999
            with self.assertRaises(ValueError):check()
