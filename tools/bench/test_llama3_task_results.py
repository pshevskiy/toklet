import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import llama3_task_results as subject

INPUTS=subject.inputs.ROOT/'benchmarks/llama3-quality-tasks-v1'
SHA=hashlib.sha256((INPUTS/'manifest.json').read_bytes()).hexdigest()


class ResultReplayTests(unittest.TestCase):
    def write(self,folder,role,record):
        manifest,_=subject.inputs.load(INPUTS,SHA)
        body=(json.dumps(record)+'\n').encode();(folder/'tasks.jsonl').write_bytes(body)
        report=dict(format=f'toklet-llama3-quality-tasks-{role}-v1',completed=True,mode='authored-probe',manifest_sha256=SHA,cases=1,full_quality_qualified=False,tasks_sha256=hashlib.sha256(body).hexdigest())
        if role=='native':report.update(repository=manifest['models']['native']['repository'],revision=manifest['models']['native']['revision'],admitted_bytes_including_app=1000,profile=dict(projection='MlxAffine',normalization='Compensated',attention='Reference',readback='FullLogits',chunk=32,capacity=566))
        else:report.update(model=manifest['models']['original'],all_original_trajectories_verified=True,peak_rss_bytes=1000)
        (folder/'result.json').write_text(json.dumps(report))
    def test_native_pending_limit_and_hidden_stop_rejections(self):
        record=dict(id='authored-probe',category='probe',prompt_tokens=4,output_reserve=4,visible_token_ids=[7,8,9,10],stop_reason=dict(kind='output_limit'),generation=dict(reason=dict(kind='output_limit'),reused_tokens=0,prefill_tokens=4,sampled_tokens=4,emitted_tokens=4,history_tokens=8,evaluated_tokens=7,pending_tokens=1))
        with tempfile.TemporaryDirectory()as name:
            folder=Path(name);self.write(folder,'native',record)
            self.assertEqual(len(subject.collection(folder,INPUTS,SHA,'native',True)[1]),1)
            for change in ['counter','stop_visible']:
                wrong=copy.deepcopy(record)
                if change=='counter':wrong['generation']['sampled_tokens']=3
                else:wrong['visible_token_ids'][0]=128009
                self.write(folder,'native',wrong)
                with self.assertRaises(ValueError):subject.collection(folder,INPUTS,SHA,'native',True)
    def test_original_prediction_records_must_match_even_if_verified_flag_is_true(self):
        stop=dict(kind='stop_token',token_id=128009)
        samples=[dict(index=i,token_id=t,margin=1.,row_sha256='0'*64)for i,t in enumerate([7,128009])]
        rows=[dict(index=i,absolute_row=3+i,expected_id=t,stateless_id=t,incremental_margin=1.,stateless_margin=1.)for i,t in enumerate([7,128009])]
        record=dict(id='authored-probe',category='probe',prompt_tokens=4,output_reserve=4,visible_token_ids=[7],stop_reason=stop,
                    generation=dict(visible_token_ids=[7],stop_reason=stop,committed_tokens=5,history_tokens=6,pending_tokens=1,samples=samples),
                    trajectory=dict(verified=True,mismatches=[],visible_predictions=1,terminal_stop_checked=True,rows=rows))
        with tempfile.TemporaryDirectory()as name:
            folder=Path(name);self.write(folder,'original',record);subject.collection(folder,INPUTS,SHA,'original',True)
            record['trajectory']['rows'][0]['stateless_id']=8;self.write(folder,'original',record)
            with self.assertRaises(ValueError):subject.collection(folder,INPUTS,SHA,'original',True)
