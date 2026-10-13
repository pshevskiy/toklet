"""Loss labels, all-row boundaries and owned failure cleanup; no Apple GPU."""
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import metal_quality as mq


def fixture(tokens,chunk=128):
    sha=hashlib.sha256(struct_bytes(tokens)).hexdigest()
    memory={key:10 for key in mq.continuation.FIELDS};memory["admitted_bytes"]=sum(memory.values())
    ready={"event":"ready","format":"toklet-metal-held-out-loss-events-v1","backend":"metal","quantized":True,
           "debug_assertions":False,"target_tokens":len(tokens)-1,"corpus_sha256":sha,"context_policy":mq.POLICY,
           "block_targets":1024,"context_capacity":1025,"prefill_chunk_size":chunk,"logit_positions":"all_causal_rows",
           "weight_dtype":"Q4 v1 + FP16 exclusions","kv_dtype":"FP16","memory":memory,"application_allowance_bytes":64*1024**2,
           "memory_admitted_bytes":memory["admitted_bytes"]+64*1024**2}
    rows=[{"event":"loss","target_index":i,"block_index":i//1024,"position":i%1024,"prefix_tokens":i%1024+1,
           "target_token":t,"negative_log_likelihood":1.0+i%7/16,"target_logit_f32_bits":0,"logits_sha256":"0"*64} for i,t in enumerate(tokens[1:])]
    chunks=[];blocks=[]
    for start in range(0,len(rows),1024):
        end=min(start+1024,len(rows));size=end-start
        for previous in range(0,size+1,chunk):
            n=min(chunk,size+1-previous)
            chunks.append({"event":"chunk_done","block_index":start//1024,"rows":n,"evaluated_tokens":previous+n,"scored_targets":min(previous+n,size),"pending_tokens":0})
        blocks.append({"event":"block_done","block_index":start//1024,"target_start":start,"target_end":end,"scored_targets":size,
                       "evaluated_tokens":size+1,"pending_tokens":0,"total_negative_log_likelihood":math.fsum(r["negative_log_likelihood"] for r in rows[start:end])})
    total=math.fsum(r["negative_log_likelihood"] for r in rows)
    done={"event":"done","scored_targets":len(rows),"evaluated_tokens":blocks[-1]["evaluated_tokens"],"pending_tokens":0,
          "total_negative_log_likelihood":total,"mean_negative_log_likelihood":total/len(rows)}
    return ready,rows,chunks,blocks,done,sha


def struct_bytes(tokens):
    import struct
    return struct.pack(f"<{len(tokens)}I",*tokens)


class MetalLoss(unittest.TestCase):
    def test_exact_bounded_parsed_bytes_are_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"payload";path.write_bytes(b"original")
            expected=hashlib.sha256(b"original").hexdigest()
            self.assertEqual(mq.frozen_bytes(path,8,expected),b"original")
            with self.assertRaises(ValueError):mq.frozen_bytes(path,7,expected)
            path.write_bytes(b"replaced")
            with self.assertRaises(ValueError):mq.frozen_bytes(path,8,expected)
    def test_all_required_positions_chunk_tails_and_block_boundaries(self):
        for count in (1,127,128,129,1024,1025,10000):
            tokens=[i%127 for i in range(count+1)]
            for chunk in (1,128):
                admission,rows,chunks,blocks,done,sha=fixture(tokens,chunk)
                mq.ready(admission,tokens,sha,True,chunk)
                for i,row in enumerate(rows):mq.loss_row(row,i,tokens)
                committed={}
                for event in chunks:
                    block=event["block_index"];previous=committed.get(block,0)
                    committed[block]=mq.chunk_done(event,block,previous,count,chunk,block*1024+event["scored_targets"])
                for i,event in enumerate(blocks):mq.block_done(event,i,tokens,rows[:event["target_end"]],committed[i])
                self.assertEqual(mq.done(done,tokens,rows,len(blocks),committed[len(blocks)-1]),done["mean_negative_log_likelihood"])
        # In a batch the logical causal prefix1 differs from physical128;
        # both are independently checked, never conflated in target labels.
        tokens=list(range(130));_,rows,chunks,_,_,_=fixture(tokens)
        self.assertEqual(rows[0]["prefix_tokens"],1)
        self.assertEqual(chunks[0]["evaluated_tokens"],128)

    def test_readiness_precision_policy_and_admission_fail_closed(self):
        tokens=[1,2,3];admission,*_,sha=fixture(tokens)
        for key,value in (("backend","cpu"),("quantized",1),("context_policy","sliding"),("prefill_chunk_size",1),
                          ("weight_dtype","FP16"),("logit_positions","last"),("corpus_sha256","0"*64),("memory_admitted_bytes",mq.LIMIT+1)):
            altered=copy.deepcopy(admission);altered[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):mq.ready(altered,tokens,sha,True,128)
        altered=copy.deepcopy(admission);altered["memory"]["weights_bytes"]=True
        with self.assertRaises(ValueError):mq.ready(altered,tokens,sha,True,128)

    def test_target_chunk_and_aggregation_mutations_are_rejected(self):
        tokens=list(range(130));_,rows,chunks,blocks,finish,_=fixture(tokens)
        for key,value in (("target_token",3),("position",128),("prefix_tokens",128),("negative_log_likelihood",math.nan),
                          ("target_logit_f32_bits",0x7f800000),("logits_sha256","g"*64),("target_index",False)):
            altered=copy.deepcopy(rows[0]);altered[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):mq.loss_row(altered,0,tokens)
        for key,value in (("rows",127),("evaluated_tokens",127),("scored_targets",127),("pending_tokens",1),("block_index",1)):
            altered=copy.deepcopy(chunks[0]);altered[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):mq.chunk_done(altered,0,0,129,128,128)
        with self.assertRaises(ValueError):mq.chunk_done(chunks[0],0,0,129,128,127)
        for key,value in (("scored_targets",128),("evaluated_tokens",129),("pending_tokens",True),("total_negative_log_likelihood",0)):
            altered=copy.deepcopy(blocks[0]);altered[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):mq.block_done(altered,0,tokens,rows,130)
        altered=copy.deepcopy(finish);altered["mean_negative_log_likelihood"]+=.01
        with self.assertRaises(ValueError):mq.done(altered,tokens,rows,1,130)
        with self.assertRaises(ValueError):mq.done(finish,tokens,rows[:-1],1,130)

    def test_failed_owned_worker_is_explicitly_reaped(self):
        real=subprocess.Popen;children=[]
        def launch(*args,**kwargs):
            self.assertTrue(kwargs["start_new_session"])
            child=real([sys.executable,"-c",'print(\'{"event":"unexpected"}\',flush=True);import time;time.sleep(60)'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
            child.wait=Mock(wraps=child.wait);children.append(child);return child
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            token_path=path/"tokens.u32le";token_path.write_bytes(struct_bytes([1,2]))
            with (path/"raw").open("x") as raw,(path/"stderr").open("xb") as stderr,patch.object(mq.subprocess,"Popen",launch):
                result=mq.child(Path("/tmp/fake"),path,path,token_path,[1,2],True,128,raw,stderr)
        self.assertFalse(result["passed"]);self.assertIn("admission/policy",result["error"])
        self.assertEqual(result["cleanup_errors"],[])
        children[0].wait.assert_called_once_with(timeout=5);self.assertIsNotNone(children[0].returncode)

    def test_outer_failure_retains_record_and_endpoint_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);(path/"models/Qwen2.5-0.5B-Instruct").mkdir(parents=True);(path/"src").mkdir()
            corpus=path/"benchmarks/contract-v1";corpus.mkdir(parents=True);(corpus/"held_out_loss.u32le").write_bytes(struct_bytes([1]*10001))
            (corpus/"manifest.json").write_text(json.dumps({"quality":{"mean_nll_increase_limit_nats_per_token":.05}}))
            baseline=path/"benchmarks/results";baseline.mkdir();(baseline/"2026-10-11-held-out-scalar-loss-analysis-v1.json").write_text(json.dumps({"context_policy":mq.POLICY,"scored_targets":10000}))
            def digest(p):
                if p.name=="model.safetensors":return mq.SOURCE_SHA
                if p.name=="held_out_loss.u32le":return mq.CORPUS_SHA
                if p.name.endswith("scalar-loss-analysis-v1.json"):return mq.BASELINE_SHA
                return "test"
            power=Mock(side_effect=[{"battery":"test"},OSError("endpoint failure")])
            output=path/"result.json"
            corpus_sha=hashlib.sha256((corpus/"held_out_loss.u32le").read_bytes()).hexdigest()
            baseline_sha=hashlib.sha256((baseline/"2026-10-11-held-out-scalar-loss-analysis-v1.json").read_bytes()).hexdigest()
            with patch.object(mq,"ROOT",path),patch.object(mq,"CORPUS_SHA",corpus_sha),patch.object(mq,"BASELINE_SHA",baseline_sha),patch.object(mq,"digest",digest),patch.object(mq.continuation,"observation",return_value="test"),patch.object(mq.continuation,"endpoints",power),patch.object(mq.quality,"probe_inputs",side_effect=ValueError("original probe failure")):
                with self.assertRaisesRegex(ValueError,"collection failed"):mq.run(path/"binary",output)
            record=json.loads(output.read_text());self.assertFalse(record["passed"])
            self.assertIn("original probe failure",record["error"]);self.assertEqual(record["endpoint_errors"],["endpoint failure"])
            self.assertTrue(record["identities_unchanged"])
            with patch.object(mq.subprocess,"Popen") as launch:
                with self.assertRaisesRegex(ValueError,"destination exists"):mq.run(path/"binary",output)
                launch.assert_not_called()


if __name__=="__main__":unittest.main()
