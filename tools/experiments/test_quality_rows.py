"""CPU reference protocol and cleanup; no model or Apple GPU required."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import quality_rows as qr
from test_metal_quality import fixture, struct_bytes


def admission(tokens,larger,chunk):
    event,*_=fixture(tokens,chunk)
    event.pop("quantized")
    event.update(format="toklet-cpu-row-loss-events-v1",backend="cpu",cpu_projection="tiled",
                 weight_dtype="FP32",kv_dtype="FP32",hidden_size=1536 if larger else 896,
                 num_layers=28 if larger else 24,vocab_size=151936,
                 external_logits_bytes=chunk*151936*4,serialization_bytes=151936*4)
    event["memory"]["weights_bytes"]=6174857216 if larger else 1976131072
    event["memory"]["kv_bytes"]=2*(28 if larger else 24)*2*1025*(128 if larger else 64)*4
    event["memory"]["admitted_bytes"]=sum(event["memory"][k] for k in qr.continuation.FIELDS)
    event["memory_admitted_bytes"]=event["memory"]["admitted_bytes"]+64*1024**2
    return event


class CpuRows(unittest.TestCase):
    def test_matching_geometry_precision_and_bounded_output(self):
        tokens=[1,2,3]
        sha=hashlib.sha256(struct_bytes(tokens)).hexdigest()
        for larger in (False,True):
            for chunk in (1,8):
                event=admission(tokens,larger,chunk)
                qr.ready(event,tokens,sha,larger,chunk)
                for key,value in (("backend","metal"),("weight_dtype","FP16"),("hidden_size",32),
                                  ("cpu_projection","scalar"),("external_logits_bytes",True),
                                  ("serialization_bytes",0),("memory_admitted_bytes",qr.LIMIT+1),
                                  ("corpus_sha256","0"*64),("debug_assertions",True)):
                    altered=copy.deepcopy(event);altered[key]=value
                    with self.subTest(key=key),self.assertRaises(ValueError):qr.ready(altered,tokens,sha,larger,chunk)
                altered=copy.deepcopy(event);altered["memory"]["admitted_bytes"]=float(altered["memory"]["admitted_bytes"])
                with self.assertRaises(ValueError):qr.ready(altered,tokens,sha,larger,chunk)

    def test_every_target_and_tail_uses_the_frozen_block_policy(self):
        for count in (1,7,8,9,1024,1025,10000):
            tokens=[i%127 for i in range(count+1)]
            for chunk in (1,8):
                _,rows,chunks,blocks,done,_=fixture(tokens,chunk)
                committed={}
                for i,row in enumerate(rows):qr.loss_row(row,i,tokens)
                for event in chunks:
                    block=event["block_index"]
                    committed[block]=qr.chunk_done(event,block,committed.get(block,0),count,chunk,block*1024+event["scored_targets"])
                for i,event in enumerate(blocks):qr.block_done(event,i,tokens,rows[:event["target_end"]],committed[i])
                self.assertEqual(qr.done(done,tokens,rows,len(blocks),committed[len(blocks)-1]),done["mean_negative_log_likelihood"])

    def test_bad_owned_worker_is_killed_reaped_and_original_error_retained(self):
        real=subprocess.Popen;children=[]
        def launch(*args,**kwargs):
            self.assertTrue(kwargs["start_new_session"])
            child=real([sys.executable,"-c",'print(\'{"event":"unexpected"}\',flush=True);import time;time.sleep(60)'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
            child.wait=Mock(wraps=child.wait);children.append(child);return child
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            with (path/"raw").open("x") as raw,(path/"stderr").open("xb") as stderr,patch.object(qr.subprocess,"Popen",launch):
                result=qr.child(path/"binary",path,path/"tokens",[1,2],True,8,raw,stderr)
        self.assertFalse(result["passed"]);self.assertIn("admission/policy",result["error"])
        self.assertEqual(result["cleanup_errors"],[])
        children[0].wait.assert_called_once_with(timeout=5);self.assertIsNotNone(children[0].returncode)

    def test_outer_failure_keeps_original_error_and_endpoint_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);(path/"src").mkdir()
            corpus=path/"benchmarks/contract-v1";corpus.mkdir(parents=True)
            (corpus/"held_out_loss.u32le").write_bytes(struct_bytes([1]*10001))
            digest=lambda p:qr.SOURCES[True] if p.name=="model.safetensors" else qr.TOKENIZER_SHA if p.name=="tokenizer.json" else qr.CONFIGS[True] if p.name=="config.json" else qr.shared.CORPUS_SHA if p.name=="held_out_loss.u32le" else "test"
            output=path/"record.json"
            power=Mock(side_effect=[{"battery":"test"},OSError("endpoint failure")])
            with patch.object(qr,"ROOT",path),patch.object(qr,"digest",digest),patch.object(qr.shared,"frozen_bytes",return_value=struct_bytes([1]*10001)),patch.object(qr.continuation,"observation",return_value="test"),patch.object(qr.continuation,"endpoints",power),patch.object(qr.shared.quality,"probe_inputs",side_effect=ValueError("original probe failure")):
                with self.assertRaisesRegex(ValueError,"collection failed"):qr.run(path/"binary",output)
            record=json.loads(output.read_text())
            self.assertFalse(record["passed"]);self.assertFalse(record["quality_completed"])
            self.assertIn("original probe failure",record["error"]);self.assertEqual(record["endpoint_errors"],["endpoint failure"])
            with patch.object(qr.subprocess,"Popen") as launch:
                with self.assertRaisesRegex(ValueError,"destination exists"):qr.run(path/"binary",output)
                launch.assert_not_called()


if __name__=="__main__":unittest.main()
