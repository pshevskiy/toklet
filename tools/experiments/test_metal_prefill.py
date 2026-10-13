"""Paired prefill protocol/timing and bounded failure cleanup, without a GPU."""
import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock,patch

import metal_prefill as mp


def fixture():
    memory={key:10 for key in mp.continuation.FIELDS};memory["admitted_bytes"]=sum(memory.values())
    ready={"event":"ready","format":"toklet-metal-q4-prefill-v1","projection_modes":["Original","Q4Tiled"],
           "debug_assertions":False,"warmups":3,"samples":10,"prompt_tokens":512,"output_tokens":1,"capacity":1024,"chunk":128,
           "original_memory":memory,"tiled_memory":copy.deepcopy(memory),"application_bytes":64*1024**2,
           "combined_admitted_bytes":memory["admitted_bytes"]*2-memory["weights_bytes"]+64*1024**2,
           "numerical_gate":{"atol":1e-4,"rtol":1e-4},"q4_execution_widths":[32,32]}
    events=[]
    for i in range(13):
        report={"reused_tokens":0,"prefill_tokens":512,"sampled_tokens":1,"emitted_tokens":1,"history_tokens":513,
                "evaluated_tokens":512,"pending_tokens":1,"reason":{"kind":"output_limit"},"decode_seconds":0}
        original=dict(report,first_token_seconds=2+i/1000,prefill_seconds=1.9,total_seconds=2.1)
        tiled=dict(report,first_token_seconds=.5+i/1000,prefill_seconds=.4,total_seconds=.6)
        events.append({"event":"sample","run":i,"order":"original_first" if i%2==0 else "tiled_first",
                       "original_report":original,"tiled_report":tiled,"greedy_id":11,"max_logit_delta":0,
                       "original_memory":memory,"tiled_memory":copy.deepcopy(memory)})
    return ready,events


class Prefill(unittest.TestCase):
    def test_complete_pairs_exclude_three_warmups_and_keep_dispersion(self):
        ready,events=fixture();mp.ready(ready)
        for i,event in enumerate(events):mp.sample(event,i,ready)
        result=mp.analyze(events)
        self.assertEqual(result["original"]["raw_seconds"],[2+i/1000 for i in range(3,13)])
        self.assertAlmostEqual(result["original"]["median_seconds"],2.0075)
        self.assertAlmostEqual(result["tiled"]["median_absolute_deviation_seconds"],.0025)
        self.assertAlmostEqual(result["improvement_fraction"],1-.5075/2.0075)
        with self.assertRaises(ValueError):mp.analyze(events[:-1])
    def test_mode_width_policy_and_shared_admission_mutations_fail(self):
        ready,_=fixture()
        for key,value in (("projection_modes",["Original","Original"]),("q4_execution_widths",[True,32]),
                          ("q4_execution_widths",[0,32]),("output_tokens",128),("chunk",True),
                          ("numerical_gate",{"atol":1e-3,"rtol":1e-3}),("combined_admitted_bytes",mp.LIMIT+1)):
            changed=copy.deepcopy(ready);changed[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):mp.ready(changed)
    def test_request_order_finite_timing_and_static_buffers_are_checked(self):
        ready,events=fixture()
        for key,value in (("run",True),("order","tiled_first"),("greedy_id",False),("max_logit_delta",math.inf)):
            changed=copy.deepcopy(events[0]);changed[key]=value
            with self.assertRaises(ValueError):mp.sample(changed,0,ready)
        for key,value in (("prefill_tokens",511),("pending_tokens",0),("first_token_seconds",math.nan),
                          ("first_token_seconds",3),("prefill_seconds",3),("decode_seconds",1)):
            changed=copy.deepcopy(events[0]);changed["tiled_report"][key]=value
            with self.assertRaises(ValueError):mp.sample(changed,0,ready)
        changed=copy.deepcopy(events[0]);changed["tiled_memory"]["scratch_bytes"]+=1
        changed["tiled_memory"]["admitted_bytes"]+=1
        with self.assertRaises(ValueError):mp.sample(changed,0,ready)
    def test_owned_worker_failure_is_reaped_original_error_retained(self):
        real=subprocess.Popen;children=[]
        def launch(*args,**kwargs):
            self.assertTrue(kwargs["start_new_session"])
            child=real([sys.executable,"-c",'print(\'{"event":"bad"}\',flush=True);import time;time.sleep(60)'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
            child.wait=Mock(wraps=child.wait);children.append(child);return child
        def digest(path):
            if path.name=="model.safetensors":return mp.SOURCE_SHA
            if path.name=="config.json":return mp.CONFIG_SHA
            return "test"
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);(path/"src").mkdir()
            output=path/"record.json"
            endpoints=Mock(side_effect=[{"battery":"test"},OSError("endpoint failure")])
            with patch.object(mp,"ROOT",path),patch.object(mp,"digest",digest),patch.object(mp,"observation",return_value="test"),patch.object(mp,"endpoints",endpoints),patch.object(mp.subprocess,"check_output",return_value="test"),patch.object(mp.subprocess,"Popen",launch):
                with self.assertRaisesRegex(ValueError,"admission/mode"):mp.run(path/"binary",output)
            result=json.loads(output.read_text())
            self.assertFalse(result["passed"]);self.assertIn("admission/mode",result["error"])
            self.assertTrue(any("endpoint failure" in error for error in result["cleanup_errors"]))
            children[0].wait.assert_called_once_with(timeout=5);self.assertIsNotNone(children[0].returncode)
            with patch.object(mp.subprocess,"Popen") as launch:
                with self.assertRaisesRegex(ValueError,"destination exists"):mp.run(path/"binary",output)
                launch.assert_not_called()


if __name__=="__main__":unittest.main()
