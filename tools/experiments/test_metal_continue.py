import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import metal_continue as mc


def admission():
    m={key:10 for key in mc.FIELDS};m["admitted_bytes"]=sum(m.values())
    return {"event":"ready","format":"toklet-metal-continuation-v1","quantized":True,
            "debug_assertions":False,"warmups":3,"samples":10,"cases":["changed128","tail_edit"],
            "capacity":1024,"chunk":128,"baseline_memory":m,"reuse_memory":copy.deepcopy(m),
            "application_bytes":64,"combined_admitted_bytes":194,"numerical_gate":{"atol":1e-4,"rtol":1e-4}}


def event(run,case):
    a=admission();keep=128 if case=="changed128" else 511
    result={"event":"sample","run":run,"case":case,"order":"reset_first" if (run+(case=="tail_edit"))%2==0 else "reuse_first",
            "greedy_id":1,"max_logit_delta":0}
    for mode,reused in [("reset",0),("reuse",keep)]:
        t=1.0 if mode=="reset" else .5
        result[f"{mode}_report"]={"reason":{"kind":"output_limit"},"reused_tokens":reused,"prefill_tokens":512-reused,
            "sampled_tokens":1,"emitted_tokens":1,"history_tokens":513,"evaluated_tokens":512,"pending_tokens":1,
            "first_token_seconds":t,"prefill_seconds":t,"total_seconds":t+.01,"decode_seconds":0}
        result[f"{mode}_memory"]=copy.deepcopy(a["baseline_memory"])
    return result


class Continuation(unittest.TestCase):
    def test_bounded_transport_time_policy(self):
        for request,lifetime in ((0,180),(181,180),(30,86401),(True,180),(float("nan"),180),(30,float("inf"))):
            with self.assertRaisesRegex(ValueError,"time limits"):
                mc.metal_engine.Transport(None,None,None,request_timeout=request,lifetime_timeout=lifetime)

    def test_declared_admission_memory_and_numerical_policy(self):
        a=admission();mc.ready(a,True)
        for field,value in [("debug_assertions",True),("quantized",False),("combined_admitted_bytes",195),("chunk",256),("numerical_gate",{"atol":1,"rtol":1})]:
            b=copy.deepcopy(a);b[field]=value
            with self.assertRaises(ValueError):mc.ready(b,True)
        b=copy.deepcopy(a);b["baseline_memory"]["weights_bytes"]=True
        with self.assertRaises(ValueError):mc.ready(b,True)

    def test_exact_coverage_state_order_timing_and_fixed_buffers(self):
        a=admission();e=event(0,"changed128");mc.sample(e,0,"changed128",a)
        for field,value in [("run",1),("order","reuse_first"),("case","tail_edit"),("greedy_id",151936),("max_logit_delta",float("nan"))]:
            b=copy.deepcopy(e);b[field]=value
            with self.assertRaises(ValueError):mc.sample(b,0,"changed128",a)
        for field,value in [("reused_tokens",0),("prefill_tokens",512),("first_token_seconds",float("nan")),("total_seconds",.1),("pending_tokens",0)]:
            b=copy.deepcopy(e);b["reuse_report"][field]=value
            with self.assertRaises(ValueError):mc.sample(b,0,"changed128",a)
        b=copy.deepcopy(e);b["reuse_memory"]["admitted_bytes"]+=1
        with self.assertRaises(ValueError):mc.sample(b,0,"changed128",a)

    def test_independent_analysis_excludes_warmups(self):
        events=[event(run,case) for run in range(13) for case in ("changed128","tail_edit")]
        for e in events:
            if e["run"]<3:e["reset_report"]["first_token_seconds"]=100
        result=mc.analyze(events)
        self.assertEqual([r["improvement_fraction"] for r in result],[.5,.5])
        self.assertEqual(result[0]["samples_seconds"]["reset"],[1.0]*10)
        with self.assertRaises(ValueError):mc.analyze(events[:-1])

    def test_failed_owned_worker_is_reaped_and_evidence_retained(self):
        self.failed_worker()

    def test_cleanup_failure_preserves_original_error_and_record(self):
        self.failed_worker(cleanup_failure=True)

    def failed_worker(self,cleanup_failure=False):
        real=subprocess.Popen;children=[]
        def launch(*args,**kwargs):
            self.assertTrue(kwargs["start_new_session"])
            child=real([sys.executable,"-c",'print(\'{"event":"unexpected"}\',flush=True); import time; time.sleep(60)'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
            child.wait=Mock(wraps=child.wait)
            children.append(child);return child
        def digest(path):return "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe" if path.name=="model.safetensors" else "test"
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/"result.json"
            kill=mc.metal_engine.kill_owned_group
            def fail_kill(child):
                kill(child)
                if cleanup_failure:raise OSError("injected cleanup failure")
            observations=0
            def power():
                nonlocal observations
                observations+=1
                if cleanup_failure and observations==2:raise OSError("injected power failure")
                return {"battery":"test","policy":"test","thermal":"test"}
            with patch.object(mc,"digest",digest),patch.object(mc.subprocess,"Popen",launch),patch.object(mc.subprocess,"check_output",return_value="test"),patch.object(mc,"endpoints",power),patch.object(mc.metal_engine,"kill_owned_group",fail_kill):
                with self.assertRaisesRegex(ValueError,"admission/mode"):mc.run(Path("/tmp/fake"),output)
            children[0].wait.assert_called_once_with(timeout=5)
            self.assertIsNotNone(children[0].returncode)
            record=json.loads(output.read_text());self.assertIn("error",record);self.assertTrue(record["identities_unchanged"])
            self.assertEqual(len(record["cleanup_errors"]),2 if cleanup_failure else 0)
            if cleanup_failure:self.assertFalse(record["passed"])
            self.assertTrue(output.with_suffix(".raw.jsonl").exists())
            with patch.object(mc.subprocess,"Popen") as launch:
                with self.assertRaisesRegex(ValueError,"destination exists"):mc.run(Path("/tmp/fake"),output)
                launch.assert_not_called()


if __name__=="__main__":unittest.main()
