"""Owned paired continuation measurement; quality/release acceptance is separate."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess

import metal_engine
import native_usage

ROOT=Path(__file__).resolve().parents[2]
LIMIT=6*1024**3
FIELDS=("weights_bytes","kv_bytes","scratch_bytes","history_bytes","logits_bytes","metadata_bytes","metadata_allowance_bytes")


def observation(command):
    return subprocess.check_output(command,text=True,timeout=10).strip()


def endpoints():
    return {"battery":observation(["/usr/bin/pmset","-g","batt"]),
            "policy":observation(["/usr/bin/pmset","-g","custom"]),
            "thermal":observation(["/usr/bin/pmset","-g","therm"])}


def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as file:
        while data:=file.read(65536):h.update(data)
    return h.hexdigest()


def memory(value):
    if any(type(value.get(k)) is not int or value[k]<0 for k in FIELDS) or value.get("admitted_bytes")!=sum(value[k] for k in FIELDS):
        raise ValueError("continuation memory accounting differs")


def ready(event,quantized):
    required={"event":"ready","format":"toklet-metal-continuation-v1","quantized":quantized,
              "debug_assertions":False,"warmups":3,"samples":10,"cases":["changed128","tail_edit"],"capacity":1024,"chunk":128}
    if any(event.get(k)!=v for k,v in required.items()):raise ValueError("continuation admission/mode differs")
    a,b=event["baseline_memory"],event["reuse_memory"];memory(a);memory(b)
    if a["weights_bytes"]!=b["weights_bytes"] or event["combined_admitted_bytes"]!=a["admitted_bytes"]+b["admitted_bytes"]-a["weights_bytes"]+event["application_bytes"] or not 0<event["combined_admitted_bytes"]<=LIMIT:
        raise ValueError("combined continuation admission differs")
    if event["numerical_gate"]!={"atol":1e-4,"rtol":1e-4}:raise ValueError("continuation numerical gate changed")


def sample(event,run,case,admission):
    index=("changed128","tail_edit").index(case);keep=128 if index==0 else 511
    if event.get("event")!="sample" or event.get("run")!=run or event.get("case")!=case or event.get("order")!=("reset_first" if (run+index)%2==0 else "reuse_first"):
        raise ValueError("continuation sample coverage/order differs")
    for mode,reused,prefill in (("reset",0,512),("reuse",keep,512-keep)):
        r=event[f"{mode}_report"]
        expected={"reused_tokens":reused,"prefill_tokens":prefill,"sampled_tokens":1,"emitted_tokens":1,"history_tokens":513,"evaluated_tokens":512,"pending_tokens":1}
        if any(r.get(k)!=v for k,v in expected.items()) or r.get("reason")!={"kind":"output_limit"}:raise ValueError("continuation request semantics differ")
        for key in ("first_token_seconds","prefill_seconds","total_seconds"):
            if type(r.get(key)) not in (int,float) or not math.isfinite(r[key]) or r[key]<=0:raise ValueError("continuation timing invalid")
        if r["first_token_seconds"]>r["total_seconds"] or r["decode_seconds"]!=0:raise ValueError("continuation timing boundaries differ")
        m=event[f"{mode}_memory"];memory(m)
        if m!=admission["baseline_memory" if mode=="reset" else "reuse_memory"]:raise ValueError("persistent continuation buffers changed")
    if not 0<=event["greedy_id"]<151936 or not math.isfinite(event["max_logit_delta"]) or event["max_logit_delta"]<0:raise ValueError("continuation correctness evidence invalid")


def analyze(events):
    rows=[]
    for case in ("changed128","tail_edit"):
        selected=[e for e in events if e["case"]==case and e["run"]>=3]
        if len(selected)!=10:raise ValueError("continuation measured coverage incomplete")
        values={mode:[e[f"{mode}_report"]["first_token_seconds"] for e in selected] for mode in ("reset","reuse")}
        medians={mode:statistics.median(v) for mode,v in values.items()}
        rows.append({"case":case,"samples_seconds":values,"median_seconds":medians,
                     "ranges_seconds":{m:[min(v),max(v)] for m,v in values.items()},
                     "improvement_fraction":1-medians["reuse"]/medians["reset"],
                     "max_logit_delta":max(e["max_logit_delta"] for e in selected)})
    return rows


def run(binary,output,quantized=True):
    paths=[output,output.with_suffix(".raw.jsonl"),output.with_suffix(".stderr")]
    if any(p.exists() for p in paths):raise ValueError("continuation evidence destination exists")
    source=ROOT/"models/Qwen2.5-0.5B-Instruct";packed=ROOT/"artifacts/m5-qwen-q4-v1"
    if digest(source/"model.safetensors")!="fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe":raise ValueError("pinned source identity differs")
    files=[Path(__file__),Path(metal_engine.__file__),Path(native_usage.__file__),ROOT/"Cargo.toml",ROOT/"Cargo.lock",ROOT/"examples/metal_continue.rs",source/"config.json",source/"model.safetensors",binary,
           ROOT/"benchmarks/contract-v1/prompt_512.u32le",ROOT/"benchmarks/contract-v1/changed_prompt_512.u32le"]+list((ROOT/"src").rglob("*.rs"))+list((ROOT/"src").rglob("*.metal"))
    if quantized:files+=list(packed.glob("*.q4"))+[packed/"manifest.json"]
    identities={str(p):digest(p) for p in files}
    result={"format":"toklet-metal-continuation-v1","passed":False,"scope":"paired cache-policy engine TTFT; no held-out quality or external speed/release acceptance",
            "revision":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"identities":identities,"events":[],"idle_usage":[],
            "platform":observation(["/usr/bin/sw_vers"]),"power_before":endpoints(),
            "compiler":observation(["rustc","-Vv"]),
            "build":{"command":"CARGO_TARGET_DIR=artifacts/m5-checks cargo build --offline --release --features metal --example metal_continue",
                     "profile":"Cargo release defaults; no RUSTFLAGS override","features":["metal"]},
            "thermal_scope":"Endpoint observations only; continuous thermal pressure is not measured."}
    child=None;transport=None
    with paths[0].open("x") as summary,paths[1].open("x") as raw,paths[2].open("xb") as stderr:
        try:
            child=subprocess.Popen([str(binary),str(source),str(packed) if quantized else "-",str(ROOT/"benchmarks/contract-v1")],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
            transport=metal_engine.Transport(child,raw,stderr,request_timeout=120,lifetime_timeout=1200)
            result["transport_policy"]={"request_seconds":120,"lifetime_seconds":1200}
            admission=transport.read();ready(admission,quantized);result["ready"]=admission
            def proceed():child.stdin.write(b"continue\n");child.stdin.flush()
            def usage(phase):
                value=native_usage.usage(child.pid)
                if not 0<value["lifetime_max_phys_footprint"]<=LIMIT:raise ValueError("native continuation charge exceeds budget")
                result["idle_usage"].append({"phase":phase,**value})
            usage("ready");proceed()
            for iteration in range(13):
                for case in ("changed128","tail_edit"):
                    event=transport.read();sample(event,iteration,case,admission);result["events"].append(event)
                    if (iteration,case) in ((2,"tail_edit"),(12,"tail_edit")):usage(f"after_run_{iteration}")
                    proceed()
            done=transport.read()
            if done!={"event":"done","pairs":26,"combined_admitted_bytes":admission["combined_admitted_bytes"]}:raise ValueError("continuation completion differs")
            proceed()
            if transport.finish()!=0 or transport.events!=28 or transport.pending:
                raise ValueError("continuation worker failed or emitted extra output after completion")
            result["analysis"]=analyze(result["events"]);result["passed"]=True
        except BaseException as error:
            result["error"]=f"{type(error).__name__}: {error}";raise
        finally:
            result["cleanup_errors"]=[]
            def cleanup(label,action):
                try:action()
                except BaseException as error:result["cleanup_errors"].append(f"{label}: {type(error).__name__}: {error}")
            if child:
                cleanup("kill owned group",lambda:metal_engine.kill_owned_group(child))
                cleanup("reap owned child",lambda:child.wait(timeout=5))
            if transport:cleanup("close transport",transport.close)
            if child:
                for stream in (child.stdin,child.stdout,child.stderr):
                    if stream:cleanup("close pipe",stream.close)
            try:result["identities_unchanged"]=all(digest(Path(p))==identity for p,identity in identities.items())
            except OSError as error:result["identities_unchanged"]=False;result["identity_error"]=str(error)
            if not result["identities_unchanged"]:result["passed"]=False
            cleanup("power endpoints",lambda:result.update(power_after=endpoints()))
            if result["cleanup_errors"]:result["passed"]=False
            json.dump(result,summary,indent=2);summary.write("\n")
    if result["cleanup_errors"]:raise RuntimeError("continuation cleanup failed; see retained evidence")
    if not result["identities_unchanged"]:raise ValueError("continuation identities changed during measurement")
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--fp16",action="store_true");args=parser.parse_args();run(args.binary.resolve(),args.output.resolve(),not args.fp16)
