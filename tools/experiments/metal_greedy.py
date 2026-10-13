"""Paired completed generation with explicit GPU greedy readback; no release gate."""
import argparse
import json
import math
from pathlib import Path
import statistics
import subprocess

import metal_engine
import metal_prefill as prefill
import native_usage

ROOT=Path(__file__).resolve().parents[2]
PROMPT_SHA="4ff74dae2ee5b778f9313f4c707f9ccb6415609a9c0653684cce7be63d457dd6"
LIMIT=prefill.LIMIT
digest=prefill.digest


def ready(event):
    required={"event":"ready","format":"toklet-metal-greedy-v1","debug_assertions":False,
              "readback_modes":["FullLogits","GreedyToken"],"projection_mode":"Q4CompensatedTiled","normalization_mode":"Compensated",
              "hidden_size":1536,"num_layers":28,"vocab_size":151936,"capacity":1024,"chunk":128,
              "prompt_tokens":512,"output_tokens":128,"warmups":3,"samples":10,"application_bytes":64*1024**2}
    if any(type(event.get(k)) is not type(v) or event[k]!=v for k,v in required.items()):raise ValueError("Greedy admission/mode differs")
    width=event.get("greedy_execution_width")
    if type(width) is not int or not 0<width<=0xffffffff:raise ValueError("Queried greedy SIMD width invalid")
    a,b=event["full_memory"],event["token_memory"]
    for value in (a,b):
        prefill.memory(value)
        if value["weights_bytes"]!=1223992516:raise ValueError("Frozen larger resident weights differ")
    if a["logits_bytes"]!=151936*8 or b["logits_bytes"]!=8 or b["scratch_bytes"]!=a["scratch_bytes"]+32:raise ValueError("Greedy host/scratch storage differs")
    if any(a[k]!=b[k] for k in prefill.continuation.FIELDS if k not in ("logits_bytes","scratch_bytes")):raise ValueError("Greedy unrelated memory differs")
    combined=a["admitted_bytes"]+b["admitted_bytes"]-a["weights_bytes"]+event["application_bytes"]
    if type(event.get("combined_admitted_bytes")) is not int or event["combined_admitted_bytes"]!=combined or not 0<combined<=LIMIT:raise ValueError("Shared greedy admission differs")


def sample(event,index,admission):
    if (event.get("event")!="sample" or type(event.get("run")) is not int or event["run"]!=index
            or event.get("order")!=("full_first" if index%2==0 else "token_first") or event.get("token_ids_equal") is not True):raise ValueError("Greedy paired coverage/refinement differs")
    ids=event.get("token_ids")
    if type(ids) is not list or len(ids)!=128 or any(type(v) is not int or not 0<=v<151936 for v in ids):raise ValueError("Greedy token coverage differs")
    expected={"reused_tokens":0,"prefill_tokens":512,"sampled_tokens":128,"emitted_tokens":128,"history_tokens":640,"evaluated_tokens":639,"pending_tokens":1}
    for mode in ("full","token"):
        report=event[f"{mode}_report"]
        if any(type(report.get(k)) is not int or report[k]!=v for k,v in expected.items()) or report.get("reason")!={"kind":"output_limit"}:raise ValueError("Greedy generation/cache semantics differ")
        for key in ("first_token_seconds","prefill_seconds","decode_seconds","sampling_seconds","total_seconds"):
            if type(report.get(key)) not in (int,float) or not math.isfinite(report[key]) or report[key]<=0:raise ValueError("Completed greedy timing invalid")
        if not report["prefill_seconds"]<=report["first_token_seconds"]<=report["total_seconds"] or sum(report[k] for k in ("prefill_seconds","decode_seconds","sampling_seconds"))>report["total_seconds"]:raise ValueError("Greedy timing boundary differs")
        if event.get(f"{mode}_memory")!=admission[f"{mode}_memory"]:raise ValueError("Persistent greedy buffers changed")
        timing=event[f"{mode}_last_timing"]
        if type(timing.get("command_count")) is not int or timing["command_count"]!=1:raise ValueError("Greedy command boundary differs")
        for key in ("command_creation_seconds","clock_calibration_seconds","commit_seconds","host_encode_seconds","wait_seconds","gpu_start_seconds","gpu_end_seconds","readback_seconds"):
            if type(timing.get(key)) not in (int,float) or not math.isfinite(timing[key]) or timing[key]<0:raise ValueError("Greedy completion timing invalid")
        if timing["gpu_end_seconds"]<timing["gpu_start_seconds"]:raise ValueError("Greedy GPU completion order differs")


def analyze(events):
    if len(events)!=13 or [e["run"] for e in events]!=list(range(13)):raise ValueError("Greedy measured coverage differs")
    result={}
    for mode in ("full","token"):
        values=[e[f"{mode}_report"]["total_seconds"] for e in events[3:]]
        median=statistics.median(values)
        result[mode]={"median_seconds":median,"minimum_seconds":min(values),"maximum_seconds":max(values),
                      "median_absolute_deviation_seconds":statistics.median(abs(v-median) for v in values),"raw_seconds":values,
                      "component_medians_seconds":{k:statistics.median(e[f"{mode}_report"][k] for e in events[3:]) for k in ("first_token_seconds","prefill_seconds","decode_seconds","sampling_seconds")}}
    result["improvement_fraction"]=1-result["token"]["median_seconds"]/result["full"]["median_seconds"]
    result["primary_metric"]="Warm complete reset512-input/128-output engine generation; GPU completion, selection/readback and preallocated ID callback included"
    result["timing_partition"]="FullLogits sampling includes CPU greedy; GreedyToken selection is in completed prefill/decode GPU work and sampling only fetches cached ID. Components cannot establish speed independently."
    return result


def run(binary,output):
    paths=[output,output.with_suffix(".raw.jsonl"),output.with_suffix(".stderr")]
    if any(p.exists() for p in paths):raise ValueError("Greedy evidence destination exists")
    source=ROOT/"models/Qwen2.5-1.5B-Instruct";packed=ROOT/"artifacts/m5-larger-q4-v1";contract=ROOT/"benchmarks/contract-v1"
    files=[Path(__file__),Path(prefill.__file__),Path(metal_engine.__file__),Path(native_usage.__file__),Path(prefill.continuation.__file__),binary,
           ROOT/"Cargo.toml",ROOT/"Cargo.lock",ROOT/"examples/metal_greedy.rs",source/"model.safetensors",source/"config.json",packed/"manifest.json",contract/"prompt_512.u32le"]
    files+=list((ROOT/"src").rglob("*.rs"))+list((ROOT/"src").rglob("*.metal"))+list(packed.glob("*.q4"))
    identities={str(p):digest(p) for p in files}
    pins={source/"model.safetensors":prefill.LARGER_SOURCE_SHA,source/"config.json":prefill.LARGER_CONFIG_SHA,
          packed/"manifest.json":prefill.LARGER_PACKED_SHA,contract/"prompt_512.u32le":PROMPT_SHA}
    if any(identities[str(p)]!=sha for p,sha in pins.items()):raise ValueError("Frozen larger greedy source/artifact/prompt changed")
    result={"format":"toklet-metal-greedy-v1","passed":False,"scope":"Paired fixed larger Q4 full512-input/128-output generation selection/readback component; quality rejection remains, no release/default acceptance",
            "revision":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"identities":identities,"events":[],"idle_usage":[],
            "platform":prefill.observation(["/usr/bin/sw_vers"]),"compiler":prefill.observation(["rustc","-Vv"]),"power_before":prefill.endpoints(),
            "build":{"command":"CARGO_TARGET_DIR=artifacts/m5-checks cargo build --offline --release --features metal --example metal_greedy","profile":"Cargo release defaults; no RUSTFLAGS override"},
            "thermal_scope":"Power/thermal endpoints only; continuous sensors and energy unmeasured"}
    child=None;transport=None
    rounds=13
    with paths[0].open("x") as summary,paths[1].open("x") as raw,paths[2].open("xb") as stderr:
        try:
            command=[str(binary),str(source),str(packed),str(contract)]
            child=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
            transport=metal_engine.Transport(child,raw,stderr,stdout_limit=1024**2,event_limit=32,request_timeout=300,lifetime_timeout=3600)
            result["transport_policy"]={"request_seconds":300,"lifetime_seconds":3600,"stdout_bytes":1024**2,"event_count":32}
            admission=transport.read();ready(admission);result["ready"]=admission
            def proceed():child.stdin.write(b"continue\n");child.stdin.flush()
            def usage(phase):
                worker=native_usage.usage(child.pid);driver=native_usage.usage()
                values=[v["lifetime_max_phys_footprint"] for v in (worker,driver)]
                if any(not 0<v<=LIMIT for v in values) or sum(values)>LIMIT:raise ValueError("Joint greedy native charge exceeds6GiB")
                result["idle_usage"].append({"phase":phase,"worker":worker,"driver":driver})
            usage("ready");proceed()
            for i in range(rounds):
                event=transport.read();sample(event,i,admission);result["events"].append(event)
                if i in (2,rounds-1):usage(f"after_round_{i}")
                proceed()
            done=transport.read()
            if done!={"event":"done","pairs":rounds,"combined_admitted_bytes":admission["combined_admitted_bytes"]}:raise ValueError("Greedy completion differs")
            proceed()
            if transport.finish()!=0 or transport.events!=rounds+2 or transport.pending:raise ValueError("Greedy worker failed or emitted extra output")
            result["analysis"]=analyze(result["events"])
            result["passed"]=True
        except BaseException as error:result["error"]=f"{type(error).__name__}: {error}"
        finally:
            result["cleanup_errors"]=[]
            def cleanup(label,action):
                try:action()
                except BaseException as error:result["cleanup_errors"].append(f"{label}: {type(error).__name__}: {error}")
            if child:
                cleanup("kill owned group",lambda:metal_engine.kill_owned_group(child));cleanup("reap owned child",lambda:child.wait(timeout=5))
            if transport:cleanup("close transport",transport.close)
            if child:
                for stream in (child.stdin,child.stdout,child.stderr):
                    if stream:cleanup("close pipe",stream.close)
            try:result["identities_unchanged"]=all(digest(Path(p))==sha for p,sha in identities.items())
            except OSError as error:result["identities_unchanged"]=False;result["identity_error"]=str(error)
            cleanup("power endpoints",lambda:result.update(power_after=prefill.endpoints()))
            if not result["identities_unchanged"] or result["cleanup_errors"]:result["passed"]=False
            json.dump(result,summary,indent=2,allow_nan=False);summary.write("\n")
    if not result["passed"]:raise ValueError("Greedy collection failed; see retained evidence")
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();run(args.binary.resolve(),args.output.resolve())
