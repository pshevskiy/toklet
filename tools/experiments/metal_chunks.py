"""Explicit fixed-larger Q4 chunk refinement/sweep; no quality/release gate."""
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
CHUNKS=[128,256,512,1024]
PROMPT_SHA="8acd07e9a1a11ae7e6642247f9e7621da9668863b58060620629bbad366a1ba9"
LIMIT=prefill.LIMIT
digest=prefill.digest


def ready(event,check=False):
    required={"event":"ready","format":"toklet-metal-chunks-v1","check_only":check,"debug_assertions":False,
              "chunks":CHUNKS,"projection_mode":"Q4CompensatedTiled","normalization_mode":"Compensated",
              "hidden_size":1536,"num_layers":28,"vocab_size":151936,"capacity":2031 if check else 2048,
              "position_offset":17 if check else 0,"prompt_tokens":1024,"output_tokens":1,
              "warmups":0 if check else 3,"samples":3 if check else 10,"requires_logit_bits":True,
              "application_bytes":64*1024**2}
    if any(type(event.get(k)) is not type(v) or event[k]!=v for k,v in required.items()):raise ValueError("Chunk admission/mode differs")
    widths=event.get("q4_execution_widths")
    if type(widths) is not list or len(widths)!=2 or any(type(w) is not int or not 0<w<=0xffffffff for w in widths) or widths[0]!=widths[1]:raise ValueError("Queried chunk refinement widths differ")
    memories=event.get("memories")
    if type(memories) is not list or len(memories)!=4:raise ValueError("Chunk memory inventory differs")
    for value in memories:
        prefill.memory(value)
        if value["weights_bytes"]!=1223992516:raise ValueError("Frozen larger resident weights differ")
    combined=sum(m["admitted_bytes"] for m in memories)-3*memories[0]["weights_bytes"]+event["application_bytes"]
    if type(event.get("combined_admitted_bytes")) is not int or event["combined_admitted_bytes"]!=combined or not 0<combined<=LIMIT:raise ValueError("Shared chunk admission differs")
    if any(memories[i]["scratch_bytes"]>=memories[i+1]["scratch_bytes"] for i in range(3)):raise ValueError("Chunk scratch inventories not increasing")


def sample(event,index,admission,check=False):
    case="reset" if not check or index==0 else "append7" if index==1 else "edit1000"
    expected={"reused_tokens":0 if case=="reset" else 1024 if case=="append7" else 1000,
              "prefill_tokens":1024 if case=="reset" else 7 if case=="append7" else 31,
              "sampled_tokens":1,"emitted_tokens":1,"history_tokens":1025 if case=="reset" else 1032,
              "evaluated_tokens":1024 if case=="reset" else 1031,"pending_tokens":1}
    order=event.get("order")
    if (event.get("event")!="sample" or type(event.get("run")) is not int or event["run"]!=index
            or event.get("case")!=case or event.get("order")!=[(index+i)%4 for i in range(4)]
            or type(order) is not list or any(type(i) is not int for i in order)
            or event.get("logit_bits_equal") is not True):raise ValueError("Chunk round coverage/refinement differs")
    ids=event.get("greedy_ids")
    if type(ids) is not list or len(ids)!=4 or any(type(v) is not int or not 0<=v<151936 or v!=ids[0] for v in ids):raise ValueError("Chunk greedy IDs differ")
    if event.get("memories")!=admission["memories"]:raise ValueError("Persistent chunk buffers changed")
    reports=event.get("reports")
    if type(reports) is not list or len(reports)!=4:raise ValueError("Chunk report coverage differs")
    for report in reports:
        if any(type(report.get(k)) is not int or report[k]!=v for k,v in expected.items()) or report.get("reason")!={"kind":"output_limit"}:raise ValueError("Chunk request/cache semantics differ")
        for key in ("first_token_seconds","prefill_seconds","total_seconds"):
            if type(report.get(key)) not in (int,float) or not math.isfinite(report[key]) or report[key]<=0:raise ValueError("Completed chunk timing invalid")
        if not report["prefill_seconds"]<=report["first_token_seconds"]<=report["total_seconds"] or report["decode_seconds"]!=0:raise ValueError("Chunk timing boundary differs")


def analyze(events):
    if len(events)!=13 or [e["run"] for e in events]!=list(range(13)):raise ValueError("Chunk measured coverage differs")
    rows=[]
    for i,chunk in enumerate(CHUNKS):
        values=[e["reports"][i]["first_token_seconds"] for e in events[3:]]
        median=statistics.median(values)
        rows.append({"chunk":chunk,"median_seconds":median,"minimum_seconds":min(values),"maximum_seconds":max(values),
                     "median_absolute_deviation_seconds":statistics.median(abs(v-median) for v in values),"raw_seconds":values})
    for row in rows:row["improvement_vs_128_fraction"]=1-row["median_seconds"]/rows[0]["median_seconds"]
    return {"rows":rows,"primary_metric":"Warm reset/prefill to first token event;1024 input,one output; completed GPU/full final-logit readback/CPU greedy included",
            "scope":"Rotating four-candidate order; component sweep only,not full128-output workload or release qualification"}


def run(binary,output,check=False):
    paths=[output,output.with_suffix(".raw.jsonl"),output.with_suffix(".stderr")]
    if any(p.exists() for p in paths):raise ValueError("Chunk evidence destination exists")
    source=ROOT/"models/Qwen2.5-1.5B-Instruct";packed=ROOT/"artifacts/m5-larger-q4-v1";contract=ROOT/"benchmarks/contract-v1"
    files=[Path(__file__),Path(prefill.__file__),Path(metal_engine.__file__),Path(native_usage.__file__),Path(prefill.continuation.__file__),binary,
           ROOT/"Cargo.toml",ROOT/"Cargo.lock",ROOT/"examples/metal_chunks.rs",source/"model.safetensors",source/"config.json",packed/"manifest.json",contract/"prompt_1024.u32le"]
    files+=list((ROOT/"src").rglob("*.rs"))+list((ROOT/"src").rglob("*.metal"))+list(packed.glob("*.q4"))
    identities={str(p):digest(p) for p in files}
    pins={source/"model.safetensors":prefill.LARGER_SOURCE_SHA,source/"config.json":prefill.LARGER_CONFIG_SHA,
          packed/"manifest.json":prefill.LARGER_PACKED_SHA,contract/"prompt_1024.u32le":PROMPT_SHA}
    if any(identities[str(p)]!=sha for p,sha in pins.items()):raise ValueError("Frozen larger chunk source/artifact/prompt changed")
    result={"format":"toklet-metal-chunks-v1","passed":False,"check_only":check,"scope":"Unchanged accurate larger Q4 refinement/chunk component; fixed held-out rejection remains, no quality/release/default acceptance",
            "revision":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"identities":identities,"events":[],"idle_usage":[],
            "platform":prefill.observation(["/usr/bin/sw_vers"]),"compiler":prefill.observation(["rustc","-Vv"]),"power_before":prefill.endpoints(),
            "build":{"command":"CARGO_TARGET_DIR=artifacts/m5-checks cargo build --offline --release --features metal --example metal_chunks","profile":"Cargo release defaults; no RUSTFLAGS override"},
            "thermal_scope":"Power/thermal endpoints only; continuous sensors and energy unmeasured"}
    child=None;transport=None
    rounds=3 if check else 13
    with paths[0].open("x") as summary,paths[1].open("x") as raw,paths[2].open("xb") as stderr:
        try:
            command=[str(binary),str(source),str(packed),str(contract)]+(["check"] if check else [])
            child=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
            transport=metal_engine.Transport(child,raw,stderr,stdout_limit=1024**2,event_limit=32,request_timeout=300,lifetime_timeout=600 if check else 3600)
            result["transport_policy"]={"request_seconds":300,"lifetime_seconds":600 if check else 3600,"stdout_bytes":1024**2,"event_count":32}
            admission=transport.read();ready(admission,check);result["ready"]=admission
            def proceed():child.stdin.write(b"continue\n");child.stdin.flush()
            def usage(phase):
                worker=native_usage.usage(child.pid);driver=native_usage.usage()
                values=[v["lifetime_max_phys_footprint"] for v in (worker,driver)]
                if any(not 0<v<=LIMIT for v in values) or sum(values)>LIMIT:raise ValueError("Joint chunk native charge exceeds6GiB")
                result["idle_usage"].append({"phase":phase,"worker":worker,"driver":driver})
            usage("ready");proceed()
            for i in range(rounds):
                event=transport.read();sample(event,i,admission,check);result["events"].append(event)
                if i in (2,rounds-1):usage(f"after_round_{i}")
                proceed()
            done=transport.read()
            if done!={"event":"done","rounds":rounds,"combined_admitted_bytes":admission["combined_admitted_bytes"]}:raise ValueError("Chunk completion differs")
            proceed()
            if transport.finish()!=0 or transport.events!=rounds+2 or transport.pending:raise ValueError("Chunk worker failed or emitted extra output")
            if not check:result["analysis"]=analyze(result["events"])
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
    if not result["passed"]:raise ValueError("Chunk collection failed; see retained evidence")
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--check",action="store_true",help="three bounded offset/append/edit refinement rounds; no performance acceptance")
    args=parser.parse_args();run(args.binary.resolve(),args.output.resolve(),args.check)
