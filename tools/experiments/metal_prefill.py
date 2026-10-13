"""Paired Q4 tile warm512-token first-event measurements; no release gate."""
import argparse
import math
from pathlib import Path
import json
import statistics
import subprocess

import metal_continue as continuation
import metal_engine
import native_usage

ROOT=Path(__file__).resolve().parents[2]
LIMIT=6*1024**3
SOURCE_SHA="fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"
CONFIG_SHA="18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45"
LARGER_SOURCE_SHA="dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee"
LARGER_CONFIG_SHA="98d2ff8cc47488d08a2b0b3acf4eb99ef210779b42bd48605f6b8e36acdbf670"
LARGER_PACKED_SHA="236e2a2071ad90967008e81b3af262275fc7c81307f303eab577f8681bc6c8c3"
digest=continuation.digest
observation=continuation.observation
endpoints=continuation.endpoints
memory=continuation.memory


def ready(event,accurate=False,larger=False):
    required={"event":"ready","format":"toklet-metal-q4-prefill-v1","projection_modes":["Q4Compensated","Q4CompensatedTiled"] if accurate else ["Original","Q4Tiled"],
              "debug_assertions":False,"warmups":3,"samples":10,"prompt_tokens":512,"output_tokens":1,"capacity":1024,"chunk":128}
    if accurate:required.update(normalization_mode="Compensated",requires_logit_bits=True,hidden_size=1536 if larger else 896,num_layers=28 if larger else 24)
    if any(type(event.get(k)) is not type(v) or event[k]!=v for k,v in required.items()):raise ValueError("Prefill admission/mode differs")
    widths=event.get("q4_execution_widths")
    if type(widths) is not list or len(widths)!=2 or any(type(w) is not int or not 0<w<=0xffffffff for w in widths):raise ValueError("Queried Q4 SIMD widths invalid")
    if accurate and widths[0]!=widths[1]:raise ValueError("Accurate tile refinement needs equal queried widths")
    a,b=event["original_memory"],event["tiled_memory"];memory(a);memory(b)
    if (type(event.get("application_bytes")) is not int or event["application_bytes"]!=64*1024**2
            or type(event.get("combined_admitted_bytes")) is not int
            or a["weights_bytes"]!=b["weights_bytes"] or event["combined_admitted_bytes"]!=a["admitted_bytes"]+b["admitted_bytes"]-a["weights_bytes"]+event["application_bytes"]
            or not 0<event["combined_admitted_bytes"]<=LIMIT):raise ValueError("Combined prefill admission differs")
    if event["numerical_gate"]!={"atol":1e-4,"rtol":1e-4}:raise ValueError("Prefill same-Q4 envelope changed")


def sample(event,index,admission):
    if event.get("event")!="sample" or type(event.get("run")) is not int or event["run"]!=index or event.get("order")!=("original_first" if index%2==0 else "tiled_first"):
        raise ValueError("Prefill paired sample coverage/order differs")
    for mode in ("original","tiled"):
        report=event[f"{mode}_report"]
        expected={"reused_tokens":0,"prefill_tokens":512,"sampled_tokens":1,"emitted_tokens":1,"history_tokens":513,"evaluated_tokens":512,"pending_tokens":1}
        if any(type(report.get(k)) is not int or report[k]!=v for k,v in expected.items()) or report.get("reason")!={"kind":"output_limit"}:raise ValueError("Prefill request semantics differ")
        for key in ("first_token_seconds","prefill_seconds","total_seconds"):
            if type(report.get(key)) not in (int,float) or not math.isfinite(report[key]) or report[key]<=0:raise ValueError("Prefill timing invalid")
        if not report["prefill_seconds"]<=report["first_token_seconds"]<=report["total_seconds"] or report["decode_seconds"]!=0:raise ValueError("Prefill timing boundary differs")
        value=event[f"{mode}_memory"];memory(value)
        if value!=admission[f"{mode}_memory"]:raise ValueError("Persistent prefill buffers changed")
    if (type(event.get("greedy_id")) is not int or not 0<=event["greedy_id"]<151936
            or type(event.get("max_logit_delta")) not in (int,float) or not math.isfinite(event["max_logit_delta"]) or event["max_logit_delta"]<0):raise ValueError("Prefill numerical evidence invalid")
    if admission.get("requires_logit_bits") and event.get("logit_bits_equal") is not True:raise ValueError("Accurate tile logit bits differ")


def analyze(events):
    measured=[event for event in events if event["run"]>=3]
    if len(events)!=13 or len(measured)!=10:raise ValueError("Prefill measured sample coverage differs")
    result={}
    for mode in ("original","tiled"):
        values=[event[f"{mode}_report"]["first_token_seconds"] for event in measured]
        median=statistics.median(values)
        result[mode]={"median_seconds":median,"minimum_seconds":min(values),"maximum_seconds":max(values),
                      "median_absolute_deviation_seconds":statistics.median(abs(v-median) for v in values),"raw_seconds":values}
    result["improvement_fraction"]=1-result["tiled"]["median_seconds"]/result["original"]["median_seconds"]
    result["max_logit_delta"]=max(event["max_logit_delta"] for event in measured)
    result["primary_metric"]="Warm request reset/prefill to first token event;512input,1output; timings include GPU completion,full last-logits readback,CPUgreedy"
    return result


def run(binary,output,accurate=False,larger=False):
    if larger and not accurate:raise ValueError("Larger prefill requires accurate mode")
    paths=[output,output.with_suffix(".raw.jsonl"),output.with_suffix(".stderr")]
    if any(p.exists() for p in paths):raise ValueError("prefill evidence destination exists")
    source=ROOT/"models"/("Qwen2.5-1.5B-Instruct" if larger else "Qwen2.5-0.5B-Instruct");packed=ROOT/"artifacts"/("m5-larger-q4-v1" if larger else "m5-qwen-q4-v1")
    source_sha=LARGER_SOURCE_SHA if larger else SOURCE_SHA;config_sha=LARGER_CONFIG_SHA if larger else CONFIG_SHA
    if digest(source/"model.safetensors")!=source_sha:raise ValueError("pinned source identity differs")
    files=[Path(__file__),Path(metal_engine.__file__),Path(native_usage.__file__),ROOT/"Cargo.toml",ROOT/"Cargo.lock",ROOT/"examples/metal_prefill.rs",source/"config.json",source/"model.safetensors",binary,
           ROOT/"benchmarks/contract-v1/prompt_512.u32le"]+list((ROOT/"src").rglob("*.rs"))+list((ROOT/"src").rglob("*.metal"))+[Path(continuation.__file__)]
    files+=list(packed.glob("*.q4"))+[packed/"manifest.json"]
    identities={str(p):digest(p) for p in files}
    if identities[str(source/"model.safetensors")]!=source_sha or identities[str(source/"config.json")]!=config_sha:raise ValueError("Initial pinned prefill source/config changed")
    if larger and identities[str(packed/"manifest.json")]!=LARGER_PACKED_SHA:raise ValueError("Frozen larger Q4 artifact changed")
    result={"format":"toklet-metal-q4-prefill-v1","passed":False,"scope":"Paired opt-in Q4 tile warm512-input/one-output engine TTFT; no full decode workload,heldout quality,external lead or release acceptance",
            "accurate":accurate,"larger":larger,"revision":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"identities":identities,"events":[],"idle_usage":[],
            "platform":observation(["/usr/bin/sw_vers"]),"power_before":endpoints(),
            "compiler":observation(["rustc","-Vv"]),
            "build":{"command":"CARGO_TARGET_DIR=artifacts/m5-checks cargo build --offline --release --features metal --example metal_prefill",
                     "profile":"Cargo release defaults; no RUSTFLAGS override","features":["metal"]},
            "thermal_scope":"Endpoint observations only; continuous thermal pressure is not measured."}
    child=None;transport=None
    with paths[0].open("x") as summary,paths[1].open("x") as raw,paths[2].open("xb") as stderr:
        try:
            command=[str(binary),str(source),str(packed),str(ROOT/"benchmarks/contract-v1")]
            if accurate:command.append("accurate")
            if larger:command.append("larger")
            child=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
            transport=metal_engine.Transport(child,raw,stderr,request_timeout=120,lifetime_timeout=1200)
            result["transport_policy"]={"request_seconds":120,"lifetime_seconds":1200}
            admission=transport.read();ready(admission,accurate,larger);result["ready"]=admission
            def proceed():child.stdin.write(b"continue\n");child.stdin.flush()
            def usage(phase):
                value=native_usage.usage(child.pid);driver=native_usage.usage()
                if any(not 0<v["lifetime_max_phys_footprint"]<=LIMIT for v in (value,driver)) or value["lifetime_max_phys_footprint"]+driver["lifetime_max_phys_footprint"]>LIMIT:raise ValueError("joint native prefill charge exceeds budget")
                result["idle_usage"].append({"phase":phase,"worker":value,"driver":driver})
            usage("ready");proceed()
            for iteration in range(13):
                event=transport.read();sample(event,iteration,admission);result["events"].append(event)
                if iteration in (2,12):usage(f"after_run_{iteration}")
                proceed()
            done=transport.read()
            if done!={"event":"done","pairs":13,"combined_admitted_bytes":admission["combined_admitted_bytes"]}:raise ValueError("prefill completion differs")
            proceed()
            if transport.finish()!=0 or transport.events!=15 or transport.pending:
                raise ValueError("prefill worker failed or emitted extra output after completion")
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
    if result["cleanup_errors"]:raise RuntimeError("prefill cleanup failed; see retained evidence")
    if not result["identities_unchanged"]:raise ValueError("prefill identities changed during measurement")
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--accurate",action="store_true",help="compare compensated projection with its eight-row tile, requiring identical logit bits")
    parser.add_argument("--larger",action="store_true",help="use the fixed larger checkpoint; requires --accurate")
    args=parser.parse_args();run(args.binary.resolve(),args.output.resolve(),args.accurate,args.larger)
