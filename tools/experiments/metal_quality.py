"""Frozen Metal loss scoring; all required positions, bounded owned workers.

No latency acceptance. Batch and one-row probes use the predeclared 1e-3 NLL
envelope. The full corpus applies only the unchanged 0.05 mean-NLL quality gate.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

import metal_continue as continuation
import metal_engine
import native_usage

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"tools/bench"))
import quality

LIMIT=6*1024**3
CORPUS_SHA="f53c5b0c446959164c4d2265166040123f95be5b6d9eec784895fb45a8215beb"
SOURCE_SHA="fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"
BASELINE_SHA="1be6b1030c1835bde93a081bf4bf87f1a02bdb038a78c3f7a3109bafb17777ac"
POLICY=quality.POLICY
digest=continuation.digest


def frozen_bytes(path,limit,expected):
    with path.open("rb") as file:data=file.read(limit+1)
    if len(data)>limit or hashlib.sha256(data).hexdigest()!=expected:
        raise ValueError("Frozen parsed loss bytes differ")
    return data


def ready(event,tokens,sha,quantized,chunk):
    expected={"event":"ready","format":"toklet-metal-held-out-loss-events-v1","backend":"metal",
              "quantized":quantized,"debug_assertions":False,"target_tokens":len(tokens)-1,
              "corpus_sha256":sha,"context_policy":POLICY,"block_targets":1024,"context_capacity":1025,
              "prefill_chunk_size":chunk,"logit_positions":"all_causal_rows",
              "weight_dtype":"Q4 v1 + FP16 exclusions" if quantized else "FP16","kv_dtype":"FP16"}
    if any(type(event.get(k)) is not type(v) or event[k]!=v for k,v in expected.items()):raise ValueError("Metal loss admission/policy differs")
    continuation.memory(event["memory"])
    if (type(event.get("application_allowance_bytes")) is not int or event["application_allowance_bytes"]!=64*1024**2
            or type(event.get("memory_admitted_bytes")) is not int
            or event["memory_admitted_bytes"]!=event["memory"]["admitted_bytes"]+event["application_allowance_bytes"]
            or not 0<event["memory_admitted_bytes"]<=LIMIT):raise ValueError("Metal loss admission differs")


def loss_row(row,index,tokens):
    integers={"target_index":index,"block_index":index//1024,"position":index%1024,
              "prefix_tokens":index%1024+1,"target_token":tokens[index+1]}
    if row.get("event")!="loss" or any(type(row.get(k)) is not int or row[k]!=v for k,v in integers.items()):
        raise ValueError("Metal loss target/causal alignment differs")
    value=row.get("negative_log_likelihood")
    bits=row.get("target_logit_f32_bits");sha=row.get("logits_sha256")
    if (type(value) not in (int,float) or not math.isfinite(value) or value<0
            or type(bits) is not int or not 0<=bits<=0xffffffff
            or not isinstance(sha,str) or len(sha)!=64 or any(c not in "0123456789abcdef" for c in sha)
            or not math.isfinite(struct.unpack("<f",struct.pack("<I",bits))[0])):
        raise ValueError("Metal loss numerical evidence differs")


def chunk_done(event,block,previous,targets,chunk,scored):
    start=block*1024;size=min(1024,targets-start);rows=min(chunk,size+1-previous)
    expected={"event":"chunk_done","block_index":block,"rows":rows,"evaluated_tokens":previous+rows,
              "scored_targets":min(previous+rows,size),"pending_tokens":0}
    if rows<=0 or any(type(event.get(k)) is not int or event[k]!=v for k,v in expected.items() if k!="event") or event.get("event")!="chunk_done" or scored!=start+expected["scored_targets"]:
        raise ValueError("Metal loss chunk coverage/commitment differs")
    return previous+rows


def block_done(event,block,tokens,losses,evaluated):
    start=block*1024;end=min(start+1024,len(tokens)-1);size=end-start
    expected={"event":"block_done","block_index":block,"target_start":start,"target_end":end,
              "scored_targets":size,"evaluated_tokens":size+1,"pending_tokens":0}
    total=math.fsum(row["negative_log_likelihood"] for row in losses[start:end])
    if (any(type(event.get(k)) is not type(v) or event[k]!=v for k,v in expected.items()) or len(losses)!=end or evaluated!=size+1
            or type(event.get("total_negative_log_likelihood")) not in (int,float)
            or not math.isclose(event["total_negative_log_likelihood"],total,rel_tol=1e-13,abs_tol=1e-10)):
        raise ValueError("Metal loss block aggregation differs")


def done(event,tokens,losses,blocks,evaluated):
    count=len(tokens)-1;total=math.fsum(row["negative_log_likelihood"] for row in losses)
    expected={"event":"done","scored_targets":count,"evaluated_tokens":(count-1)%1024+2,"pending_tokens":0}
    if (any(type(event.get(k)) is not type(v) or event[k]!=v for k,v in expected.items()) or len(losses)!=count or blocks!=(count+1023)//1024
            or evaluated!=expected["evaluated_tokens"] or type(event.get("total_negative_log_likelihood")) not in (int,float)
            or not math.isclose(event["total_negative_log_likelihood"],total,rel_tol=1e-13,abs_tol=1e-10)
            or event.get("mean_negative_log_likelihood")!=event["total_negative_log_likelihood"]/count):
        raise ValueError("Metal loss completion aggregation differs")
    return total/count


def child(binary,source,packed,token_path,tokens,quantized,chunk,raw,stderr,*,admission=ready,extra_args=()):
    command=[str(binary),str(source),str(packed) if quantized else "-",str(token_path),str(chunk)]+list(extra_args)
    worker=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
    transport=None
    result={"command":command,"worker_pid":worker.pid,"passed":False,"losses":[],"chunks":[],"blocks":[],"idle_usage":[],"cleanup_errors":[],
            "transport_policy":{"request_seconds":120,"lifetime_seconds":3600,"event_limit":32768,"stdout_bytes":8*1024**2}}
    try:
        transport=metal_engine.Transport(worker,raw,stderr,request_timeout=120,lifetime_timeout=3600,event_limit=32768)
        def proceed():worker.stdin.write(b"continue\n");worker.stdin.flush()
        def usage(phase):
            left=native_usage.usage(worker.pid);right=native_usage.usage()
            if any(not 0<v["lifetime_max_phys_footprint"]<=LIMIT for v in (left,right)) or left["lifetime_max_phys_footprint"]+right["lifetime_max_phys_footprint"]>LIMIT:
                raise ValueError("Metal loss joint native charge exceeds budget")
            result["idle_usage"].append({"phase":phase,"worker":left,"driver":right})
        # Bind the worker's exact parsed input to the driver's admitted IDs,
        # rather than hashing a path again after the worker has read it.
        token_sha=hashlib.sha256(struct.pack(f"<{len(tokens)}I",*tokens)).hexdigest()
        event=transport.read();admission(event,tokens,token_sha,quantized,chunk);result["ready"]=event
        usage("ready");proceed();evaluated=0
        while True:
            event=transport.read();kind=event.get("event");block=len(result["blocks"])
            if kind=="loss":
                index=len(result["losses"])
                if index>=len(tokens)-1 or index//1024!=block:raise ValueError("Metal loss extra/out-of-block target")
                # Every target belongs to the next completed causal chunk.
                if index%1024<evaluated or index%1024>=min(evaluated+chunk,1024):raise ValueError("Metal loss target outside current chunk")
                loss_row(event,index,tokens);result["losses"].append(event)
            elif kind=="chunk_done":
                evaluated=chunk_done(event,block,evaluated,len(tokens)-1,chunk,len(result["losses"]))
                result["chunks"].append(event)
                if evaluated==min(1024,len(tokens)-1-block*1024)+1:usage(f"block_{block}")
                proceed()
            elif kind=="block_done":
                block_done(event,block,tokens,result["losses"],evaluated);result["blocks"].append(event)
                evaluated=0
            elif kind=="done":
                # Last block's completion retained its full commitment separately.
                result["mean_negative_log_likelihood"]=done(event,tokens,result["losses"],block,result["blocks"][-1]["evaluated_tokens"])
                result["done"]=event;usage("done");proceed();break
            else:raise ValueError("Metal loss event order differs")
        if transport.finish()!=0 or transport.pending or transport.events!=2+len(result["losses"])+len(result["chunks"])+len(result["blocks"]):
            raise ValueError("Metal loss worker exit/trailing output differs")
        result["passed"]=True
    except BaseException as error:
        result["error"]=f"{type(error).__name__}: {error}"
    finally:
        def cleanup(label,action):
            try:action()
            except BaseException as error:result["cleanup_errors"].append(f"{label}: {type(error).__name__}: {error}")
        cleanup("kill owned group",lambda:metal_engine.kill_owned_group(worker))
        cleanup("reap owned worker",lambda:worker.wait(timeout=5))
        if transport:cleanup("close transport",transport.close)
        for stream in (worker.stdin,worker.stdout,worker.stderr):
            if stream:cleanup("close pipe",stream.close)
        result["exit_code"]=worker.returncode
        if result["cleanup_errors"]:result["passed"]=False
    return result


def run(binary,output,quantized=True):
    paths=[output,output.with_suffix(".raw.jsonl"),output.with_suffix(".stderr")]
    if any(p.exists() for p in paths):raise ValueError("Metal loss evidence destination exists")
    source=ROOT/"models/Qwen2.5-0.5B-Instruct";packed=ROOT/"artifacts/m5-qwen-q4-v1"
    corpus=ROOT/"benchmarks/contract-v1/held_out_loss.u32le"
    baseline_path=ROOT/"benchmarks/results/2026-10-11-held-out-scalar-loss-analysis-v1.json"
    if digest(source/"model.safetensors")!=SOURCE_SHA:raise ValueError("Frozen source identity differs")
    data=frozen_bytes(corpus,10001*4,CORPUS_SHA);tokens=list(struct.unpack("<10001I",data))
    baseline=json.loads(frozen_bytes(baseline_path,1024**2,BASELINE_SHA));contract=json.loads((corpus.parent/"manifest.json").read_text())
    if baseline["context_policy"]!=POLICY or baseline["scored_targets"]!=10000 or contract["quality"]["mean_nll_increase_limit_nats_per_token"]!=.05:
        raise ValueError("Frozen loss context/gate changed")
    files=[Path(__file__),Path(metal_engine.__file__),Path(native_usage.__file__),Path(continuation.__file__),Path(quality.__file__),Path(quality.prepare.__file__),binary,
           ROOT/"tests/fixtures/qwen-v2/manifest.json",ROOT/"artifacts/qwen-forward-v2/manifest.json",
           corpus,corpus.parent/"manifest.json",baseline_path,source/"config.json",source/"model.safetensors",ROOT/"Cargo.toml",ROOT/"Cargo.lock",
           ROOT/"examples/metal_quality.rs",ROOT/"examples/support/loss.rs",ROOT/"examples/quality_bench.rs"]+list((ROOT/"src").rglob("*.rs"))+list((ROOT/"src").rglob("*.metal"))
    if quantized:files+=list(packed.glob("*.q4"))+[packed/"manifest.json"]
    identities={str(p):digest(p) for p in files}
    if (identities[str(source/"model.safetensors")]!=SOURCE_SHA or identities[str(corpus)]!=CORPUS_SHA
            or identities[str(baseline_path)]!=BASELINE_SHA):raise ValueError("Initial frozen loss/source identities changed")
    result={"format":"toklet-metal-held-out-loss-v1","passed":False,"collection_completed":False,"quality_completed":False,"quantized":quantized,
            "scope":"Frozen 10000-target loss; task quality and latency acceptance separate","identities":identities,"runs":[],
            "engine_revision":continuation.observation(["git","rev-parse","HEAD"]),"compiler":continuation.observation(["rustc","-Vv"]),
            "build":"CARGO_TARGET_DIR=artifacts/m5-checks cargo build --offline --release --features metal --example metal_quality; default release flags",
            "platform":continuation.observation(["/usr/bin/sw_vers"]),"python":sys.version,"power_before":continuation.endpoints(),
            "memory_scope":"Admission counts shared allocations once; libproc own/owned-worker charge may omit clean/shared pages."}
    with paths[0].open("x") as summary,paths[1].open("x") as raw,paths[2].open("xb") as stderr:
        try:
            with tempfile.TemporaryDirectory(prefix="toklet-metal-loss-probes-") as directory:
                probes=quality.probe_inputs(Path(directory));result["probes"]=probes
                for probe in probes:
                    paired=[]
                    for chunk in (1,128):
                        record=child(binary,source,packed,Path(probe["path"]),probe["tokens"],quantized,chunk,raw,stderr)
                        result["runs"].append(record)
                        if not record["passed"]:raise ValueError("Metal loss probe worker failed")
                        paired.append(record["losses"])
                    probe["batch_minus_single_nll"]=quality.check_probe(paired[1],[r["negative_log_likelihood"] for r in paired[0]])
                    probe["batch_minus_unquantized_fp32_nll"]=[r["negative_log_likelihood"]-n for r,n in zip(paired[1],probe["source_nll"])]
            record=child(binary,source,packed,corpus,tokens,quantized,128,raw,stderr);result["runs"].append(record)
            if not record["passed"]:raise ValueError("Metal full loss worker failed")
            delta=record["mean_negative_log_likelihood"]-baseline["mean_negative_log_likelihood"]
            result.update(collection_completed=True,quality_completed=True,mean_negative_log_likelihood=record["mean_negative_log_likelihood"],
                          baseline_mean_negative_log_likelihood=baseline["mean_negative_log_likelihood"],increase_nats_per_token=delta,
                          quality_gate={"maximum_increase_nats_per_token":.05,"passed":delta<=.05},passed=delta<=.05)
        except BaseException as error:result["error"]=f"{type(error).__name__}: {error}"
        finally:
            result["endpoint_errors"]=[]
            try:result["identities_unchanged"]=all(digest(Path(p))==sha for p,sha in identities.items())
            except OSError as error:result["identities_unchanged"]=False;result["endpoint_errors"].append(str(error))
            try:result["power_after"]=continuation.endpoints()
            except BaseException as error:result["endpoint_errors"].append(str(error))
            if not result["identities_unchanged"] or result["endpoint_errors"]:
                result["passed"]=False;result["collection_completed"]=False;result["quality_completed"]=False
            json.dump(result,summary,indent=2,allow_nan=False);summary.write("\n")
    if not result["collection_completed"]:raise ValueError("Metal loss collection failed; see retained evidence")
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--fp16",action="store_true");args=parser.parse_args()
    result=run(args.binary.resolve(),args.output.resolve(),not args.fp16)
    if not result["passed"]:sys.exit(2)
