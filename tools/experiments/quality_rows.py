"""Bounded FP32 larger-model reference scoring; no timing acceptance.

Identical frozen corpus/block policy, existing scalar/tiled FP64 arithmetic,
and the unchanged 1e-3 short batch guard. One model per owned process.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

import metal_quality as shared
import metal_continue as continuation
import metal_engine
import native_usage

ROOT=Path(__file__).resolve().parents[2]
LIMIT=6*1024**3
POLICY=shared.POLICY
digest=continuation.digest
loss_row=shared.loss_row
chunk_done=shared.chunk_done
block_done=shared.block_done
done=shared.done
SOURCES={False:"fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe",True:"dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee"}
CONFIGS={False:"18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45",True:"98d2ff8cc47488d08a2b0b3acf4eb99ef210779b42bd48605f6b8e36acdbf670"}
TOKENIZER_SHA="c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"


def ready(event,tokens,sha,larger,chunk):
    required={"event":"ready","format":"toklet-cpu-row-loss-events-v1","backend":"cpu","cpu_projection":"tiled",
              "debug_assertions":False,"target_tokens":len(tokens)-1,"corpus_sha256":sha,"context_policy":POLICY,
              "block_targets":1024,"context_capacity":1025,"prefill_chunk_size":chunk,"logit_positions":"all_causal_rows",
              "weight_dtype":"FP32","kv_dtype":"FP32","hidden_size":1536 if larger else 896,"num_layers":28 if larger else 24,
              "vocab_size":151936,"application_allowance_bytes":64*1024**2}
    if any(type(event.get(k)) is not type(v) or event[k]!=v for k,v in required.items()):raise ValueError("CPU row admission/policy differs")
    memory=event["memory"]
    continuation.memory(memory)
    if type(memory.get("admitted_bytes")) is not int:raise ValueError("CPU row admission type differs")
    expected_weights=6174857216 if larger else 1976131072
    kv=2*(28 if larger else 24)*2*1025*(128 if larger else 64)*4
    if memory["weights_bytes"]!=expected_weights or memory["kv_bytes"]!=kv:raise ValueError("CPU row model/KV admission differs")
    for key,minimum in (("external_logits_bytes",chunk*151936*4),("serialization_bytes",151936*4)):
        if type(event.get(key)) is not int or event[key]<minimum:raise ValueError("CPU row external buffer admission differs")
    if (event["external_logits_bytes"]+event["serialization_bytes"]>required["application_allowance_bytes"]//2
            or type(event.get("memory_admitted_bytes")) is not int
            or event["memory_admitted_bytes"]!=memory["admitted_bytes"]+required["application_allowance_bytes"]
            or not 0<event["memory_admitted_bytes"]<=LIMIT):raise ValueError("CPU row memory admission differs")


def child(binary,source,token_path,tokens,larger,chunk,raw,stderr):
    command=[str(binary),str(source),str(token_path),str(chunk)]
    worker=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
    transport=None
    result={"command":command,"worker_pid":worker.pid,"passed":False,"losses":[],"chunks":[],"blocks":[],"idle_usage":[],"cleanup_errors":[],
            "transport_policy":{"request_seconds":300,"lifetime_seconds":21600,"event_limit":32768,"stdout_bytes":8*1024**2}}
    try:
        transport=metal_engine.Transport(worker,raw,stderr,request_timeout=300,lifetime_timeout=21600,event_limit=32768)
        def proceed():worker.stdin.write(b"continue\n");worker.stdin.flush()
        def usage(phase):
            left=native_usage.usage(worker.pid);right=native_usage.usage()
            if any(not 0<v["lifetime_max_phys_footprint"]<=LIMIT for v in (left,right)) or left["lifetime_max_phys_footprint"]+right["lifetime_max_phys_footprint"]>LIMIT:
                raise ValueError("CPU row loss joint native charge exceeds budget")
            result["idle_usage"].append({"phase":phase,"worker":left,"driver":right})
        # Bind the worker's exact parsed input to the driver's admitted IDs,
        # rather than hashing a path again after the worker has read it.
        token_sha=hashlib.sha256(struct.pack(f"<{len(tokens)}I",*tokens)).hexdigest()
        event=transport.read();ready(event,tokens,token_sha,larger,chunk);result["ready"]=event
        usage("ready");proceed();evaluated=0
        while True:
            event=transport.read();kind=event.get("event");block=len(result["blocks"])
            if kind=="loss":
                index=len(result["losses"])
                if index>=len(tokens)-1 or index//1024!=block:raise ValueError("CPU row loss extra/out-of-block target")
                # Every target belongs to the next completed causal chunk.
                if index%1024<evaluated or index%1024>=min(evaluated+chunk,1024):raise ValueError("CPU row loss target outside current chunk")
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
            else:raise ValueError("CPU row loss event order differs")
        if transport.finish()!=0 or transport.pending or transport.events!=2+len(result["losses"])+len(result["chunks"])+len(result["blocks"]):
            raise ValueError("CPU row loss worker exit/trailing output differs")
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



def run(binary,output,larger=True,full=False):
    paths=[output,output.with_suffix(".raw.jsonl"),output.with_suffix(".stderr")]
    if any(p.exists() for p in paths):raise ValueError("CPU row evidence destination exists")
    source=ROOT/"models"/("Qwen2.5-1.5B-Instruct" if larger else "Qwen2.5-0.5B-Instruct")
    corpus=ROOT/"benchmarks/contract-v1/held_out_loss.u32le"
    data=shared.frozen_bytes(corpus,10001*4,shared.CORPUS_SHA)
    tokens=list(struct.unpack("<10001I",data))
    if digest(source/"model.safetensors")!=SOURCES[larger] or digest(source/"tokenizer.json")!=TOKENIZER_SHA or digest(source/"config.json")!=CONFIGS[larger]:
        raise ValueError("Pinned FP32 reference source/tokenizer differs")
    files=[Path(__file__),Path(shared.__file__),Path(metal_engine.__file__),Path(native_usage.__file__),Path(continuation.__file__),
           Path(shared.quality.__file__),Path(shared.quality.prepare.__file__),binary,corpus,corpus.parent/"manifest.json",
           ROOT/"tests/fixtures/qwen-v2/manifest.json",ROOT/"artifacts/qwen-forward-v2/manifest.json",
           source/"config.json",source/"model.safetensors",source/"tokenizer.json",ROOT/"Cargo.toml",ROOT/"Cargo.lock",
           ROOT/"examples/quality_rows.rs",ROOT/"examples/support/loss.rs"]+list((ROOT/"src").rglob("*.rs"))+list((ROOT/"src").rglob("*.metal"))
    if larger:files+=[ROOT/"benchmarks/results/2026-10-13-m5-larger-preselection-v1.json",ROOT/"benchmarks/m5-larger-preselection-v1/model_api.json",ROOT/"benchmarks/m5-larger-preselection-v1/source_header.bin"]
    identities={str(p):digest(p) for p in files}
    if identities[str(source/"model.safetensors")]!=SOURCES[larger] or identities[str(corpus)]!=shared.CORPUS_SHA or identities[str(source/"tokenizer.json")]!=TOKENIZER_SHA or identities[str(source/"config.json")]!=CONFIGS[larger]:
        raise ValueError("Initial FP32 reference identities changed")
    result={"format":"toklet-cpu-row-loss-v1","passed":False,"collection_completed":False,"quality_completed":False,
            "larger":larger,"full_corpus_requested":full,"scope":"Matched FP32 reference collection; no GPU quality or performance acceptance",
            "identities":identities,"runs":[],"engine_revision":continuation.observation(["git","rev-parse","HEAD"]),
            "compiler":continuation.observation(["rustc","-Vv"]),"platform":continuation.observation(["/usr/bin/sw_vers"]),
            "build":"CARGO_TARGET_DIR=artifacts/m5-checks cargo build --offline --release --example quality_rows; default release flags",
            "python":sys.version,"power_before":continuation.endpoints(),
            "memory_scope":"One FP32 model; separately admitted external buffers; idle libproc worker+driver lifetime peaks under6GiB; clean/shared pages may be omitted."}
    with paths[0].open("x") as summary,paths[1].open("x") as raw,paths[2].open("xb") as stderr:
        try:
            with tempfile.TemporaryDirectory(prefix="toklet-cpu-row-probes-") as directory:
                probes=shared.quality.probe_inputs(Path(directory));result["probes"]=probes
                for probe in probes:
                    paired=[]
                    for chunk in (1,8):
                        record=child(binary,source,Path(probe["path"]),probe["tokens"],larger,chunk,raw,stderr)
                        result["runs"].append(record)
                        if not record["passed"]:raise ValueError("CPU row probe worker failed")
                        paired.append(record["losses"])
                    probe["batch_minus_single_nll"]=shared.quality.check_probe(paired[1],[r["negative_log_likelihood"] for r in paired[0]])
                    if not larger:
                        probe["batch_minus_frozen_scalar_nll"]=shared.quality.check_probe(paired[1],probe["source_nll"])
            result["probes_completed"]=True
            if full:
                record=child(binary,source,corpus,tokens,larger,8,raw,stderr);result["runs"].append(record)
                if not record["passed"]:raise ValueError("CPU row full reference worker failed")
                result.update(collection_completed=True,scored_targets=10000,context_policy=POLICY,
                              mean_negative_log_likelihood=record["mean_negative_log_likelihood"])
            result["passed"]=True
        except BaseException as error:result["error"]=f"{type(error).__name__}: {error}"
        finally:
            result["endpoint_errors"]=[]
            try:result["identities_unchanged"]=all(digest(Path(p))==sha for p,sha in identities.items())
            except OSError as error:result["identities_unchanged"]=False;result["endpoint_errors"].append(str(error))
            try:result["power_after"]=continuation.endpoints()
            except BaseException as error:result["endpoint_errors"].append(str(error))
            if not result["identities_unchanged"] or result["endpoint_errors"]:
                result["passed"]=False;result["collection_completed"]=False
            json.dump(result,summary,indent=2,allow_nan=False);summary.write("\n")
    if not result["passed"]:raise ValueError("CPU row reference collection failed; see retained evidence")
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--small",action="store_true");parser.add_argument("--full",action="store_true")
    args=parser.parse_args();run(args.binary.resolve(),args.output.resolve(),not args.small,args.full)
