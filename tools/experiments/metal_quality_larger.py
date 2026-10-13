"""Matched larger Metal loss collection under the frozen policy; no timing claims."""
import argparse
import json
from pathlib import Path
import struct
import sys
import tempfile

import metal_quality as shared
import quality_rows

ROOT=Path(__file__).resolve().parents[2]
BASELINE_SHA="5967fb03c60420556bfbf3d3893269b27c3bb54df4328de19fc667074dc0da6b"
BASELINE_REL="benchmarks/results/2026-10-13-larger-fp32-held-out-loss-v1.json"
BASELINE_MEAN=2.485958031019153
SOURCE_SHA=quality_rows.SOURCES[True]
CONFIG_SHA=quality_rows.CONFIGS[True]
TOKENIZER_SHA=quality_rows.TOKENIZER_SHA
PACKED_SHA="236e2a2071ad90967008e81b3af262275fc7c81307f303eab577f8681bc6c8c3"
digest=shared.digest


def ready(event,tokens,sha,quantized,chunk):
    shared.ready(event,tokens,sha,quantized,chunk)
    expected={"hidden_size":1536,"num_layers":28,"vocab_size":151936,
              "projection_mode":"Q4Compensated" if quantized else "Original",
              "normalization_mode":"Compensated" if quantized else "Original"}
    if any(type(event.get(k)) is not type(v) or event[k]!=v for k,v in expected.items()):
        raise ValueError("Larger loss geometry/numerical mode differs")
    if event["memory"]["weights_bytes"]!=(1223992516 if quantized else 3087433340):
        raise ValueError("Larger loss resident weights differ")


def reference(value,tokens):
    expected={"format":"toklet-cpu-row-loss-v1","passed":True,"collection_completed":True,
              "probes_completed":True,"larger":True,"scored_targets":10000,
              "context_policy":shared.POLICY,"mean_negative_log_likelihood":BASELINE_MEAN,
              "identities_unchanged":True,"endpoint_errors":[]}
    if any(type(value.get(k)) is not type(v) or value[k]!=v for k,v in expected.items()):
        raise ValueError("Matched larger FP32 baseline incomplete or changed")
    sources=[sha for path,sha in value["identities"].items() if path.endswith("/models/Qwen2.5-1.5B-Instruct/model.safetensors")]
    if sources!=[SOURCE_SHA] or len(value["runs"])!=5 or len(value["probes"])!=2:
        raise ValueError("Matched larger FP32 baseline source/inventory differs")
    if any(not r["passed"] or r["exit_code"]!=0 or r["cleanup_errors"] for r in value["runs"]):
        raise ValueError("Matched larger FP32 baseline workers incomplete")
    full=value["runs"][-1]
    cpu={"backend":"cpu","cpu_projection":"tiled","weight_dtype":"FP32","kv_dtype":"FP32",
         "hidden_size":1536,"num_layers":28,"vocab_size":151936,"context_policy":shared.POLICY}
    if any(type(full["ready"].get(k)) is not type(v) or full["ready"][k]!=v for k,v in cpu.items()):
        raise ValueError("Matched larger FP32 baseline numerical configuration differs")
    if len(full["losses"])!=10000 or full["ready"]["corpus_sha256"]!=shared.CORPUS_SHA:
        raise ValueError("Matched larger FP32 baseline corpus coverage differs")
    for i,row in enumerate(full["losses"]):shared.loss_row(row,i,tokens)
    mean=shared.done(full["done"],tokens,full["losses"],len(full["blocks"]),full["blocks"][-1]["evaluated_tokens"])
    if mean!=BASELINE_MEAN:raise ValueError("Matched larger FP32 baseline aggregation differs")
    return value


def check_initial_pins(identities,source,corpus,baseline_path,packed,quantized,contract_sha):
    pins={source/"model.safetensors":SOURCE_SHA,source/"config.json":CONFIG_SHA,
          source/"tokenizer.json":TOKENIZER_SHA,corpus:shared.CORPUS_SHA,
          corpus.parent/"manifest.json":contract_sha,baseline_path:BASELINE_SHA}
    if any(identities.get(str(p))!=sha for p,sha in pins.items()):
        raise ValueError("Initial larger loss identities changed")
    if quantized and identities.get(str(packed/"manifest.json"))!=PACKED_SHA:
        raise ValueError("Frozen numerically qualified larger Q4 artifact changed")


def run(binary,output,full=False,quantized=True):
    paths=[output,output.with_suffix(".raw.jsonl"),output.with_suffix(".stderr")]
    if any(p.exists() for p in paths):raise ValueError("Larger Metal evidence destination exists")
    source=ROOT/"models/Qwen2.5-1.5B-Instruct"
    packed=ROOT/"artifacts/m5-larger-q4-v1"
    corpus=ROOT/"benchmarks/contract-v1/held_out_loss.u32le"
    data=shared.frozen_bytes(corpus,10001*4,shared.CORPUS_SHA)
    tokens=list(struct.unpack("<10001I",data))
    baseline_path=ROOT/BASELINE_REL
    baseline=reference(json.loads(shared.frozen_bytes(baseline_path,8*1024**2,BASELINE_SHA)),tokens)
    contract_shas=[sha for path,sha in baseline["identities"].items() if path.endswith("/benchmarks/contract-v1/manifest.json")]
    if len(contract_shas)!=1:raise ValueError("Matched larger contract identity differs")
    contract=json.loads(shared.frozen_bytes(corpus.parent/"manifest.json",2*1024**2,contract_shas[0]))
    if contract["quality"]["mean_nll_increase_limit_nats_per_token"]!=.05:
        raise ValueError("Frozen larger quality gate changed")
    files=[Path(__file__),Path(shared.__file__),Path(quality_rows.__file__),Path(shared.metal_engine.__file__),
           Path(shared.native_usage.__file__),Path(shared.continuation.__file__),Path(shared.quality.__file__),Path(shared.quality.prepare.__file__),binary,
           corpus,corpus.parent/"manifest.json",baseline_path,source/"config.json",source/"model.safetensors",source/"tokenizer.json",
           ROOT/"Cargo.toml",ROOT/"Cargo.lock",ROOT/"examples/metal_quality.rs",ROOT/"examples/support/loss.rs",
           ROOT/"tests/fixtures/qwen-larger-q4-v1/manifest.json",ROOT/"tests/fixtures/qwen-larger-half-v1/manifest.json",
           ROOT/"tests/fixtures/qwen-v2/manifest.json",ROOT/"artifacts/qwen-forward-v2/manifest.json"]
    files+=list((ROOT/"src").rglob("*.rs"))+list((ROOT/"src").rglob("*.metal"))
    if quantized:files+=list(packed.glob("*.q4"))+[packed/"manifest.json"]
    identities={str(p):digest(p) for p in files}
    check_initial_pins(identities,source,corpus,baseline_path,packed,quantized,contract_shas[0])
    result={"format":"toklet-larger-metal-held-out-loss-v1","passed":False,"probes_completed":False,"collection_completed":False,"quality_completed":False,
            "full_corpus_requested":full,"quantized":quantized,"scope":"Matched larger loss only; task quality and latency separate",
            "identities":identities,"runs":[],"baseline_mean_negative_log_likelihood":BASELINE_MEAN,
            "engine_revision":shared.continuation.observation(["git","rev-parse","HEAD"]),"compiler":shared.continuation.observation(["rustc","-Vv"]),
            "build":"CARGO_TARGET_DIR=artifacts/m5-checks cargo build --offline --release --features metal --example metal_quality; default release flags",
            "platform":shared.continuation.observation(["/usr/bin/sw_vers"]),"python":sys.version,"power_before":shared.continuation.endpoints(),
            "memory_scope":"One resident model; admission and idle worker+driver lifetime native peaks under6GiB; shared physical allocations counted once in admission."}
    with paths[0].open("x") as summary,paths[1].open("x") as raw,paths[2].open("xb") as stderr:
        try:
            with tempfile.TemporaryDirectory(prefix="toklet-larger-metal-loss-") as directory:
                probes=shared.quality.probe_inputs(Path(directory));result["probes"]=probes
                if [(p["name"],p["tokens"],p["corpus_sha256"]) for p in probes]!=[(p["name"],p["tokens"],p["corpus_sha256"]) for p in baseline["probes"]]:
                    raise ValueError("Matched larger probes differ")
                for i,probe in enumerate(probes):
                    # These matched CPU rows replace the small model's informational
                    # source_nll from the shared token-input generator.
                    probe["matched_fp32_nll"]=[r["negative_log_likelihood"] for r in baseline["runs"][2*i+1]["losses"]]
                    if len(probe["matched_fp32_nll"])!=len(probe["tokens"])-1:raise ValueError("Matched probe row coverage differs")
                    for key in ("source_nll","source_logit_path","source_logit_sha256"):del probe[key]
                    paired=[]
                    for chunk in (1,128):
                        record=shared.child(binary,source,packed,Path(probe["path"]),probe["tokens"],quantized,chunk,raw,stderr,admission=ready,extra_args=("larger",))
                        result["runs"].append(record)
                        if not record["passed"]:raise ValueError("Larger Metal loss probe worker failed")
                        paired.append(record["losses"])
                    probe["batch_minus_single_nll"]=shared.quality.check_probe(paired[1],[r["negative_log_likelihood"] for r in paired[0]])
                    probe["batch_minus_matched_fp32_nll"]=[r["negative_log_likelihood"]-n for r,n in zip(paired[1],probe["matched_fp32_nll"])]
            result["probes_completed"]=True;result["passed"]=True
            if full:
                record=shared.child(binary,source,packed,corpus,tokens,quantized,128,raw,stderr,admission=ready,extra_args=("larger",))
                result["runs"].append(record)
                if not record["passed"]:raise ValueError("Larger Metal full loss worker failed")
                mean=record["mean_negative_log_likelihood"];delta=mean-BASELINE_MEAN
                result.update(collection_completed=True,quality_completed=True,scored_targets=10000,context_policy=shared.POLICY,
                              mean_negative_log_likelihood=mean,increase_nats_per_token=delta,
                              quality_gate={"maximum_increase_nats_per_token":.05,"passed":delta<=.05},passed=delta<=.05)
        except BaseException as error:
            result["passed"]=False;result["error"]=f"{type(error).__name__}: {error}"
        finally:
            result["endpoint_errors"]=[]
            try:result["identities_unchanged"]=all(digest(Path(p))==sha for p,sha in identities.items())
            except OSError as error:result["identities_unchanged"]=False;result["endpoint_errors"].append(str(error))
            try:result["power_after"]=shared.continuation.endpoints()
            except BaseException as error:result["endpoint_errors"].append(str(error))
            if not result["identities_unchanged"] or result["endpoint_errors"]:
                result["passed"]=False;result["collection_completed"]=False;result["quality_completed"]=False
            json.dump(result,summary,indent=2,allow_nan=False);summary.write("\n")
    if (not result["probes_completed"] or (full and not result["collection_completed"])
            or not result["identities_unchanged"] or result["endpoint_errors"]):
        raise ValueError("Larger Metal loss collection failed; see retained evidence")
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--full",action="store_true");parser.add_argument("--fp16",action="store_true")
    args=parser.parse_args();result=run(args.binary.resolve(),args.output.resolve(),args.full,not args.fp16)
    if not result["passed"]:sys.exit(2)
