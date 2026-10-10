"""Qualify bit-exact Half embedding gather with frozen tiny source gates."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

import rust_metal_rms as rms

ROOT=rms.ROOT
GUARD=0x55a1


def cases():
    half=rms.manifest(rms.FP16,rms.FP16_MANIFEST_SHA)
    single=rms.manifest(rms.FP32,rms.FP32_MANIFEST_SHA)
    data,weight=rms.payload(rms.FP16,half,"weights.embed_tokens","f16le")
    if weight["shape"]!=[64,32]:raise ValueError("Frozen embedding table shape differs")
    table=list(struct.unpack("<2048H",data))
    inputs=[];anchors={}
    for name,spec in sorted(half["cases"].items()):
        key=f"cases.{name}.embedding"
        tokens=spec["tokens"]
        oracle,item=rms.payload(rms.FP16,half,key,"f16le")
        reference,ref=rms.payload(rms.FP32,single,key,"f32le")
        if item["shape"]!=[len(tokens),32] or ref["shape"]!=item["shape"]:
            raise ValueError("Frozen embedding output shape differs")
        count=len(tokens)*32
        bits=list(struct.unpack(f"<{count}H",oracle))
        if bits!=[value for token in tokens for value in table[token*32:(token+1)*32]]:
            raise ValueError("Frozen embedding oracle differs from table gather")
        inputs.append({"id":key,"tokens":tokens})
        anchors[key]={"bits":bits,"reference":list(struct.unpack(f"<{count}f",reference)),
            "gate":item["gate_against_source_fp32"],"oracle_fp16_sha256":item["sha256"],
            "source_fp32_sha256":ref["sha256"],"table_fp16_sha256":weight["sha256"]}
    if len(inputs)!=5 or sum(len(item["tokens"])*32 for item in inputs)!=1408:
        raise ValueError("Frozen embedding inventory differs")
    frozen={"format":"toklet-metal-embedding-input-v1","hidden":32,"vocab":64,"table_bits":table,"cases":inputs}
    # Raw integer gather has an independent exact oracle. Every finite Half
    # also embeds exactly in FP32; no new numerical tolerance is needed.
    row0=[0,0x8000,1,0x8001,0x7bff,0xfbff,0x3c00,0xbc00]*4
    manual_table=row0+[0x3000+i for i in range(32)]+[0x3400+i for i in range(32)]+[0x3800+i for i in range(32)]
    tokens=[3,0,1,0,3,2,0]*2+[3,1,0]
    bits=[value for token in tokens for value in manual_table[token*32:(token+1)*32]]
    key="manual.bit_copy"
    anchors[key]={"bits":bits,"reference":list(struct.unpack("<544e",struct.pack("<544H",*bits))),
        "gate":None,"manual_oracle":"exact integer row gather; finite Half values have exact FP32 representations",
        "oracle_fp16_sha256":hashlib.sha256(struct.pack("<544H",*bits)).hexdigest(),
        "table_fp16_sha256":hashlib.sha256(struct.pack("<128H",*manual_table)).hexdigest()}
    manual={"format":frozen["format"],"hidden":32,"vocab":4,"table_bits":manual_table,"cases":[{"id":key,"tokens":tokens}]}
    return (frozen,manual),anchors


def validate(event,source,anchors):
    if event.get("event")!="idle_result" or event.get("passed") is not True:
        raise ValueError("Native embedding failed")
    result=event["result"];metadata=result["metadata"]
    if (result["operation"]!="embedding_fp16_bit_copy" or result["table_allocation_reused_across_cases"] is not True
            or metadata["unified_memory"] is not True or metadata["language_version"]!="2.3"
            or metadata["math_mode"]!="safe" or metadata["math_functions"]!="precise"
            or metadata["minimum_macos_major"]!=15 or not 0<metadata["maximum_admitted_bytes"]<=rms.probe.LIMIT
            or len(metadata["device_max_threads"])!=3 or min(metadata["device_max_threads"])<=0
            or [item["id"] for item in result["cases"]]!=[item["id"] for item in source["cases"]]):
        raise ValueError("Native embedding policy/inventory differs")
    table_bytes=(source["hidden"]*source["vocab"]+7)*2
    for case,item in zip(result["cases"],source["cases"]):
        rows=len(item["tokens"]);count=rows*source["hidden"];ids=(rows+7)*4;output=(count+7)*2
        width=case["threadgroup_width"]
        if (case["rows_count"]!=rows or case["hidden"]!=source["hidden"] or case["vocab"]!=source["vocab"]
                or case["guards"]!=7 or case["output_guard_bits"]!=GUARD
                or not 0<width<=min(metadata["execution_width"],metadata["pipeline_max_threads"],metadata["device_max_threads"][0])
                or case["threadgroups"]!=(count+width-1)//width or len(case["rows"])!=13
                or case["shared_buffer_bytes"]!=table_bytes+ids+output
                or case["host_upload_and_readback_bytes"]!=2*table_bytes+2*ids+output
                or case["admitted_bytes"]!=3*table_bytes+3*ids+2*output+rms.probe.ALLOWANCE
                or case["allocations_reused_across_repetitions"] is not True):
            raise ValueError("Native embedding allocation/shape/launch differs")
        anchor=anchors[item["id"]]
        for index,row in enumerate(case["rows"]):
            bits=row["output_bits"]
            if (row["phase"]!=("warmup" if index<3 else "sample") or row["repetition"]!=(index+1 if index<3 else index-2)
                    or row["command_completed"] is not True or row["immutable_table_tokens_and_guards"] is not True
                    or not math.isfinite(row["encode_commit_wait_seconds"]) or row["encode_commit_wait_seconds"]<=0
                    or len(bits)!=count+7 or any(type(value) is not int or not 0<=value<=65535 for value in bits)
                    or bits[count:]!=[GUARD]*7 or bits[:count]!=anchor["bits"]):
                raise ValueError("Embedding completed-work/exact bits/guards differ")
            payload=struct.pack(f"<{count}H",*bits[:count]);actual=struct.unpack(f"<{count}e",payload)
            errors=[abs(value-ref) for value,ref in zip(actual,anchor["reference"])]
            gate=anchor["gate"]
            fraction=max(error/(gate["atol"]+gate["rtol"]*abs(ref)) for error,ref in zip(errors,anchor["reference"])) if gate else 0.0
            if fraction>1 or gate is None and max(errors)>0:raise ValueError("Frozen embedding FP32 source gate failed")
            row["qualification"]={"exact_independent_fp16_bits":True,"source_fp32_max_abs_error":max(errors),
                "against_source_fp32_max_gate_fraction":fraction,"output_sha256":hashlib.sha256(payload).hexdigest(),"output_values":count,"output_guard_values":7}
            del row["output_bits"]
        case["anchors"]={key:value for key,value in anchor.items() if key not in ("bits","reference")}
    if metadata["maximum_admitted_bytes"]!=max(case["admitted_bytes"] for case in result["cases"]):
        raise ValueError("Native embedding maximum admission differs")


def run(output,build,directory):
    rms.probe.check_paths(output,build)
    evidence=(output,output.with_suffix(output.suffix+".stderr.txt"),output.with_suffix(output.suffix+".native.jsonl"))
    if directory.exists():raise ValueError("Refusing to overwrite embedding inputs")
    for path in (build,*evidence):
        if directory.resolve()==path.resolve() or directory.resolve() in path.resolve().parents or path.resolve() in directory.resolve().parents:
            raise ValueError("Embedding inputs/build/evidence must be disjoint")
    if any(root.resolve()==path.resolve() or root.resolve() in path.resolve().parents for root in (rms.FP16,rms.FP32) for path in (build,directory,*evidence)):
        raise ValueError("Embedding outputs must be outside source fixtures")
    embedding,anchors=cases();rms_inputs,rms_anchors=rms.cases()
    directory.mkdir(parents=True)
    jobs=[];identities={}
    payloads=[("rms_regression",{"format":"toklet-metal-rms-input-v1","cases":rms_inputs},"--rms-input",lambda event:rms.validate(event,rms_inputs,rms_anchors))]
    for label,source in zip(("embedding_frozen","embedding_manual"),embedding):
        payloads.append((label,source,"--embedding-input",lambda event,source=source:validate(event,source,anchors)))
    for label,payload,option,validator in payloads:
        path=directory/f"{label}.json"
        with path.open("x") as file:file.write(json.dumps(payload,separators=(",",":"))+"\n")
        identities[str(path.resolve())]=rms.probe.common.digest(path)
        jobs.append((label,[option,str(path.resolve())],validator))
    driver_sha=rms.probe.common.digest(Path(__file__));rms_sha=rms.probe.common.digest(Path(rms.__file__))
    previous=jobs[-1][2]
    def final(event):
        previous(event)
        rms.manifest(rms.FP16,rms.FP16_MANIFEST_SHA);rms.manifest(rms.FP32,rms.FP32_MANIFEST_SHA)
        if rms.probe.common.digest(Path(__file__))!=driver_sha or rms.probe.common.digest(Path(rms.__file__))!=rms_sha:
            raise ValueError("Embedding qualification driver changed")
        for path,sha in identities.items():
            if rms.probe.common.digest(Path(path))!=sha:raise ValueError("Embedding inputs changed")
    jobs[-1]=(jobs[-1][0],jobs[-1][1],final)
    context={"embedding_driver_sha256":driver_sha,"rms_driver_sha256":rms_sha,"input_sha256":identities,
        "fp16_manifest_sha256":rms.FP16_MANIFEST_SHA,"source_fp32_manifest_sha256":rms.FP32_MANIFEST_SHA,
        "scope":"Five frozen tiny embedding cases and exact manual signedzero/subnormal/maxfinite/last-row/repeated-ID gather. Width32 only; no Qwen table, decoder, cache, performance or M4 completion."}
    rms.probe.run(output,build,extra_runs=jobs,extra_context=context)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True);parser.add_argument("--build-dir",type=Path,required=True)
    parser.add_argument("--input-dir",type=Path,required=True);args=parser.parse_args();run(args.output,args.build_dir,args.input_dir)


if __name__=="__main__":main()
