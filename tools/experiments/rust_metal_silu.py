"""Qualify stable FP32 SiLU with Half storage under frozen tiny source gates."""
import argparse
import hashlib
import math
from pathlib import Path
import struct
import json

import rust_metal_residual as residual

rms=residual.rms
ROOT=rms.ROOT
GUARD=0x55a1


def cases():
    half=rms.manifest(rms.FP16,rms.FP16_MANIFEST_SHA)
    single=rms.manifest(rms.FP32,rms.FP32_MANIFEST_SHA)
    if (half["source_manifest_sha256"]!=rms.FP32_MANIFEST_SHA or half["model"]!=single["model"]
            or half["model"]["num_layers"]!=2 or half["model"]["intermediate_size"]!=64):
        raise ValueError("Frozen SiLU topology differs")
    inputs=[];anchors={}
    for name,spec in sorted(half["cases"].items()):
        rows=len(spec["tokens"]);count=rows*64
        for layer in range(2):
            prefix=f"cases.{name}.layers.{layer}";key=f"{prefix}.gate_silu"
            source,source_item=rms.payload(rms.FP16,half,f"{prefix}.gate_projection","f16le")
            oracle,item=rms.payload(rms.FP16,half,key,"f16le")
            reference,ref=rms.payload(rms.FP32,single,key,"f32le")
            if any(record["shape"]!=[rows,64] for record in (source_item,item,ref)):
                raise ValueError("Frozen SiLU shape differs")
            inputs.append({"id":key,"rows":rows,"width":64,"input_bits":list(struct.unpack(f"<{count}H",source))})
            anchors[key]={"bits":list(struct.unpack(f"<{count}H",oracle)),
                "reference":list(struct.unpack(f"<{count}f",reference)),"gate":item["gate_against_source_fp32"],
                "input_fp16_sha256":source_item["sha256"],"oracle_fp16_sha256":item["sha256"],"source_fp32_sha256":ref["sha256"]}
    if len(inputs)!=10 or sum(item["rows"]*64 for item in inputs)!=5632:
        raise ValueError("Frozen SiLU inventory differs")
    # Exact manual results fixed before GPU evaluation. Subnormal bits4 produce
    # bits2 after Half rounding, away from a Half tie; +/-20 test both sigmoid
    # branches and negative tiny output. Maximum finite inputs expose an unsafe
    # exp(-x) overflow on the negative branch, and zero signs must survive.
    edges=[(0,0),(0x8000,0x8000),(0x7bff,0x7bff),(0xfbff,0x8000),
           (4,2),(0x8004,0x8002),(0x4d00,0x4d00),(0xcd00,0x8001)]
    for input_bits,expected in edges:
        x=struct.unpack("<e",struct.pack("<H",input_bits))[0];e=math.exp(-abs(x))
        sigmoid=1/(1+e) if x>=0 else e/(1+e)
        actual=struct.unpack("<H",struct.pack("<e",x*sigmoid))[0]
        if actual!=expected:raise ValueError("Independent manual SiLU rounding oracle differs")
    count=17*64;edge=[edges[i%len(edges)] for i in range(count)];key="manual.stable_silu"
    bits=[expected for _,expected in edge]
    inputs.append({"id":key,"rows":17,"width":64,"input_bits":[source for source,_ in edge]})
    anchors[key]={"bits":bits,"reference":list(struct.unpack(f"<{count}e",struct.pack(f"<{count}H",*bits))),"gate":None,
        "manual_oracle":"Preselected exact Half signedzero, subnormal, maxfinite and +/-20 stable sigmoid outputs; independently checked with stdlib FP64 and IEEE Half RNE",
        "oracle_fp16_sha256":hashlib.sha256(struct.pack(f"<{count}H",*bits)).hexdigest()}
    return inputs,anchors


def validate(event,inputs,anchors):
    if event.get("event")!="idle_result" or event.get("passed") is not True:raise ValueError("Native SiLU check failed")
    result=event["result"];metadata=result["metadata"]
    if (result["operation"]!="silu_fp16_storage_stable_fp32" or metadata["unified_memory"] is not True
            or metadata["language_version"]!="2.3" or metadata["math_mode"]!="safe" or metadata["math_functions"]!="precise"
            or metadata["minimum_macos_major"]!=15 or not 0<metadata["maximum_admitted_bytes"]<=rms.probe.LIMIT
            or len(metadata["device_max_threads"])!=3 or min(metadata["device_max_threads"])<=0
            or [case["id"] for case in result["cases"]]!=[item["id"] for item in inputs]):
        raise ValueError("Native SiLU policy/inventory differs")
    for case,item in zip(result["cases"],inputs):
        count=item["rows"]*item["width"];size=(count+7)*2;width=case["threadgroup_width"]
        if (case["rows_count"]!=item["rows"] or case["width"]!=item["width"] or case["guards"]!=7
                or case["output_guard_bits"]!=GUARD or case["allocations_reused_across_repetitions"] is not True
                or len(case["rows"])!=13 or not 0<width<=min(metadata["execution_width"],metadata["pipeline_max_threads"],metadata["device_max_threads"][0])
                or case["threadgroups"]!=(count+width-1)//width or case["shared_buffer_bytes"]!=size*2
                or case["host_input_and_readback_bytes"]!=size*3 or case["admitted_bytes"]!=size*5+rms.probe.ALLOWANCE):
            raise ValueError("Native SiLU shape/ownership/launch/admission differs")
        anchor=anchors[item["id"]]
        for index,row in enumerate(case["rows"]):
            bits=row["output_bits"]
            if (row["phase"]!=("warmup" if index<3 else "sample") or row["repetition"]!=(index+1 if index<3 else index-2)
                    or row["command_completed"] is not True or row["immutable_input_and_guards"] is not True
                    or not math.isfinite(row["encode_commit_wait_seconds"]) or row["encode_commit_wait_seconds"]<=0
                    or len(bits)!=count+7 or any(type(value) is not int or not 0<=value<=65535 for value in bits)
                    or bits[count:]!=[GUARD]*7 or any(value & 0x7c00==0x7c00 for value in bits[:count])):
                raise ValueError("SiLU completed-work/finite bits/guards differ")
            payload=struct.pack(f"<{count}H",*bits[:count]);actual=struct.unpack(f"<{count}e",payload)
            errors=[abs(value-ref) for value,ref in zip(actual,anchor["reference"])];gate=anchor["gate"]
            fraction=max(error/(gate["atol"]+gate["rtol"]*abs(ref)) for error,ref in zip(errors,anchor["reference"])) if gate else 0.0
            exact=bits[:count]==anchor["bits"]
            if fraction>1 or gate is None and not exact:raise ValueError("Frozen SiLU FP32 source gate or exact manual bits failed")
            row["qualification"]={"exact_independent_fp16_bits":exact,
                "independent_fp16_differing_values":sum(a!=b for a,b in zip(bits[:count],anchor["bits"])),
                "source_fp32_max_abs_error":max(errors),"against_source_fp32_max_gate_fraction":fraction,
                "output_sha256":hashlib.sha256(payload).hexdigest(),"output_values":count,"output_guard_values":7}
            del row["output_bits"]
        case["anchors"]={key:value for key,value in anchor.items() if key not in ("bits","reference")}
    if metadata["maximum_admitted_bytes"]!=max(case["admitted_bytes"] for case in result["cases"]):
        raise ValueError("Native SiLU maximum admission differs")


def run(output,build,directory):
    rms.probe.check_paths(output,build)
    evidence=(output,output.with_suffix(output.suffix+".stderr.txt"),output.with_suffix(output.suffix+".native.jsonl"))
    if directory.exists():raise ValueError("Refusing to overwrite SiLU inputs")
    for path in (build,*evidence):
        if directory.resolve()==path.resolve() or directory.resolve() in path.resolve().parents or path.resolve() in directory.resolve().parents:
            raise ValueError("SiLU inputs/build/evidence must be disjoint")
    if any(root.resolve()==path.resolve() or root.resolve() in path.resolve().parents for root in (rms.FP16,rms.FP32) for path in (build,directory,*evidence)):
        raise ValueError("SiLU outputs must be outside source fixtures")
    inputs,anchors=cases();rms_inputs,rms_anchors=rms.cases()
    embedding=residual.embedding;embedding_inputs,embedding_anchors=embedding.cases()
    residual_inputs,residual_anchors,overflow=residual.cases()
    directory.mkdir(parents=True);jobs=[];identities={}
    payloads=[("rms_regression",{"format":"toklet-metal-rms-input-v1","cases":rms_inputs},"--rms-input",lambda event:rms.validate(event,rms_inputs,rms_anchors))]
    for label,source in zip(("embedding_frozen_regression","embedding_manual_regression"),embedding_inputs):
        payloads.append((label,source,"--embedding-input",lambda event,source=source:embedding.validate(event,source,embedding_anchors)))
    payloads.extend([
        ("residual_regression",{"format":"toklet-metal-residual-input-v1","cases":residual_inputs},"--residual-input",lambda event:residual.validate(event,residual_inputs,residual_anchors)),
        ("residual_overflow_regression",{"format":"toklet-metal-residual-input-v1","cases":[overflow]},"--residual-input",residual.overflow_rejected),
        ("silu_frozen_manual",{"format":"toklet-metal-silu-input-v1","cases":inputs},"--silu-input",lambda event:validate(event,inputs,anchors))])
    for label,payload,option,validator in payloads:
        path=directory/f"{label}.json"
        with path.open("x") as file:file.write(json.dumps(payload,separators=(",",":"))+"\n")
        identities[str(path.resolve())]=rms.probe.common.digest(path);jobs.append((label,[option,str(path.resolve())],validator))
    drivers={str(Path(module.__file__).resolve()):rms.probe.common.digest(Path(module.__file__)) for module in (rms,embedding,residual)}
    drivers[str(Path(__file__).resolve())]=rms.probe.common.digest(Path(__file__))
    previous=jobs[-1][2]
    def final(event):
        previous(event)
        rms.manifest(rms.FP16,rms.FP16_MANIFEST_SHA);rms.manifest(rms.FP32,rms.FP32_MANIFEST_SHA)
        for path,sha in {**identities,**drivers}.items():
            if rms.probe.common.digest(Path(path))!=sha:raise ValueError("SiLU input/driver identity changed")
    jobs[-1]=(jobs[-1][0],jobs[-1][1],final)
    context={"qualification_driver_sha256":drivers,"input_sha256":identities,"fp16_manifest_sha256":rms.FP16_MANIFEST_SHA,
        "source_fp32_manifest_sha256":rms.FP32_MANIFEST_SHA,
        "scope":"10 frozen tiny SiLU stages and exact manual stable/signedzero/subnormal/saturation edges; width64. Unchanged source gates, exact frozen Half bits diagnostic. No Qwen, MLP composition, cache, performance or M4 completion."}
    rms.probe.run(output,build,extra_runs=jobs,extra_context=context)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True);parser.add_argument("--build-dir",type=Path,required=True)
    parser.add_argument("--input-dir",type=Path,required=True);args=parser.parse_args();run(args.output,args.build_dir,args.input_dir)


if __name__=="__main__":main()
