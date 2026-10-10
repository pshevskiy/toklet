"""Qualify in-place Half residuals under frozen tiny FP32-source gates."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

import rust_metal_embedding as embedding

rms=embedding.rms
ROOT=rms.ROOT
GUARD=0x51aa


def half_sum(a,b):
    """Independent exact Half operands -> FP32 addition -> one Half RNE."""
    left,right=struct.unpack("<2e",struct.pack("<2H",a,b))
    single=struct.unpack("<f",struct.pack("<f",left+right))[0]
    return struct.unpack("<H",struct.pack("<e",single))[0]


def cases():
    half=rms.manifest(rms.FP16,rms.FP16_MANIFEST_SHA)
    single=rms.manifest(rms.FP32,rms.FP32_MANIFEST_SHA)
    if half["model"]["num_layers"]!=2 or half["model"]["hidden_size"]!=32:
        raise ValueError("Frozen residual topology differs")
    inputs=[];anchors={}
    for name,spec in sorted(half["cases"].items()):
        prefix=f"cases.{name}";rows=len(spec["tokens"]);count=rows*32
        for layer in range(2):
            block=f"{prefix}.layers.{layer}"
            previous=f"{prefix}.embedding" if layer==0 else f"{prefix}.layers.{layer-1}.output"
            for state_name,delta_name,key in ((previous,f"{block}.attention_projection",f"{block}.attention_residual"),
                    (f"{block}.attention_residual",f"{block}.down_projection",f"{block}.output")):
                payloads=[rms.payload(rms.FP16,half,n,"f16le") for n in (state_name,delta_name,key)]
                reference,ref=rms.payload(rms.FP32,single,key,"f32le")
                if ref["shape"]!=[rows,32] or any(item["shape"]!=[rows,32] for _,item in payloads):
                    raise ValueError("Frozen residual shape differs")
                (state,state_item),(delta,delta_item),(oracle,item)=payloads
                a=list(struct.unpack(f"<{count}H",state));b=list(struct.unpack(f"<{count}H",delta));bits=list(struct.unpack(f"<{count}H",oracle))
                if bits!=[half_sum(left,right) for left,right in zip(a,b)]:
                    raise ValueError("Frozen residual oracle differs from independent Half sum")
                inputs.append({"id":key,"rows":rows,"hidden":32,"state_bits":a,"delta_bits":b})
                anchors[key]={"bits":bits,"reference":list(struct.unpack(f"<{count}f",reference)),"gate":item["gate_against_source_fp32"],
                    "state_fp16_sha256":state_item["sha256"],"delta_fp16_sha256":delta_item["sha256"],
                    "oracle_fp16_sha256":item["sha256"],"source_fp32_sha256":ref["sha256"]}
    if len(inputs)!=20 or sum(item["rows"]*32 for item in inputs)!=5632:
        raise ValueError("Frozen residual inventory differs")
    # Expected bits fixed before target evaluation; hardcoded ties independently
    # rechecked via the stdlib IEEE conversions. Opposite-sign cancellation is
    # positive zero, while negative-zero + negative-zero stays negative.
    edges=[(0,0,0),(0x8000,0x8000,0x8000),(0x8000,0,0),
        (0x3c00,0x1000,0x3c00),(0x3c01,0x1000,0x3c02),
        (0xbc00,0x9000,0xbc00),(0xbc01,0x9000,0xbc02),
        (1,1,2),(0x8001,0x8001,0x8002),(0x8001,1,0),
        (0x7bff,0xfbff,0),(0x77ff,0x77ff,0x7bff),
        (0x7bff,0,0x7bff),(0xfbff,0,0xfbff),(0x3c00,0xbc00,0),(0x0400,0x8001,0x03ff)]
    if any(half_sum(a,b)!=value for a,b,value in edges):
        raise ValueError("Manual residual Half RNE oracle differs")
    count=17*32;edge=[edges[i%len(edges)] for i in range(count)];bits=[value for _,_,value in edge];key="manual.rounding"
    manual={"id":key,"rows":17,"hidden":32,"state_bits":[a for a,_,_ in edge],"delta_bits":[b for _,b,_ in edge]}
    anchors[key]={"bits":bits,"reference":list(struct.unpack(f"<{count}e",struct.pack(f"<{count}H",*bits))),"gate":None,
        "manual_oracle":"Hardcoded Half RNE ties, signedzero, cancellation, subnormal/normal boundary, finite max; independently rechecked with FP32 add and one IEEE Half RNE",
        "oracle_fp16_sha256":hashlib.sha256(struct.pack(f"<{count}H",*bits)).hexdigest()}
    inputs.append(manual)
    overflow={"id":"manual.overflow","rows":1,"hidden":32,"state_bits":[0x7bff]*32,"delta_bits":[0x7bff]*32}
    return inputs,anchors,overflow


def validate(event,inputs,anchors):
    if event.get("event")!="idle_result" or event.get("passed") is not True:
        raise ValueError("Native residual check failed")
    result=event["result"];metadata=result["metadata"]
    if (result["operation"]!="residual_in_place_fp16_storage_fp32_add"
            or metadata["unified_memory"] is not True or metadata["language_version"]!="2.3"
            or metadata["math_mode"]!="safe" or metadata["math_functions"]!="precise"
            or metadata["minimum_macos_major"]!=15 or not 0<metadata["maximum_admitted_bytes"]<=rms.probe.LIMIT
            or len(metadata["device_max_threads"])!=3 or min(metadata["device_max_threads"])<=0
            or [case["id"] for case in result["cases"]]!=[item["id"] for item in inputs]):
        raise ValueError("Native residual policy/inventory differs")
    for case,item in zip(result["cases"],inputs):
        count=item["rows"]*item["hidden"];size=(count+7)*2;width=case["threadgroup_width"]
        if (case["rows_count"]!=item["rows"] or case["hidden"]!=item["hidden"] or case["guards"]!=7
                or case["output_guard_bits"]!=GUARD or case["in_place_state"] is not True
                or case["baseline_restored_before_each_repetition"] is not True
                or case["allocations_reused_across_repetitions"] is not True or len(case["rows"])!=13
                or not 0<width<=min(metadata["execution_width"],metadata["pipeline_max_threads"],metadata["device_max_threads"][0])
                or case["threadgroups"]!=(count+width-1)//width or case["shared_buffer_bytes"]!=size*2
                or case["host_baseline_and_readback_bytes"]!=size*4 or case["admitted_bytes"]!=size*6+rms.probe.ALLOWANCE):
            raise ValueError("Native residual shape/ownership/launch/admission differs")
        anchor=anchors[item["id"]]
        for index,row in enumerate(case["rows"]):
            bits=row["output_bits"]
            if (row["phase"]!=("warmup" if index<3 else "sample") or row["repetition"]!=(index+1 if index<3 else index-2)
                    or row["command_completed"] is not True or row["immutable_delta_and_state_guards"] is not True
                    or not math.isfinite(row["encode_commit_wait_seconds"]) or row["encode_commit_wait_seconds"]<=0
                    or len(bits)!=count+7 or any(type(value) is not int or not 0<=value<=65535 for value in bits)
                    or bits[count:]!=[GUARD]*7 or bits[:count]!=anchor["bits"]):
                raise ValueError("Residual completed-work/exact bits/guards differ")
            payload=struct.pack(f"<{count}H",*bits[:count]);actual=struct.unpack(f"<{count}e",payload)
            errors=[abs(value-ref) for value,ref in zip(actual,anchor["reference"])];gate=anchor["gate"]
            fraction=max(error/(gate["atol"]+gate["rtol"]*abs(ref)) for error,ref in zip(errors,anchor["reference"])) if gate else 0.0
            if fraction>1 or gate is None and max(errors)>0:raise ValueError("Frozen residual FP32 source gate failed")
            row["qualification"]={"exact_independent_fp16_bits":True,"source_fp32_max_abs_error":max(errors),
                "against_source_fp32_max_gate_fraction":fraction,"output_sha256":hashlib.sha256(payload).hexdigest(),"output_values":count,"output_guard_values":7}
            del row["output_bits"]
        case["anchors"]={key:value for key,value in anchor.items() if key not in ("bits","reference")}
    if metadata["maximum_admitted_bytes"]!=max(case["admitted_bytes"] for case in result["cases"]):
        raise ValueError("Native residual maximum admission differs")


def overflow_rejected(event):
    if (event.get("event")!="idle_result" or event.get("passed") is not False
            or event.get("error",{}).get("stage")!="validation"
            or event["error"]["message"]!="completed residual contains nonfinite Half; in-place result invalid and discarded"):
        raise ValueError("Completed Half overflow rejection was not preserved")
    return False # Negative fixture; base driver additionally requires exit1.


def run(output,build,directory):
    rms.probe.check_paths(output,build)
    evidence=(output,output.with_suffix(output.suffix+".stderr.txt"),output.with_suffix(output.suffix+".native.jsonl"))
    if directory.exists():raise ValueError("Refusing to overwrite residual inputs")
    for path in (build,*evidence):
        if directory.resolve()==path.resolve() or directory.resolve() in path.resolve().parents or path.resolve() in directory.resolve().parents:
            raise ValueError("Residual inputs/build/evidence must be disjoint")
    if any(root.resolve()==path.resolve() or root.resolve() in path.resolve().parents for root in (rms.FP16,rms.FP32) for path in (build,directory,*evidence)):
        raise ValueError("Residual outputs must be outside source fixtures")
    inputs,anchors,overflow=cases();rms_inputs,rms_anchors=rms.cases();embedding_inputs,embedding_anchors=embedding.cases()
    directory.mkdir(parents=True);jobs=[];identities={}
    payloads=[("rms_regression",{"format":"toklet-metal-rms-input-v1","cases":rms_inputs},"--rms-input",lambda event:rms.validate(event,rms_inputs,rms_anchors))]
    for label,source in zip(("embedding_frozen_regression","embedding_manual_regression"),embedding_inputs):
        payloads.append((label,source,"--embedding-input",lambda event,source=source:embedding.validate(event,source,embedding_anchors)))
    payloads.extend([("residual_frozen_manual",{"format":"toklet-metal-residual-input-v1","cases":inputs},"--residual-input",lambda event:validate(event,inputs,anchors)),
        ("residual_overflow",{"format":"toklet-metal-residual-input-v1","cases":[overflow]},"--residual-input",overflow_rejected)])
    for label,payload,option,validator in payloads:
        path=directory/f"{label}.json"
        with path.open("x") as file:file.write(json.dumps(payload,separators=(",",":"))+"\n")
        identities[str(path.resolve())]=rms.probe.common.digest(path);jobs.append((label,[option,str(path.resolve())],validator))
    drivers={str(Path(module.__file__).resolve()):rms.probe.common.digest(Path(module.__file__)) for module in (rms,embedding)}
    drivers[str(Path(__file__).resolve())]=rms.probe.common.digest(Path(__file__))
    previous=jobs[-1][2]
    def final(event):
        outcome=previous(event)
        rms.manifest(rms.FP16,rms.FP16_MANIFEST_SHA);rms.manifest(rms.FP32,rms.FP32_MANIFEST_SHA)
        for path,sha in {**identities,**drivers}.items():
            if rms.probe.common.digest(Path(path))!=sha:raise ValueError("Residual input/driver identity changed")
        return outcome
    jobs[-1]=(jobs[-1][0],jobs[-1][1],final)
    context={"qualification_driver_sha256":drivers,"input_sha256":identities,"fp16_manifest_sha256":rms.FP16_MANIFEST_SHA,
        "source_fp32_manifest_sha256":rms.FP32_MANIFEST_SHA,"scope":"20 frozen tiny residuals plus exact manual Half RNE edges; width32. Completed overflow is invalid and discarded, without rollback or session recovery qualification. No Qwen, composition, cache, performance or M4 completion."}
    rms.probe.run(output,build,extra_runs=jobs,extra_context=context)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True);parser.add_argument("--build-dir",type=Path,required=True)
    parser.add_argument("--input-dir",type=Path,required=True);args=parser.parse_args();run(args.output,args.build_dir,args.input_dir)


if __name__=="__main__":main()
