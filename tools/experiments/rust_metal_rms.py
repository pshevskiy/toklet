"""Check GPU RMS normalization against frozen pre-GPU tiny FP16 gates."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

import rust_metal_probe as probe

ROOT = probe.ROOT
FP16 = ROOT/"tests/fixtures/fp16-v1"
FP32 = ROOT/"tests/fixtures/tiny-v1"
OUTPUT_GUARD = 0x55A1
FP16_MANIFEST_SHA = "4a416b21f52ecc89c6bbafd4c948b173cd9709c3f5b1a60cd6ce037a63964e6b"
FP32_MANIFEST_SHA = "1d40cec4f24ece5fdc7b727d3e96f41237621120908833ee6c7df83352acfdd4"


def manifest(root, expected_sha):
    with (root/"manifest.json").open("rb") as file:
        data=file.read((1 << 20)+1)
    if len(data)>1 << 20 or hashlib.sha256(data).hexdigest()!=expected_sha:
        raise ValueError("Frozen RMS manifest identity differs")
    return json.loads(data)


def payload(root,manifest,name,dtype):
    item=manifest["arrays"][name]
    size={"f16le":2,"f32le":4}[dtype]
    count=math.prod(item["shape"])
    if item["dtype"]!=dtype or item["bytes"]!=count*size or count>8192:
        raise ValueError("Frozen RMS payload metadata differs")
    with (root/item["file"]).open("rb") as file:
        data=file.read(item["bytes"]+1)
    if len(data)!=item["bytes"] or hashlib.sha256(data).hexdigest()!=item["sha256"]:
        raise ValueError("Frozen RMS payload identity differs")
    return data,item


def cases():
    half=manifest(FP16,FP16_MANIFEST_SHA)
    single=manifest(FP32,FP32_MANIFEST_SHA)
    if (half["source_manifest_sha256"]!=FP32_MANIFEST_SHA
            or half["model"]!=single["model"] or half["model"]["hidden_size"]!=32
            or half["model"]["num_layers"]!=2):
        raise ValueError("Frozen RMS source topology differs")
    inputs=[]
    expected={}
    for name in sorted(half["cases"]):
        prefix=f"cases.{name}"
        operations=[("layers.0.input_norm","embedding","layers.0.input_layernorm"),
                    ("layers.0.post_attention_norm","layers.0.attention_residual","layers.0.post_attention_layernorm"),
                    ("layers.1.input_norm","layers.0.output","layers.1.input_layernorm"),
                    ("layers.1.post_attention_norm","layers.1.attention_residual","layers.1.post_attention_layernorm"),
                    ("final_norm","layers.1.output","norm")]
        for output,input_name,weight_name in operations:
            key=f"{prefix}.{output}"
            x,item=payload(FP16,half,f"{prefix}.{input_name}","f16le")
            w,weight=payload(FP16,half,f"weights.{weight_name}","f16le")
            reference,ref_item=payload(FP32,single,key,"f32le")
            oracle,oracle_item=payload(FP16,half,key,"f16le")
            if item["shape"]!=ref_item["shape"] or item["shape"]!=oracle_item["shape"] or weight["shape"]!=[32]:
                raise ValueError("Frozen RMS input/output shape differs")
            count=math.prod(item["shape"])
            inputs.append({"id":key,"rows":item["shape"][0],"hidden":32,
                           "epsilon_bits":probe.float_bits(half["model"]["rms_norm_eps"]),
                           "input_bits":list(struct.unpack(f"<{count}H",x)),
                           "weight_bits":list(struct.unpack("<32H",w))})
            expected[key]={"reference":list(struct.unpack(f"<{count}f",reference)),
                           "oracle_bits":list(struct.unpack(f"<{count}H",oracle)),
                           "gate":oracle_item["gate_against_source_fp32"],
                           "input_sha256":item["sha256"],"weight_sha256":weight["sha256"],
                           "source_fp32_sha256":ref_item["sha256"],"oracle_fp16_sha256":oracle_item["sha256"]}
    if len(inputs)!=25 or sum(item["rows"]*item["hidden"] for item in inputs)!=7040:
        raise ValueError("Frozen RMS case inventory differs")
    # Independent mathematical checks frozen before the first target run.
    # Constant rows normalize to x/sqrt(x*x + eps); exact Half expectations
    # detect missing epsilon, Half accumulation overflow and flushed input.
    epsilon=struct.unpack("<f",struct.pack("<f",half["model"]["rms_norm_eps"]))[0]
    for label,input_bits,output_bits in (("zero",0x0000,0x0000),("maximum_half",0x7bff,0x3c00),("minimum_half_subnormal",0x0001,0x03e8)):
        key=f"manual.{label}"
        value=struct.unpack("<e",struct.pack("<H",input_bits))[0]
        reference=struct.unpack("<f",struct.pack("<f",value/math.sqrt(value*value+epsilon)))[0]
        if struct.unpack("<H",struct.pack("<e",reference))[0]!=output_bits:
            raise ValueError("Manual Half RMS mathematical oracle differs")
        inputs.append({"id":key,"rows":1,"hidden":32,"epsilon_bits":probe.float_bits(epsilon),
                       "input_bits":[input_bits]*32,"weight_bits":[0x3c00]*32})
        expected[key]={"reference":[reference]*32,"oracle_bits":[output_bits]*32,
                       "gate":single["numerical_policy"]["tolerances"]["rmsnorm"].copy(),
                       "require_exact_half_bits":True,"manual_oracle":"constant row FP64 formula then FP32 reference and one IEEE Half RNE; hardcoded Half bits independently checked",
                       "input_sha256":hashlib.sha256(struct.pack("<32H",*inputs[-1]["input_bits"])).hexdigest(),
                       "weight_sha256":hashlib.sha256(struct.pack("<32H",*inputs[-1]["weight_bits"])).hexdigest(),
                       "source_fp32_sha256":hashlib.sha256(struct.pack("<32f",*([reference]*32))).hexdigest(),
                       "oracle_fp16_sha256":hashlib.sha256(struct.pack("<32H",*([output_bits]*32))).hexdigest()}
    return inputs,expected


def validate(event,inputs,expected):
    if event.get("event")!="idle_result" or event.get("passed") is not True:
        raise ValueError("Native RMS check failed")
    result=event["result"]
    metadata=result["metadata"]
    if (result["operation"]!="rms_norm_fp16_storage_fp32_serial"
            or metadata["unified_memory"] is not True or metadata["language_version"]!="2.3"
            or metadata["math_mode"]!="safe" or metadata["math_functions"]!="precise"
            or metadata["minimum_macos_major"]!=15 or not 0<metadata["maximum_admitted_bytes"]<=probe.LIMIT
            or len(metadata["device_max_threads"])!=3 or min(metadata["device_max_threads"])<=0):
        raise ValueError("Native RMS policy differs")
    results=result["cases"]
    if [case["id"] for case in results]!=[case["id"] for case in inputs]:
        raise ValueError("Native RMS inventory differs")
    for case,source in zip(results,inputs):
        count=source["rows"]*source["hidden"]
        xbytes,wbytes=(count+7)*2,(source["hidden"]+7)*2
        width=case["threadgroup_width"]
        if (case["rows_count"]!=source["rows"] or case["hidden"]!=source["hidden"]
                or case["epsilon_bits"]!=source["epsilon_bits"] or case["guards"]!=7
                or case["output_guard_bits"]!=OUTPUT_GUARD
                or not 0<width<=min(metadata["execution_width"],metadata["pipeline_max_threads"],metadata["device_max_threads"][0])
                or case["threadgroups"]!=(source["rows"]+width-1)//width
                or case["shared_buffer_bytes"]!=xbytes*2+wbytes
                or case["host_upload_and_readback_bytes"]!=xbytes*3+wbytes*2
                or case["admitted_bytes"]!=xbytes*5+wbytes*3+probe.ALLOWANCE
                or case["allocations_reused_across_repetitions"] is not True or len(case["rows"])!=13):
            raise ValueError("Native RMS shape/launch/allocation differs")
        anchor=expected[case["id"]]
        for index,row in enumerate(case["rows"]):
            bits=row["output_bits"]
            if (row["phase"]!=("warmup" if index<3 else "sample")
                    or row["repetition"]!=(index+1 if index<3 else index-2)
                    or row["command_completed"] is not True
                    or row["immutable_input_weight_and_guards"] is not True
                    or not math.isfinite(row["encode_commit_wait_seconds"]) or row["encode_commit_wait_seconds"]<=0
                    or len(bits)!=count+7 or bits[count:]!=[OUTPUT_GUARD]*7
                    or any(type(value) is not int or not 0<=value<=65535 for value in bits)):
                raise ValueError("Native RMS completed-work/output/guards differ")
            data=struct.pack(f"<{count}H",*bits[:count])
            actual=struct.unpack(f"<{count}e",data)
            if not all(math.isfinite(value) for value in actual):raise ValueError("Nonfinite RMS output")
            fractions=[abs(value-reference)/(anchor["gate"]["atol"]+anchor["gate"]["rtol"]*abs(reference))
                       for value,reference in zip(actual,anchor["reference"])]
            maximum=max(fractions)
            if maximum>1:raise ValueError(f"Frozen RMS gate failed: {case['id']} fraction {maximum}")
            if anchor.get("require_exact_half_bits") and bits[:count]!=anchor["oracle_bits"]:
                raise ValueError(f"Manual exact Half RMS failed: {case['id']}")
            row["qualification"]={"against_source_fp32_max_gate_fraction":maximum,
                "source_fp32_max_abs_error":max(abs(value-reference) for value,reference in zip(actual,anchor["reference"])),
                "exact_independent_fp16_bits":bits[:count]==anchor["oracle_bits"],
                "output_sha256":hashlib.sha256(data).hexdigest(),"output_values":count,"output_guard_values":7}
            # Full native output is preserved in ignored raw JSONL. Compact
            # final evidence binds it by hash, checks every element beforehand.
            del row["output_bits"]
        case["anchors"]={key:value for key,value in anchor.items() if key not in ("reference","oracle_bits")}
    if metadata["maximum_admitted_bytes"]!=max(case["admitted_bytes"] for case in results):
        raise ValueError("Native RMS maximum admission differs")


def run(output,build,fixture):
    probe.check_paths(output,build)
    evidence=(output,output.with_suffix(output.suffix+".stderr.txt"),output.with_suffix(output.suffix+".native.jsonl"))
    if fixture.resolve() in {path.resolve() for path in evidence} or build.resolve()==fixture.resolve() or build.resolve() in fixture.resolve().parents:
        raise ValueError("RMS input must be outside the build directory and distinct from evidence")
    if fixture.exists():raise ValueError("Refusing to overwrite RMS inputs")
    inputs,expected=cases()
    fixture.parent.mkdir(parents=True,exist_ok=True)
    with fixture.open("x") as file:
        file.write(json.dumps({"format":"toklet-metal-rms-input-v1","cases":inputs},separators=(",",":"))+"\n")
    fixture_sha=probe.common.digest(fixture)
    context={"rms_driver_sha256":probe.common.digest(Path(__file__)),"rms_input_sha256":fixture_sha,
             "fp16_manifest_sha256":FP16_MANIFEST_SHA,
             "source_fp32_manifest_sha256":FP32_MANIFEST_SHA,
             "scope":"25 independent tiny RMS stages and three exact manual cases; fixed pre-GPU source gates unchanged. Serial FP32 reduction, Half storage. No composed decoder or performance acceptance."}
    def qualified(event):
        validate(event,inputs,expected)
        if probe.common.digest(fixture)!=fixture_sha:raise ValueError("RMS inputs changed during execution")
        if probe.common.digest(FP16/"manifest.json")!=context["fp16_manifest_sha256"] or probe.common.digest(FP32/"manifest.json")!=context["source_fp32_manifest_sha256"]:
            raise ValueError("Frozen RMS manifests changed")
    probe.run(output,build,extra_runs=[("rms_norm",["--rms-input",str(fixture.resolve())],qualified)],extra_context=context)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--build-dir",type=Path,required=True)
    parser.add_argument("--input",type=Path,required=True)
    args=parser.parse_args()
    run(args.output,args.build_dir,args.input)


if __name__=="__main__":main()
