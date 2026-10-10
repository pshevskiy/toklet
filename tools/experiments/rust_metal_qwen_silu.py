"""Qualify saved Qwen SiLU rows without changing frozen gates or reading weights."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

import rust_metal_qwen_rms as qwen
import rust_metal_silu as silu

ROOT=qwen.ROOT
WIDTH=4864
LAYERS=(0,1,23)


def checked_references():
    half,single,_=qwen.checked_manifests()
    with qwen.CONFIG.open("rb") as file:data=file.read(1024*1024+1)
    if len(data)>1024*1024 or hashlib.sha256(data).hexdigest()!=qwen.CONFIG_SHA:
        raise ValueError("Qwen SiLU config identity differs")
    config=json.loads(data)
    if config["intermediate_size"]!=WIDTH or config["hidden_act"]!="silu":
        raise ValueError("Qwen SiLU width or activation differs")
    return half,single


def payload(root,item,dtype,rows):
    # Saved selected arrays have at most four rows. Read complete payloads and
    # bind their full hashes before splitting; native cases retain <=8192 values.
    size={"f16le":2,"f32le":4}[dtype]
    if (not 1<=rows<=4 or item["dtype"]!=dtype or item["shape"]!=[rows,WIDTH]
            or item["bytes"]!=rows*WIDTH*size):
        raise ValueError("Qwen SiLU payload shape/dtype/size differs")
    path=root/item["file"]
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Qwen SiLU payload escapes reference directory")
    with path.open("rb") as file:data=file.read(item["bytes"]+1)
    if len(data)!=item["bytes"] or hashlib.sha256(data).hexdigest()!=item["sha256"]:
        raise ValueError("Qwen SiLU complete payload identity differs")
    return data


def assemble(half,single,half_root,single_root):
    inputs=[];anchors={}
    for name,rows in qwen.CASE_ROWS.items():
        for layer in LAYERS:
            prefix=f"cases.{name}.layers.{layer}";stage=f"{prefix}.gate_silu"
            source_item=half["arrays"][f"{prefix}.gate_projection"];oracle_item=half["arrays"][stage]
            # The original source declares FP32 globally. This read view adds
            # its declared dtype without editing source artifacts or gate policy.
            ref_item={"dtype":"f32le",**single["arrays"][stage]}
            source=payload(half_root,source_item,"f16le",rows)
            oracle=payload(half_root,oracle_item,"f16le",rows)
            reference=payload(single_root,ref_item,"f32le",rows)
            if oracle_item["source_fp32_full_sha256"]!=ref_item["sha256"]:
                raise ValueError("Qwen SiLU FP32 source anchor differs")
            count=rows*WIDTH
            source_bits=list(struct.unpack(f"<{count}H",source));oracle_bits=list(struct.unpack(f"<{count}H",oracle))
            ref_values=list(struct.unpack(f"<{count}f",reference))
            if (any(value & 0x7c00==0x7c00 for value in source_bits+oracle_bits)
                    or not all(math.isfinite(value) for value in ref_values)):
                raise ValueError("Qwen SiLU source/oracle contains nonfinite values")
            for row in range(rows):
                key=f"{stage}.row.{row}";start=row*WIDTH;end=start+WIDTH
                inputs.append({"id":key,"rows":1,"width":WIDTH,"input_bits":source_bits[start:end]})
                anchors[key]={"bits":oracle_bits[start:end],"reference":ref_values[start:end],
                    "gate":oracle_item["gate_against_source_fp32"],"source_stage":stage,"row_index":row,
                    "input_full_fp16_sha256":source_item["sha256"],"oracle_full_fp16_sha256":oracle_item["sha256"],
                    "source_full_fp32_sha256":ref_item["sha256"],
                    "input_fp16_sha256":hashlib.sha256(source[row*WIDTH*2:(row+1)*WIDTH*2]).hexdigest(),
                    "oracle_fp16_sha256":hashlib.sha256(oracle[row*WIDTH*2:(row+1)*WIDTH*2]).hexdigest(),
                    "source_fp32_sha256":hashlib.sha256(reference[row*WIDTH*4:(row+1)*WIDTH*4]).hexdigest()}
    if len(inputs)!=36 or sum(item["rows"]*item["width"] for item in inputs)!=175104:
        raise ValueError("Qwen SiLU selected row inventory differs")
    return inputs,anchors


def run(output,build,directory):
    qwen.check_paths(output,build,directory)
    half,single=checked_references();inputs,anchors=assemble(half,single,qwen.HALF,qwen.SINGLE)
    tiny_inputs,tiny_anchors=silu.cases()
    directory.mkdir(parents=True);identities={};jobs=[]
    groups=[("tiny_silu_regression",tiny_inputs,tiny_anchors)]+[
        (f"qwen_silu_row_{index+1}",[item],anchors) for index,item in enumerate(inputs)]
    for label,group,expected in groups:
        path=directory/f"{label}.json"
        with path.open("x") as file:file.write(json.dumps({"format":"toklet-metal-silu-input-v1","cases":group},separators=(",",":"))+"\n")
        sha=qwen.rms.probe.common.digest(path);identities[str(path.resolve())]=sha
        def qualified(event,source=group,expected=expected,fixture=path,expected_sha=sha):
            silu.validate(event,source,expected)
            if qwen.rms.probe.common.digest(fixture)!=expected_sha:
                raise ValueError("Qwen SiLU batch changed during execution")
        jobs.append((label,["--silu-input",str(path.resolve())],qualified))
    modules=(qwen,qwen.rms,silu,silu.residual,silu.residual.embedding)
    drivers={str(Path(module.__file__).resolve()):qwen.rms.probe.common.digest(Path(module.__file__)) for module in modules}
    drivers[str(Path(__file__).resolve())]=qwen.rms.probe.common.digest(Path(__file__))
    previous=jobs[-1][2]
    def final(event):
        previous(event);checked_references()
        silu.rms.manifest(silu.rms.FP16,silu.rms.FP16_MANIFEST_SHA)
        silu.rms.manifest(silu.rms.FP32,silu.rms.FP32_MANIFEST_SHA)
        for path,sha in {**identities,**drivers}.items():
            if qwen.rms.probe.common.digest(Path(path))!=sha:raise ValueError("Qwen SiLU driver/input changed")
    jobs[-1]=(jobs[-1][0],jobs[-1][1],final)
    context={"qualification_driver_sha256":drivers,"input_sha256":identities,
        "manifest_sha256":{str(root.relative_to(ROOT)):sha for root,sha in qwen.IDENTITIES.items()},
        "tiny_manifest_sha256":{"fp16":silu.rms.FP16_MANIFEST_SHA,"source_fp32":silu.rms.FP32_MANIFEST_SHA},
        "config_sha256":qwen.CONFIG_SHA,"frozen_source_checkpoint_sha256":qwen.WEIGHT_SHA,
        "frozen_source_checkpoint_bytes":qwen.WEIGHT_BYTES,"checkpoint_reads":False,
        "scope":"12 selected width4864 SiLU stages split into36 complete rows; unchanged pre-GPU source gates and diagnostic exact Half bits. Same-binary tiny regression. No checkpoint reads, other layers, gating/MLP composition, sessions, quality, performance, trace or M4 completion."}
    qwen.rms.probe.run(output,build,extra_runs=jobs,extra_context=context)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True);parser.add_argument("--build-dir",type=Path,required=True)
    parser.add_argument("--input-dir",type=Path,required=True);args=parser.parse_args();run(args.output,args.build_dir,args.input_dir)


if __name__=="__main__":main()
