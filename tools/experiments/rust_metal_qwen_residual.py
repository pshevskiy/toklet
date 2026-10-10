"""Qualify selected Qwen Half residuals using saved operands and frozen gates."""
import argparse
import json
from pathlib import Path
import struct

import rust_metal_qwen_rms as qwen
import rust_metal_residual as residual

ROOT=qwen.ROOT
OPERATIONS=(
    ("layers.0.attention_residual","embedding","layers.0.attention_projection"),
    ("layers.0.output","layers.0.attention_residual","layers.0.down_projection"),
    ("layers.1.attention_residual","layers.0.output","layers.1.attention_projection"),
    ("layers.1.output","layers.1.attention_residual","layers.1.down_projection"),
    ("layers.23.output","layers.23.attention_residual","layers.23.down_projection"))


def assemble(half,single,half_root,single_root):
    inputs=[];anchors={}
    for name,rows in qwen.CASE_ROWS.items():
        prefix=f"cases.{name}";count=rows*896
        for operation,state_name,delta_name in OPERATIONS:
            key=f"{prefix}.{operation}"
            payloads=[qwen.rms.payload(half_root,half,f"{prefix}.{n}","f16le") for n in (state_name,delta_name,operation)]
            # Original source artifacts declare FP32 globally. Add that dtype
            # only to a read view; saved source/gate metadata is never changed.
            view={"arrays":{key:{"dtype":"f32le",**single["arrays"][key]}}}
            reference,ref=qwen.rms.payload(single_root,view,key,"f32le")
            (state,state_item),(delta,delta_item),(oracle,item)=payloads
            if (ref["shape"]!=[rows,896] or any(meta["shape"]!=[rows,896] for _,meta in payloads)
                    or item["source_fp32_full_sha256"]!=ref["sha256"]):
                raise ValueError("Qwen residual activation/source shape differs")
            a=list(struct.unpack(f"<{count}H",state));b=list(struct.unpack(f"<{count}H",delta));bits=list(struct.unpack(f"<{count}H",oracle))
            if bits!=[residual.half_sum(left,right) for left,right in zip(a,b)]:
                raise ValueError("Qwen residual oracle differs from independent Half sum")
            inputs.append({"id":key,"rows":rows,"hidden":896,"state_bits":a,"delta_bits":b})
            anchors[key]={"bits":bits,"reference":list(struct.unpack(f"<{count}f",reference)),"gate":item["gate_against_source_fp32"],
                "state_fp16_sha256":state_item["sha256"],"delta_fp16_sha256":delta_item["sha256"],
                "oracle_fp16_sha256":item["sha256"],"source_fp32_sha256":ref["sha256"]}
    if len(inputs)!=20 or sum(item["rows"]*item["hidden"] for item in inputs)!=53760:
        raise ValueError("Qwen residual selected inventory differs")
    return inputs,anchors


def run(output,build,directory):
    # Reuse disjoint/fresh/protected fixture/config-path admission. No model
    # payload or norm_weights/stream_digest call occurs in this driver.
    qwen.check_paths(output,build,directory)
    half,single,_=qwen.checked_manifests()
    inputs,anchors=assemble(half,single,qwen.HALF,qwen.SINGLE)
    grouped=qwen.batches(inputs)
    if len(grouped)!=8:raise ValueError("Qwen residual batch inventory differs")
    directory.mkdir(parents=True);identities={};jobs=[]
    for index,group in enumerate(grouped):
        path=directory/f"batch-{index+1}.json"
        with path.open("x") as file:file.write(json.dumps({"format":"toklet-metal-residual-input-v1","cases":group},separators=(",",":"))+"\n")
        sha=qwen.rms.probe.common.digest(path);identities[str(path.resolve())]=sha
        def qualified(event,source=group,fixture=path,expected_sha=sha):
            residual.validate(event,source,anchors)
            if qwen.rms.probe.common.digest(fixture)!=expected_sha:
                raise ValueError("Qwen residual batch changed during execution")
        jobs.append((f"qwen_residual_batch_{index+1}",["--residual-input",str(path.resolve())],qualified))
    drivers={str(Path(module.__file__).resolve()):qwen.rms.probe.common.digest(Path(module.__file__)) for module in (qwen,qwen.rms,residual)}
    drivers[str(Path(__file__).resolve())]=qwen.rms.probe.common.digest(Path(__file__))
    last=jobs[-1][2]
    def final(event):
        last(event);qwen.checked_manifests()
        for path,sha in {**identities,**drivers}.items():
            if qwen.rms.probe.common.digest(Path(path))!=sha:raise ValueError("Qwen residual driver/input changed")
    jobs[-1]=(jobs[-1][0],jobs[-1][1],final)
    context={"qualification_driver_sha256":drivers,"input_sha256":identities,
        "manifest_sha256":{str(root.relative_to(ROOT)):sha for root,sha in qwen.IDENTITIES.items()},"config_sha256":qwen.CONFIG_SHA,
        "frozen_source_checkpoint_sha256":qwen.WEIGHT_SHA,"frozen_source_checkpoint_bytes":qwen.WEIGHT_BYTES,
        "checkpoint_reads":False,
        "scope":"20 saved width896 residual stages, four frozen short Qwen cases; exact independent Half results and unchanged source gates. No checkpoint payload read, layer23attention residual, otherlayers, composition/cache/recovery/quality/performance or M4 completion."}
    qwen.rms.probe.run(output,build,extra_runs=jobs,extra_context=context)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True);parser.add_argument("--build-dir",type=Path,required=True)
    parser.add_argument("--input-dir",type=Path,required=True);args=parser.parse_args();run(args.output,args.build_dir,args.input_dir)


if __name__=="__main__":main()
