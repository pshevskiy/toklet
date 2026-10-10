"""Qualify selected Qwen width-896 RMS stages with unchanged pre-GPU gates."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

import rust_metal_rms as rms

ROOT=rms.ROOT
HALF=ROOT/"artifacts/qwen-fp16-full-v2"
SINGLE=ROOT/"artifacts/qwen-forward-v2"
HALF_PROBES=ROOT/"tests/fixtures/qwen-fp16-v1"
SINGLE_PROBES=ROOT/"tests/fixtures/qwen-v2"
CONFIG=ROOT/"models/Qwen2.5-0.5B-Instruct/config.json"
WEIGHTS=ROOT/"models/Qwen2.5-0.5B-Instruct/model.safetensors"
IDENTITIES={HALF:"49206b6f5bed4f605671bf45474f25c81cb0c48ee3c33a9f741c513eb9ecbb72",
    SINGLE:"bdd368d21dfae574197e3b64423b75d4f1b2b37c083c6172c429e36bb4dd69c7",
    HALF_PROBES:"1f43732796e0a4e60d428d27beb6f549ee14aa4989870369d390654afe43f4fd",
    SINGLE_PROBES:"691429f672aa89875e06475f48b5b691343f03c08e47fd204fa60fec2eef20d1"}
CONFIG_SHA="18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45"
WEIGHT_SHA="fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"
WEIGHT_BYTES=988097824
OPERATIONS=(
    ("layers.0.input_norm","embedding","model.layers.0.input_layernorm.weight"),
    ("layers.0.post_attention_norm","layers.0.attention_residual","model.layers.0.post_attention_layernorm.weight"),
    ("layers.1.input_norm","layers.0.output","model.layers.1.input_layernorm.weight"),
    ("layers.1.post_attention_norm","layers.1.attention_residual","model.layers.1.post_attention_layernorm.weight"),
    ("layers.23.post_attention_norm","layers.23.attention_residual","model.layers.23.post_attention_layernorm.weight"),
    ("final_norm","layers.23.output","model.norm.weight"))
CASE_ROWS={"future_changed":4,"offset":3,"short":4,"single":1}


def stream_digest(file):
    file.seek(0)
    sha=hashlib.sha256()
    for block in iter(lambda:file.read(1024*1024),b""):sha.update(block)
    return sha.hexdigest()


def unique_object(items):
    result={}
    for key,value in items:
        if key in result:raise ValueError("Duplicate safetensors metadata key")
        result[key]=value
    return result


def norm_weights(path,expected_sha,expected_bytes,names,hidden=896):
    """Read only norm rows; stream source hash without mapping/copying the model."""
    if len(set(names))!=len(names):raise ValueError("Duplicate requested norm tensor")
    with path.open("rb") as file:
        if path.stat().st_size!=expected_bytes or stream_digest(file)!=expected_sha:
            raise ValueError("Qwen checkpoint identity differs")
        file.seek(0)
        prefix=file.read(8)
        if len(prefix)!=8:raise ValueError("Truncated safetensors header prefix")
        size=struct.unpack("<Q",prefix)[0]
        if not 2<=size<=1024*1024 or size>expected_bytes-8:
            raise ValueError("Safetensors header exceeds bounds")
        payload=file.read(size)
        if len(payload)!=size:raise ValueError("Truncated safetensors header")
        metadata=json.loads(payload,object_pairs_hook=unique_object)
        if not isinstance(metadata,dict):raise ValueError("Safetensors header must be an object")
        ranges=[]
        weights={}
        for name in names:
            item=metadata.get(name)
            if not isinstance(item,dict) or item.get("dtype")!="BF16" or item.get("shape")!=[hidden]:
                raise ValueError("Norm tensor dtype/shape differs")
            bounds=item.get("data_offsets")
            if (not isinstance(bounds,list) or len(bounds)!=2
                    or any(type(value) is not int for value in bounds)
                    or not 0<=bounds[0]<bounds[1]<=expected_bytes-8-size
                    or bounds[1]-bounds[0]!=hidden*2 or bounds[0]%2):
                raise ValueError("Norm tensor range differs")
            if any(bounds[0]<end and start<bounds[1] for start,end in ranges):
                raise ValueError("Norm tensor ranges overlap")
            ranges.append(tuple(bounds))
            file.seek(8+size+bounds[0])
            data=file.read(hidden*2)
            if len(data)!=hidden*2:raise ValueError("Truncated norm payload")
            bits=[]
            for value in struct.unpack(f"<{hidden}H",data):
                # BF16 bits shift into their exact FP32 representation; Python
                # then performs one IEEE Half round-to-nearest-even storage cast.
                wide=struct.unpack("<f",struct.pack("<I",value << 16))[0]
                if not math.isfinite(wide):raise ValueError("Nonfinite BF16 norm")
                try:half=struct.unpack("<H",struct.pack("<e",wide))[0]
                except OverflowError as error:raise ValueError("Norm exceeds Half range") from error
                if half & 0x7c00 == 0x7c00:raise ValueError("Nonfinite Half norm")
                bits.append(half)
            weights[name]={"bits":bits,"source_bf16_sha256":hashlib.sha256(data).hexdigest(),
                           "fp16_sha256":hashlib.sha256(struct.pack(f"<{hidden}H",*bits)).hexdigest(),
                           "data_offsets":bounds}
        # Bind loaded bytes to the same source descriptor after small reads.
        if path.stat().st_size!=expected_bytes or stream_digest(file)!=expected_sha:
            raise ValueError("Qwen checkpoint changed during norm loading")
    return weights


def checked_manifests():
    manifests={root:rms.manifest(root,sha) for root,sha in IDENTITIES.items()}
    half,single,hprobe,sprobe=(manifests[root] for root in (HALF,SINGLE,HALF_PROBES,SINGLE_PROBES))
    if (half["source_full_manifest_sha256"]!=IDENTITIES[SINGLE]
            or half["source_probes_manifest_sha256"]!=IDENTITIES[SINGLE_PROBES]
            or half["config_sha256"]!=CONFIG_SHA
            or any(item["model"]!=half["model"] or item["cases"]!=half["cases"] for item in (single,hprobe,sprobe))
            or {name:len(case["tokens"]) for name,case in half["cases"].items()}!=CASE_ROWS
            or half["model"]["weights"]["sha256"]!=WEIGHT_SHA
            or half["model"]["weights"]["bytes"]!=WEIGHT_BYTES):
        raise ValueError("Frozen Qwen RMS provenance differs")
    for key,item in half["arrays"].items():
        anchor=hprobe["arrays"][key]
        if (item["sha256"]!=anchor["full_array_sha256"] or item["bytes"]!=anchor["full_array_bytes"]
                or item["shape"]!=anchor["shape"] or item["gate_against_source_fp32"]!=anchor["gate_against_source_fp32"]
                or item["source_fp32_full_sha256"]!=single["arrays"][key]["sha256"]
                or single["arrays"][key]["sha256"]!=sprobe["arrays"][key]["full_array_sha256"]):
            raise ValueError("Full Qwen RMS arrays/gates differ from committed anchors")
    with CONFIG.open("rb") as file:data=file.read(1024*1024+1)
    if len(data)>1024*1024 or hashlib.sha256(data).hexdigest()!=CONFIG_SHA:
        raise ValueError("Pinned Qwen config differs")
    config=json.loads(data)
    if config["hidden_size"]!=896 or config["num_hidden_layers"]!=24:
        raise ValueError("Pinned Qwen RMS shape differs")
    return half,single,config["rms_norm_eps"]


def assemble(half,single,half_root,single_root,weights,epsilon):
    inputs=[]
    anchors={}
    for name,rows in CASE_ROWS.items():
        for operation,input_name,weight_name in OPERATIONS:
            key=f"cases.{name}.{operation}"
            x,item=rms.payload(half_root,half,f"cases.{name}.{input_name}","f16le")
            # Original full Transformers artifacts declare FP32 globally rather
            # than per array. Add that declared dtype only to this read view.
            view={"arrays":{key:{"dtype":"f32le",**single["arrays"][key]}}}
            reference,ref=rms.payload(single_root,view,key,"f32le")
            oracle,out=rms.payload(half_root,half,key,"f16le")
            if (any(shape!=[rows,896] for shape in (item["shape"],ref["shape"],out["shape"]))
                    or out["source_fp32_full_sha256"]!=ref["sha256"]):
                raise ValueError("Qwen RMS activation shape differs")
            count=rows*896
            weight=weights[weight_name]
            if len(weight["bits"])!=896:raise ValueError("Qwen norm Half count differs")
            inputs.append({"id":key,"rows":rows,"hidden":896,"epsilon_bits":rms.probe.float_bits(epsilon),
                "input_bits":list(struct.unpack(f"<{count}H",x)),"weight_bits":weight["bits"]})
            anchors[key]={"reference":list(struct.unpack(f"<{count}f",reference)),
                "oracle_bits":list(struct.unpack(f"<{count}H",oracle)),"gate":out["gate_against_source_fp32"],
                "input_sha256":item["sha256"],"weight_sha256":weight["fp16_sha256"],
                "source_bf16_weight_sha256":weight["source_bf16_sha256"],"weight_name":weight_name,
                "source_fp32_sha256":ref["sha256"],"oracle_fp16_sha256":out["sha256"]}
    if len(inputs)!=24 or sum(item["rows"]*item["hidden"] for item in inputs)!=64512:
        raise ValueError("Qwen RMS selected inventory differs")
    return inputs,anchors


def batches(inputs):
    groups=[]
    group=[]
    count=0
    for item in inputs:
        size=item["rows"]*item["hidden"]
        if not 0<size<=8192:raise ValueError("Qwen RMS case exceeds native inventory")
        if count+size>8192 or len(group)>=32:
            groups.append(group);group=[];count=0
        group.append(item);count+=size
    if group:groups.append(group)
    return groups


def check_paths(output,build,directory):
    rms.probe.check_paths(output,build)
    protected=(build,directory,HALF,SINGLE,HALF_PROBES,SINGLE_PROBES,CONFIG.parent)
    evidence=(output,output.with_suffix(output.suffix+".stderr.txt"),output.with_suffix(output.suffix+".native.jsonl"))
    if directory.exists():raise ValueError("Refusing to overwrite Qwen RMS inputs")
    if any(a.resolve()==b.resolve() or a.resolve() in b.resolve().parents or b.resolve() in a.resolve().parents
           for a,b in ((build,directory),*((directory,path) for path in evidence))):
        raise ValueError("Qwen RMS build, inputs and evidence must be distinct and disjoint")
    if any(parent.resolve()==path.resolve() or parent.resolve() in path.resolve().parents
           for parent in protected[2:] for path in (build,directory,*evidence)):
        raise ValueError("Qwen RMS outputs must be outside source fixtures/model")


def run(output,build,directory):
    check_paths(output,build,directory)
    half,single,epsilon=checked_manifests()
    names=[operation[2] for operation in OPERATIONS]
    weights=norm_weights(WEIGHTS,WEIGHT_SHA,WEIGHT_BYTES,names)
    inputs,anchors=assemble(half,single,HALF,SINGLE,weights,epsilon)
    grouped=batches(inputs)
    if len(grouped)!=9:raise ValueError("Qwen RMS batch inventory differs")
    directory.mkdir(parents=True)
    identity={}
    jobs=[]
    for index,group in enumerate(grouped):
        path=directory/f"batch-{index+1}.json"
        with path.open("x") as file:
            file.write(json.dumps({"format":"toklet-metal-rms-input-v1","cases":group},separators=(",",":"))+"\n")
        sha=rms.probe.common.digest(path)
        identity[str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path.resolve())]=sha
        def qualified(event,source=group,fixture=path,expected_sha=sha):
            rms.validate(event,source,anchors)
            if rms.probe.common.digest(fixture)!=expected_sha:
                raise ValueError("Qwen RMS batch changed during execution")
        jobs.append((f"qwen_rms_batch_{index+1}",["--rms-input",str(path.resolve())],qualified))
    driver_sha=rms.probe.common.digest(Path(__file__))
    rms_driver_sha=rms.probe.common.digest(Path(rms.__file__))
    context={"qwen_rms_driver_sha256":driver_sha,"rms_driver_sha256":rms_driver_sha,
        "manifest_sha256":{str(root.relative_to(ROOT)):sha for root,sha in IDENTITIES.items()},
        "config_sha256":CONFIG_SHA,"checkpoint_sha256":WEIGHT_SHA,"checkpoint_bytes":WEIGHT_BYTES,
        "input_sha256":identity,"norm_weights":{name:{key:value for key,value in item.items() if key!="bits"} for name,item in weights.items()},
        "scope":"24 selected RMS stages at width896, four short frozen Qwen cases; unchanged pre-GPU FP32-source gates. Layer23 input normalization, other layers, composed decoder/cache/quality/performance remain unqualified."}
    # Append an integrity check to the last native validator, before the shared
    # driver declares completion. All checkpoint hashing stays outside timing.
    last=jobs[-1][2]
    def final(event):
        last(event)
        checked_manifests()
        with WEIGHTS.open("rb") as file:
            if WEIGHTS.stat().st_size!=WEIGHT_BYTES or stream_digest(file)!=WEIGHT_SHA:
                raise ValueError("Qwen checkpoint changed during execution")
        if rms.probe.common.digest(Path(__file__))!=driver_sha or rms.probe.common.digest(Path(rms.__file__))!=rms_driver_sha:
            raise ValueError("Qwen RMS qualification driver changed")
        for path,sha in identity.items():
            if rms.probe.common.digest(ROOT/path)!=sha:raise ValueError("Qwen RMS input changed")
    jobs[-1]=(jobs[-1][0],jobs[-1][1],final)
    rms.probe.run(output,build,extra_runs=jobs,extra_context=context)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--build-dir",type=Path,required=True)
    parser.add_argument("--input-dir",type=Path,required=True)
    args=parser.parse_args()
    run(args.output,args.build_dir,args.input_dir)


if __name__=="__main__":main()
