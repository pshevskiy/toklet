"""Independent scalar Q4 tiny decoder oracle; freeze before running Metal."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import fp16
import q4_tiny

ROOT = Path(__file__).resolve().parents[2]
POLICY = q4_tiny.POLICY

SOURCE_MANIFEST_SHA = "1d40cec4f24ece5fdc7b727d3e96f41237621120908833ee6c7df83352acfdd4"
CONFIG_SHA = "84b6e6078f35006f49ae2e2a8663093cdc8d2813e2b7f28ebfa6142113e9c1d9"
SOURCE_SHARD_SHA = "b01a114bf2d44f0c6cb438035e85535b658e803c76691729facd78b15eefbce2"


def bounded(path, limit, expected=None):
    with path.open("rb") as stream: data=stream.read(limit+1)
    if len(data)>limit: raise ValueError("Independent v2 input exceeds bound")
    identity=hashlib.sha256(data).hexdigest()
    if expected is not None and identity!=expected: raise ValueError("Independent v2 consumed bytes differ")
    return data,identity


def digest(path): return bounded(path,16*1024**2)[1]


def unique(pairs):
    result={}
    for key,value in pairs:
        if key in result: raise ValueError("Duplicate independent v2 JSON key")
        result[key]=value
    return result


def source_array(root,source,name,endpoints):
    item=source["arrays"][name];path=root/item["file"]
    data,identity=bounded(path,16*1024**2,item["sha256"]);endpoints[path]=identity
    dtype=fp16.DTYPES[item["dtype"]]
    if len(data)!=item["bytes"] or len(data)!=int(np.prod(item["shape"],dtype=object))*np.dtype(dtype).itemsize:
        raise ValueError("Independent v2 source shape/length differs")
    result=np.frombuffer(data,dtype=dtype).reshape(item["shape"])
    if not np.isfinite(result).all(): raise ValueError("Independent v2 source nonfinite")
    return result


def reconstruct(data, rows, columns):
    if type(rows) is not int or type(columns) is not int or rows<=0 or columns<=0 or len(data)<64:
        raise ValueError("Q4 v2 reference dimensions/header differ")
    magic, version, group, r, c, stride, payload = struct.unpack("<8sIIQQQQ16x", data[:64])
    expected_stride=((columns+7)//8*8+63)//64*64
    if (magic,version,group,r,c,stride,payload,len(data)) != (b"TOKQ4V2\0",2,8,rows,columns,expected_stride,rows*expected_stride,64+rows*expected_stride):
        raise ValueError("Q4 v2 reference header/length differs")
    if any(data[48:64]): raise ValueError("Q4 v2 reserved bytes differ")
    output = np.empty((rows, columns), dtype=np.float32)
    for row in range(rows):
        logical=((columns+7)//8)*8
        if any(data[64+row*stride+logical:64+(row+1)*stride]): raise ValueError("Q4 v2 row padding differs")
        for first in range(0,columns,8):
            start=64+row*stride+first
            scale,bias=struct.unpack_from("<ee",data,start)
            if not np.isfinite(scale) or not np.isfinite(bias) or data[start+1]&128:
                raise ValueError("Q4 v2 metadata differs")
            for lane in range(8):
                code=(data[start+4+lane//2]>>(4*(lane%2)))&15
                if (first+lane>=columns or scale==0) and code:
                    raise ValueError("Q4 v2 constant/tail codes differ")
                if first+lane<columns:
                    output[row,first+lane]=np.float32(np.float32(scale)*np.float32(code)+np.float32(bias))
    return output


def export(packed, output):
    if output.exists(): raise ValueError("refusing to overwrite a frozen Q4 reference")
    if np.__version__ != "2.3.2": raise ValueError("use frozen NumPy 2.3.2")
    source_path = ROOT / "tests/fixtures/tiny-v1"
    source_bytes,source_sha=bounded(source_path/"manifest.json",1024**2,SOURCE_MANIFEST_SHA)
    manifest_bytes,manifest_sha=bounded(packed/"manifest.json",1024**2)
    source=json.loads(source_bytes,object_pairs_hook=unique);manifest=json.loads(manifest_bytes,object_pairs_hook=unique)
    endpoints={source_path/"manifest.json":source_sha,packed/"manifest.json":manifest_sha}
    suffixes=("self_attn.q_proj.weight","self_attn.k_proj.weight","self_attn.v_proj.weight","self_attn.o_proj.weight","mlp.gate_proj.weight","mlp.up_proj.weight","mlp.down_proj.weight")
    expected={f"model.layers.{layer}.{suffix}":f"layers.{layer}.{suffix.split('.')[1]}.weight" for layer in (0,1) for suffix in suffixes}
    if (manifest["format"]!="toklet-q4-model-v2" or manifest["quantizer"]!="toklet-q4-affine-f32-source-half-metadata-rne-g8-v2"
            or manifest["config_sha256"]!=CONFIG_SHA or manifest["sources"]!=[{"file":"model.safetensors","sha256":SOURCE_SHARD_SHA}]
            or manifest["exclusions"]!=["norms","qkv_biases","tied_embedding_output"] or manifest["excluded_dtype"]!="F16"
            or len(manifest["matrices"])!=14 or {entry["name"] for entry in manifest["matrices"]}!=set(expected)):
        raise ValueError("Q4 v2 exact source/format/inventory differs")
    weights={name.removeprefix("weights."):source_array(source_path,source,name,endpoints).astype(np.float16)
             for name in source["arrays"] if name.startswith("weights.")}
    identities={};packed_total=0
    for entry in manifest["matrices"]:
        key=expected[entry["name"]];original=source_array(source_path,source,f"weights.{key}",endpoints)
        if (entry["file"]!=entry["name"]+".q4" or list(original.shape)!=[entry["rows"],entry["columns"]]
                or hashlib.sha256(original.astype('<f4').tobytes()).hexdigest()!=entry["source_f32_sha256"]):
            raise ValueError("Q4 v2 tensor does not bind exact frozen source")
        path=packed/entry["file"];data,identity=bounded(path,16*1024**2,entry["packed_sha256"])
        if len(data)!=entry["packed_bytes"]: raise ValueError("Q4 v2 packed byte count differs")
        weights[key]=reconstruct(data,*original.shape);endpoints[path]=identity
        identities[entry["name"]]=identity;packed_total+=len(data)
    if packed_total!=manifest["packed_matrix_bytes"]: raise ValueError("Q4 v2 packed inventory total differs")
    oracle_sources={path.name:digest(path) for path in (Path(__file__),Path(q4_tiny.__file__),Path(fp16.__file__))}
    endpoints.update({path:oracle_sources[path.name] for path in (Path(__file__),Path(q4_tiny.__file__),Path(fp16.__file__))})
    output.mkdir(parents=True)
    arrays = {}; cases = {}
    pattern = [1,7,13,3,47,11,2,5] * 3
    inputs = {f"rows{n}":(np.asarray(pattern[:n],dtype=np.uint32),0) for n in (7,8,9,17,24)}
    inputs["offset17"] = (np.asarray(pattern[:17],dtype=np.uint32),5)
    for case,(tokens,offset) in inputs.items():
        computed = fp16.forward(tokens, offset, weights, source["model"], np.float32)
        wide = fp16.forward(tokens, offset, weights, source["model"], np.float64)
        cases[case] = {"tokens": tokens.tolist(), "position_offset": offset,
                       "greedy": np.argmax(computed["logits"], axis=1).tolist()}
        for stage, value in computed.items():
            if stage.endswith(("attention_scores", "attention_probabilities")): continue
            name = f"cases.{case}.{stage}"
            dtype = "f16le" if value.dtype == np.float16 else "f32le"
            filename = f"{name}.{dtype}"
            data = np.asarray(value, dtype='<f2' if dtype == "f16le" else '<f4').tobytes()
            (output/filename).write_bytes(data)
            envelope = float(np.max(np.abs(value.astype(np.float64)-wide[stage].astype(np.float64))))
            arrays[name] = {"file":filename,"shape":list(value.shape),"dtype":dtype,"bytes":len(data),
                           "sha256":hashlib.sha256(data).hexdigest(),"gate":{"atol":max(1e-4,2*envelope),"rtol":1e-3},
                           "independent_precision_envelope":envelope}
    record = {"format":"toklet-q4-tiny-prefill-numerical-v2","policy":POLICY,"numpy":np.__version__,
              "source_manifest_sha256":source_sha,
              "packed_manifest_sha256":manifest_sha,"packed_matrix_hashes":identities,
              "oracle_sources":oracle_sources,
              "model":source["model"],"cases":cases,"arrays":arrays}
    for path,identity in endpoints.items():
        if digest(path)!=identity: raise ValueError("Independent v2 source/packed endpoint changed")
    with (output/"manifest.json").open("x") as stream: stream.write(json.dumps(record,indent=2)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packed",type=Path); parser.add_argument("output",type=Path)
    args = parser.parse_args(); export(args.packed,args.output)


if __name__ == "__main__": main()
