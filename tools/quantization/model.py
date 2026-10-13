"""Stream source Qwen linear matrices into a source-bound Toklet Q4 artifact."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import sys

import q4

SUFFIXES = ("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
            "self_attn.o_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight")


def unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result: raise ValueError(f"duplicate JSON key {key}")
        result[key] = value
    return result


def read_json_identity(path):
    with path.open("rb") as file:
        size = os.fstat(file.fileno()).st_size
        if size > 1024 * 1024: raise ValueError("JSON exceeds 1 MiB")
        data = file.read(1024 * 1024 + 1)
        if len(data) > 1024 * 1024 or len(data) != size: raise ValueError("JSON size changed during bounded read")
    return json.loads(data, object_pairs_hook=unique), hashlib.sha256(data).hexdigest()


def read_json(path):
    return read_json_identity(path)[0]


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as file:
        while data := file.read(64 * 1024): value.update(data)
    return value.hexdigest()


def plain(name):
    if not isinstance(name, str) or not name or name in (".", "..") or Path(name).name != name or "/" in name or "\\" in name:
        raise ValueError("artifact filenames must be plain local filenames")
    return name


def dimensions(config):
    h, m = config["hidden_size"], config["intermediate_size"]
    heads, kv = config["num_attention_heads"], config["num_key_value_heads"]
    layers = config["num_hidden_layers"]
    if any(type(x) is not int or x <= 0 for x in (h, m, heads, kv, layers)) or layers > 1024 or h % heads or heads % kv:
        raise ValueError("invalid Qwen dimensions")
    k = h // heads * kv
    shapes = [(h, h), (k, h), (k, h), (h, h), (m, h), (m, h), (h, m)]
    return {f"model.layers.{i}.{suffix}": shape for i in range(layers) for suffix, shape in zip(SUFFIXES, shapes)}


def shards(source):
    index = source / "model.safetensors.index.json"
    index_exists = index.exists()
    parsed_index, index_hash = read_json_identity(index) if index_exists else (None, None)
    weight_map = parsed_index["weight_map"] if parsed_index is not None else None
    names = sorted(set(weight_map.values())) if weight_map is not None else ["model.safetensors"]
    if not names or len(names) > 1024: raise ValueError("invalid shard count")
    tensors = {}; identities = []; total_headers = 0
    for name in names:
        plain(name)
        if not name.endswith(".safetensors"): raise ValueError("invalid shard extension")
        path = source / name
        # Identity predates header parsing; final checks cover this entire interval.
        source_hash = digest(path)
        with path.open("rb") as file:
            length = file.read(8)
            if len(length) != 8: raise ValueError("truncated safetensors header")
            count = int.from_bytes(length, "little"); total_headers += count
            if total_headers > 16 * 1024**2: raise ValueError("aggregate headers exceed 16 MiB")
            header = file.read(count)
            if len(header) != count: raise ValueError("truncated safetensors header")
            entries = json.loads(header, object_pairs_hook=unique)
        end = 0
        for tensor, entry in sorted(((n, e) for n, e in entries.items() if n != "__metadata__"), key=lambda x: x[1]["data_offsets"][0]):
            begin, finish = entry["data_offsets"]
            if type(begin) is not int or type(finish) is not int or begin != end or finish < begin:
                raise ValueError("safetensors payload gaps/overlaps")
            end = finish
            if tensor in tensors or (weight_map is not None and weight_map.get(tensor) != name):
                raise ValueError("duplicate tensor or index/shard mismatch")
            tensors[tensor] = (path, 8 + count + begin, finish - begin, entry)
        if 8 + count + end != path.stat().st_size: raise ValueError("safetensors payload length mismatch")
        identities.append({"file": name, "sha256": source_hash})
    if weight_map is not None and set(weight_map) != set(tensors): raise ValueError("index tensor inventory mismatch")
    if index_exists: identities.append({"file": index.name, "sha256": index_hash})
    return tensors, identities


def decode(raw, dtype):
    if dtype == "F32": return struct.unpack(f"<{len(raw)//4}f", raw)
    if dtype == "F16": return struct.unpack(f"<{len(raw)//2}e", raw)
    if dtype == "BF16":
        return [struct.unpack("<f", bytes(2) + raw[i:i+2])[0] for i in range(0, len(raw), 2)]
    raise ValueError("only F32/F16/BF16 sources supported")


def pack_model(source, destination):
    config, config_hash = read_json_identity(source / "config.json")
    if config.get("model_type") != "qwen2" or config.get("tie_word_embeddings") is not True:
        raise ValueError("only tied Qwen2 sources supported")
    shapes = dimensions(config)
    tensors, identities = shards(source)
    total = q4.HEADER * len(shapes) + sum(q4.layout(*s)[1] for s in shapes.values())
    if total > q4.LIMIT: raise ValueError("packed matrices exceed 6 GiB")
    for name, shape in shapes.items():
        path, offset, count, item = tensors[name]
        width = {"F32": 4, "F16": 2, "BF16": 2}.get(item["dtype"])
        if width is None or item["shape"] != list(shape) or count != shape[0] * shape[1] * width:
            raise ValueError("source matrix shape/dtype/length mismatch")
        if shape[1] * width > 16 * 1024**2: raise ValueError("source row conversion exceeds bound")
    destination.mkdir()  # exclusive artifact directory, never overwrite any previous run
    matrices = []
    for name, (rows, columns) in shapes.items():
        path, offset, count, item = tensors[name]
        width = {"F32": 4, "F16": 2, "BF16": 2}[item["dtype"]]
        output = destination / f"{name}.q4"
        source_hash = hashlib.sha256(); packed_hash = hashlib.sha256()
        with path.open("rb") as src:
            src.seek(offset)
            owned = output.open("xb")
            try:
                with owned as dst:
                    header = q4.header(rows, columns); dst.write(header); packed_hash.update(header)
                    for _ in range(rows):
                        raw = src.read(columns * width)
                        if len(raw) != columns * width: raise ValueError("source matrix shortened")
                        values = decode(raw, item["dtype"])
                        source_hash.update(struct.pack(f"<{columns}f", *values))
                        packed = q4.pack_row(values); dst.write(packed); packed_hash.update(packed)
            except BaseException:
                output.unlink(missing_ok=True)
                raise
        matrices.append({"name": name, "file": output.name, "rows": rows, "columns": columns,
                         "packed_bytes": output.stat().st_size, "packed_sha256": packed_hash.hexdigest(),
                         "source_f32_sha256": source_hash.hexdigest()})
        print(f"packed {name}", file=sys.stderr, flush=True)
    if digest(source / "config.json") != config_hash or any(digest(source / i["file"]) != i["sha256"] for i in identities):
        raise ValueError("source identities changed during packing")
    manifest = {"format": "toklet-q4-model-v1", "quantizer": q4.POLICY, "config_sha256": config_hash,
                "sources": identities, "matrices": matrices, "packed_matrix_bytes": total,
                "exclusions": ["norms", "qkv_biases", "tied_embedding_output"], "excluded_dtype": "F16"}
    manifest_path = destination / "manifest.json"
    owned = manifest_path.open("x")
    try:
        with owned as file:
            json.dump(manifest, file, indent=2); file.write("\n")
    except BaseException:
        manifest_path.unlink(missing_ok=True)
        raise
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path); parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    pack_model(args.source, args.destination)


if __name__ == "__main__": main()
