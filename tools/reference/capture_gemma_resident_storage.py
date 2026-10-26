"""Pre-target source inventory for one pinned Gemma4 E2B resident weight bundle.

No engine, GPU, tokenizer, repository code or full CPU model is loaded. The
independent replay deliberately does not import this implementation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import sys

MODEL_BYTES = 3550670554
MODEL_SHA = "038e39a37a7667373d2c3991375446b10c96ae1d717a68674870343db376b76e"
PINS = {
    "config.json": (6395, "6397cb6eca41b911d1dcab74e17941351057bd759284052a2331918ff6f9246c"),
    "index.json": (218323, "edb157dbf495e23f37377af4a628a9ad13c4ee7937f93ccb36ec9e9a19940f16"),
    "header.json": (318220, "4d1819ededded19812059691c0f77cfec0fe10c9c89c720950f919bfc26544a7"),
}
PREFIX = "language_model.model."
CHUNK_ROWS = 16384
GUARD = struct.pack("<I", 0x5A170031) * 7
STREAM = 65536


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON field")
        result[key] = value
    return result


def parse(raw):
    return json.loads(raw, object_pairs_hook=unique)


def failed(emit, value, error):
    """Secondary diagnostic failure must never replace the original cause."""
    value.update(error_type=type(error).__name__, error=str(error)[:1024], authenticated=False)
    try:
        emit("FAILED_UNAUTHENTICATED_COMPONENT", value)
    except BaseException as secondary:
        try:
            print("GEMMA_STORAGE_REFERENCE_SECONDARY_DIAGNOSTIC_FAILURE "
                  + type(secondary).__name__ + ": " + str(secondary)[:1024], file=sys.stderr, flush=True)
        except BaseException:
            pass


def exact(reader, count):
    result = bytearray()
    while len(result) < count:
        part = reader.read(count - len(result))
        require(part and len(part) <= count - len(result), "truncated source")
        result.extend(part)
    return bytes(result)


def pinned_metadata(folder):
    result = {}
    for name, (size, digest) in PINS.items():
        with (folder / name).open("rb") as file:
            raw = file.read(size + 1)
        require(len(raw) == size and hashlib.sha256(raw).hexdigest() == digest,
                "metadata identity " + name)
        parse(raw)
        result[name] = raw
    return result


def logical_inventory():
    result = []

    def matrix(name, rows, columns):
        result.append(dict(name=PREFIX + name + ".weight", kind="matrix",
                           rows=rows, columns=columns))

    def vector(name, count, suffix=".weight"):
        result.append(dict(name=PREFIX + name + suffix, kind="vector", count=count))

    matrix("embed_tokens", 262144, 1536)
    matrix("embed_tokens_per_layer", 262144, 8960)
    matrix("per_layer_model_projection", 8960, 1536)
    vector("per_layer_projection_norm", 256)
    vector("norm", 1536)
    for layer in range(35):
        base = f"layers.{layer}."
        dim = 512 if layer % 5 == 4 else 256
        ff = 6144 if layer < 15 else 12288
        for name in ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
                     "post_feedforward_layernorm", "post_per_layer_input_norm"):
            vector(base + name, 1536)
        vector(base + "layer_scalar", 1, "")
        vector(base + "self_attn.q_norm", dim)
        for name, rows, columns in (
            ("self_attn.q_proj", 8 * dim, 1536), ("self_attn.o_proj", 1536, 8 * dim),
            ("mlp.gate_proj", ff, 1536), ("mlp.up_proj", ff, 1536),
            ("mlp.down_proj", 1536, ff), ("per_layer_input_gate", 256, 1536),
            ("per_layer_projection", 1536, 256),
        ):
            matrix(base + name, rows, columns)
        if layer < 15:
            vector(base + "self_attn.k_norm", dim)
            matrix(base + "self_attn.k_proj", dim, 1536)
            matrix(base + "self_attn.v_proj", dim, 1536)
    require(len(result) == 540, "logical inventory")
    return result


def physical_inventory(header, logical):
    expected = {}
    for tensor in logical:
        name = tensor["name"]
        if tensor["kind"] == "vector":
            expected[name] = ("BF16", [tensor["count"]], tensor["count"] * 2)
        else:
            n, k = tensor["rows"], tensor["columns"]
            expected[name] = ("U32", [n, k // 8], n * k // 2)
            for suffix in ("scales", "biases"):
                expected[name[:-6] + suffix] = ("BF16", [n, k // 64], n * k // 32)
    require(len(expected) == 1096, "physical inventory")
    require({n for n in header if n.startswith(PREFIX)} == set(expected), "text names")
    for name, (dtype, shape, count) in expected.items():
        value = header[name]
        require(value["dtype"] == dtype and value["shape"] == shape, "source shape " + name)
        start, end = value["data_offsets"]
        require(end - start == count, "source length " + name)
    return expected


def phase_one(reader, metadata, text_names, emit):
    require(reader.seek(0) == 0, "initial seek")
    prefix = exact(reader, 8)
    require(struct.unpack("<Q", prefix)[0] == len(metadata["header.json"]), "header length")
    raw_header = exact(reader, len(metadata["header.json"]))
    require(raw_header == metadata["header.json"], "parsed/authenticated header differs")
    header = parse(raw_header)
    spans = sorted((v["data_offsets"][0], v["data_offsets"][1], name)
                   for name, v in header.items() if name != "__metadata__")
    whole = hashlib.sha256(prefix + raw_header)
    cursor = 0
    components = {}
    for start, end, name in spans:
        require(type(start) is int and type(end) is int and start == cursor < end,
                "container contiguous coverage")
        component = hashlib.sha256()
        remaining = end - start
        consumed = 0
        try:
            while remaining:
                raw = reader.read(min(STREAM, remaining))
                require(raw and len(raw) <= min(STREAM, remaining), "truncated source")
                whole.update(raw)
                component.update(raw)
                consumed += len(raw)
                remaining -= len(raw)
        except BaseException as error:
            failed(emit, dict(phase="container", name=name, offset=8 + len(raw_header) + start,
                              consumed_bytes=consumed, prefix_sha256=component.hexdigest()), error)
            raise
        if name in text_names:
            components[name] = dict(offset=8 + len(raw_header) + start,
                                    bytes=end - start, sha256=component.hexdigest())
            emit("COMPONENT_CANDIDATE", dict(name=name, **components[name]))
        cursor = end
    require(8 + len(raw_header) + cursor == MODEL_BYTES and reader.read(1) == b"",
            "container byte count/EOF")
    require(whole.hexdigest() == MODEL_SHA, "container SHA256")
    require(set(components) == set(text_names), "authenticated component completeness")
    emit("AUTHENTICATED_CONTAINER", dict(bytes=MODEL_BYTES, sha256=MODEL_SHA))
    return components


def selected(reader, offset, count, digest):
    require(reader.seek(offset) == offset, "component seek")
    remaining = count
    while remaining:
        raw = reader.read(min(STREAM, remaining))
        require(raw and len(raw) <= min(STREAM, remaining), "truncated selected source")
        digest.update(raw)
        remaining -= len(raw)
        yield raw


def packed_chunks(reader, tensor, components, emit):
    n, k, name = tensor["rows"], tensor["columns"], tensor["name"]
    roles = [("codes", name, k // 2), ("scales", name[:-6] + "scales", k // 32),
             ("biases", name[:-6] + "biases", k // 32)]
    chunks = [dict(first=first, rows=min(CHUNK_ROWS, n - first), buffers={})
              for first in range(0, n, CHUNK_ROWS)]
    for role, physical, stride in roles:
        source = components[physical]
        full = hashlib.sha256()
        covered = 0
        for chunk in chunks:
            count = chunk["rows"] * stride
            offset = source["offset"] + chunk["first"] * stride
            require(chunk["first"] * stride == covered, "increasing component coverage")
            payload = hashlib.sha256()
            guarded = hashlib.sha256()
            consumed = 0
            try:
                for raw in selected(reader, offset, count, full):
                    consumed += len(raw)
                    payload.update(raw)
                    guarded.update(raw)
            except BaseException as error:
                failed(emit, dict(phase="packed", name=physical, offset=offset, first=chunk["first"],
                                  consumed_bytes=consumed, prefix_sha256=payload.hexdigest(),
                                  component_consumed_bytes=covered + consumed,
                                  component_prefix_sha256=full.hexdigest()), error)
                raise
            padding = (-count) % 4
            guarded.update(b"\0" * padding)
            guarded.update(GUARD)
            entry = dict(offset=offset, source_bytes=count, source_sha256=payload.hexdigest(),
                         padding_bytes=padding, guarded_bytes=count + padding + len(GUARD),
                         guarded_sha256=guarded.hexdigest())
            chunk["buffers"][role] = entry
            covered += count
            emit("EXPECTED_PACKED_BUFFER", dict(name=name, first=chunk["first"], role=role, **entry))
        try:
            require(covered == source["bytes"] and full.hexdigest() == source["sha256"],
                    "selected packed source changed " + physical)
        except BaseException as error:
            failed(emit, dict(phase="packed", name=physical, offset=source["offset"],
                              consumed_bytes=covered, prefix_sha256=full.hexdigest()), error)
            raise
    return dict(**tensor, chunks=chunks)


def resident_vector(reader, tensor, components, emit):
    source = components[tensor["name"]]
    full = hashlib.sha256()
    converted = hashlib.sha256()
    guarded = hashlib.sha256()
    negative_zeros = 0
    consumed, converted_count = 0, 0
    offending = {}
    pending_byte = b""
    try:
        for raw in selected(reader, source["offset"], source["bytes"], full):
            consumed += len(raw)
            paired = pending_byte + raw
            complete = len(paired) // 2 * 2
            pending_byte = paired[complete:]
            for (bits,) in struct.iter_unpack("<H", paired[:complete]):
                if bits & 0x7F80 == 0x7F80:
                    offending = dict(offending_bf16_bits=bits, offending_vector_index=converted_count)
                    raise ValueError("nonfinite BF16 vector")
                word = struct.pack("<I", bits << 16)
                converted.update(word)
                guarded.update(word)
                negative_zeros += bits == 0x8000
                converted_count += 1
        require(not pending_byte, "BF16 source alignment")
        require(full.hexdigest() == source["sha256"], "selected vector changed")
    except BaseException as error:
        failed(emit, dict(phase="vector", name=tensor["name"], offset=source["offset"],
                          consumed_bytes=consumed, prefix_sha256=full.hexdigest(),
                          converted_values=converted_count, converted_prefix_sha256=converted.hexdigest(),
                          **offending), error)
        raise
    guarded.update(GUARD)
    result = dict(**tensor, source=source, fp32_bytes=tensor["count"] * 4,
                  fp32_sha256=converted.hexdigest(), negative_zeros=negative_zeros,
                  guarded_bytes=tensor["count"] * 4 + len(GUARD), guarded_sha256=guarded.hexdigest())
    emit("EXPECTED_VECTOR_BUFFER", result)
    return result


def capture(model, metadata_folder, emit):
    metadata = pinned_metadata(metadata_folder)
    header = parse(metadata["header.json"])
    logical = logical_inventory()
    physical = physical_inventory(header, logical)
    with model.open("rb") as reader:
        components = phase_one(reader, metadata, physical, emit)
        matrices, vectors = [], []
        for tensor in logical:
            if tensor["kind"] == "matrix":
                matrices.append(packed_chunks(reader, tensor, components, emit))
            else:
                vectors.append(resident_vector(reader, tensor, components, emit))
    chunks = [c for m in matrices for c in m["chunks"]]
    packed_buffers = [b for c in chunks for b in c["buffers"].values()]
    maximum_source = max(sum(b["source_bytes"] for b in c["buffers"].values()) for c in chunks)
    require((len(matrices), len(vectors), len(chunks), len(packed_buffers)) == (278, 262, 308, 924),
            "complete storage counts")
    accounting = dict(matrix_inline_owner_bytes=278 * 112, chunk_descriptor_bytes=308 * 112,
                      packed_guarded_buffer_bytes=sum(b["guarded_bytes"] for b in packed_buffers),
                      vector_guarded_buffer_bytes=sum(v["guarded_bytes"] for v in vectors),
                      maximum_three_source_bytes=maximum_source,
                      maximum_three_source_with_descriptors=maximum_source + 72,
                      maximum_vector_raw_bytes=max(v["source"]["bytes"] for v in vectors),
                      context_policy_bytes=64 * 1024**2, cold_metadata_policy_bytes=64 * 1024**2,
                      maximum_vector_conversion_overlap_bytes=max(v["source"]["bytes"] for v in vectors),
                      aggregate_model_load_policy_bytes=None,
                      vector_owner_abi_bytes=None, bundle_owner_abi_bytes=None,
                      new_bundle_actual_capacities_admitted=False)
    accounting["known_matrix_resident_requested_bytes"] = sum(accounting[k] for k in
        ("matrix_inline_owner_bytes", "chunk_descriptor_bytes", "packed_guarded_buffer_bytes"))
    return dict(format="toklet-gemma-resident-storage-reference-v1",
                model=dict(bytes=MODEL_BYTES, sha256=MODEL_SHA),
                metadata={n:dict(bytes=s, sha256=h) for n, (s, h) in PINS.items()},
                source_components=components, matrices=matrices, vectors=vectors,
                counts=dict(matrices=278, vectors=262, source_components=1096,
                            packed_chunks=308, guarded_gpu_allocations=1186),
                accounting=accounting, reference_only=True, engine_or_gpu_executed=False,
                complete_model_storage_admitted=False, original_bf16_source_qualified=False,
                whole_graph_qualified=False, performance_qualified=False, phone_qualified=False,
                scope="Historical whole-container anchor and matching selected text bytes; "
                      "expected resident storage only; no current-path or excluded-pass2 assertion")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    def emit(label, value):
        print("GEMMA_STORAGE_REFERENCE_" + label + " " + json.dumps(value, separators=(",", ":")), flush=True)

    require(not os.path.lexists(args.output), "existing output")
    result = capture(args.model, args.metadata, emit)
    raw = (json.dumps(result, indent=2, allow_nan=False) + "\n").encode()
    require(len(raw) <= 4 * 1024**2, "inventory publication bound")
    with args.output.open("xb") as file:
        require(file.write(raw) == len(raw), "short publication")
    emit("COMPLETE", dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), counts=result["counts"]))


if __name__ == "__main__":
    main()
