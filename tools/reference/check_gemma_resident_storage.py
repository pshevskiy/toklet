"""Independent source replay for the pre-target Gemma resident-storage inventory."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

SOURCE_BYTES = 3550670554
SOURCE_SHA = "038e39a37a7667373d2c3991375446b10c96ae1d717a68674870343db376b76e"
METADATA = {
    "config.json": (6395, "6397cb6eca41b911d1dcab74e17941351057bd759284052a2331918ff6f9246c"),
    "index.json": (218323, "edb157dbf495e23f37377af4a628a9ad13c4ee7937f93ccb36ec9e9a19940f16"),
    "header.json": (318220, "4d1819ededded19812059691c0f77cfec0fe10c9c89c720950f919bfc26544a7"),
}
GUARD_WORDS = bytes.fromhex("3100175a") * 7
SCOPE = ("Historical whole-container anchor and matching selected text bytes; "
         "expected resident storage only; no current-path or excluded-pass2 assertion")


def check(condition, message):
    if not condition:
        raise ValueError(message)


def fields(pairs):
    result = {}
    for key, value in pairs:
        check(key not in result, "duplicate field")
        result[key] = value
    return result


def document(raw):
    return json.loads(raw, object_pairs_hook=fields)


def bounded(path, limit):
    with path.open("rb") as file:
        raw = file.read(limit + 1)
    check(len(raw) <= limit, "document bound")
    return raw


def region(reader, start, length):
    check(reader.seek(start) == start and reader.tell() == start, "source seek")
    remaining = length
    while remaining:
        raw = reader.read(min(8192, remaining))
        check(0 < len(raw) <= remaining, "truncated source region")
        remaining -= len(raw)
        yield raw


def packed_buffer(reader, offset, length, component=None):
    original = hashlib.sha256()
    guarded = hashlib.sha256()
    for raw in region(reader, offset, length):
        original.update(raw)
        guarded.update(raw)
        if component is not None:
            component.update(raw)
    padded = (length + 3) // 4 * 4
    guarded.update(bytes(padded - length))
    guarded.update(GUARD_WORDS)
    return dict(offset=offset, source_bytes=length, source_sha256=original.hexdigest(),
                padding_bytes=padded - length, guarded_bytes=padded + 28,
                guarded_sha256=guarded.hexdigest())


def converted_vector(reader, offset, length):
    original = hashlib.sha256()
    converted = hashlib.sha256()
    guarded = hashlib.sha256()
    negative_zeros = 0
    for raw in region(reader, offset, length):
        check(len(raw) % 2 == 0, "BF16 pair")
        original.update(raw)
        for index in range(0, len(raw), 2):
            value = struct.unpack("<f", b"\0\0" + raw[index:index + 2])[0]
            check(math.isfinite(value), "finite BF16 vector")
            fp32 = struct.pack("<f", value)
            converted.update(fp32)
            guarded.update(fp32)
            negative_zeros += value == 0 and math.copysign(1, value) < 0
    guarded.update(GUARD_WORDS)
    return original.hexdigest(), converted.hexdigest(), guarded.hexdigest(), negative_zeros


def expected_inventory(model, metadata_folder):
    raw_metadata = {}
    for name, (size, digest) in METADATA.items():
        raw = bounded(metadata_folder / name, size)
        check(len(raw) == size and hashlib.sha256(raw).hexdigest() == digest, "metadata identity")
        raw_metadata[name] = raw
    header = document(raw_metadata["header.json"])
    prefix = "language_model.model."
    text = {n: v for n, v in header.items() if n.startswith(prefix)}
    matrix_names = sorted(n for n, v in text.items() if v["dtype"] == "U32")
    vector_names = sorted(n for n, v in text.items() if len(v["shape"]) == 1)
    check((len(text), len(matrix_names), len(vector_names)) == (1096, 278, 262), "source inventory")
    check(set(matrix_names).isdisjoint(vector_names), "source roles")
    with model.open("rb") as reader:
        check(reader.seek(0) == 0, "header seek")
        prefix_bytes, header_bytes = reader.read(8), reader.read(318220)
        check(prefix_bytes == struct.pack("<Q", 318220)
              and header_bytes == raw_metadata["header.json"], "container header identity")
        whole = hashlib.sha256(prefix_bytes + header_bytes)
        components = {}
        cursor = 0
        for name, value in sorted(((n, v) for n, v in header.items() if n != "__metadata__"),
                                  key=lambda entry: entry[1]["data_offsets"][0]):
            first, last = value["data_offsets"]
            check(first == cursor < last, "full container coverage")
            start = first + 318228
            digest = hashlib.sha256()
            for raw in region(reader, start, last - first):
                whole.update(raw)
                digest.update(raw)
            if name in text:
                components[name] = dict(offset=start, bytes=last - first, sha256=digest.hexdigest())
            cursor = last
        check(cursor + 318228 == SOURCE_BYTES and reader.read(1) == b""
              and whole.hexdigest() == SOURCE_SHA, "container identity")
        matrices, vectors = [], []
        used = set()
        for name in matrix_names:
            desc = text[name]
            rows, packed_columns = desc["shape"]
            columns = packed_columns * 8
            check(name.endswith(".weight"), "matrix stem")
            stem = name.rsplit(".", 1)[0]
            roles = [("codes", name, columns // 2), ("scales", stem + ".scales", columns // 32),
                     ("biases", stem + ".biases", columns // 32)]
            check(columns % 64 == 0 and rows > 0, "matrix geometry")
            chunks = []
            component_hashes = {role: hashlib.sha256() for role, _, _ in roles}
            for first in range(0, rows, 16384):
                count = min(rows - first, 16384)
                buffers = {}
                for role, physical, stride in roles:
                    component = components[physical]
                    check(component["bytes"] == rows * stride, "source row coverage")
                    offset = component["offset"] + first * stride
                    buffers[role] = packed_buffer(reader, offset, count * stride, component_hashes[role])
                chunks.append(dict(first=first, rows=count, buffers=buffers))
            for role, physical, _ in roles:
                check(component_hashes[role].hexdigest() == components[physical]["sha256"],
                      "packed reread identity")
            used.update(physical for _, physical, _ in roles)
            matrices.append(dict(name=name, kind="matrix", rows=rows, columns=columns, chunks=chunks))
        for name in vector_names:
            desc = text[name]
            check(desc["dtype"] == "BF16", "vector dtype")
            count = desc["shape"][0]
            source = components[name]
            check(source["bytes"] == count * 2, "vector source size")
            original, converted, guarded, negative_zeros = converted_vector(reader, source["offset"], count * 2)
            check(original == source["sha256"], "vector reread identity")
            vectors.append(dict(name=name, kind="vector", count=count, source=source,
                                fp32_bytes=count * 4, fp32_sha256=converted,
                                negative_zeros=negative_zeros, guarded_bytes=count * 4 + 28,
                                guarded_sha256=guarded))
            used.add(name)
    check(used == set(text), "every source component consumed")
    chunks = [c for m in matrices for c in m["chunks"]]
    buffers = [b for c in chunks for b in c["buffers"].values()]
    check((len(chunks), len(buffers)) == (308, 924), "chunk inventory")
    maximum = max(sum(v["source_bytes"] for v in c["buffers"].values()) for c in chunks)
    accounting = dict(matrix_inline_owner_bytes=31136, chunk_descriptor_bytes=34496,
                      packed_guarded_buffer_bytes=sum(b["guarded_bytes"] for b in buffers),
                      vector_guarded_buffer_bytes=sum(v["guarded_bytes"] for v in vectors),
                      maximum_three_source_bytes=maximum, maximum_three_source_with_descriptors=maximum + 72,
                      maximum_vector_raw_bytes=max(v["source"]["bytes"] for v in vectors),
                      context_policy_bytes=67108864, cold_metadata_policy_bytes=67108864,
                      maximum_vector_conversion_overlap_bytes=max(v["source"]["bytes"] for v in vectors),
                      aggregate_model_load_policy_bytes=None,
                      vector_owner_abi_bytes=None, bundle_owner_abi_bytes=None,
                      new_bundle_actual_capacities_admitted=False)
    accounting["known_matrix_resident_requested_bytes"] = (accounting["packed_guarded_buffer_bytes"]
                                                          + 31136 + 34496)
    return dict(format="toklet-gemma-resident-storage-reference-v1",
                model=dict(bytes=SOURCE_BYTES, sha256=SOURCE_SHA),
                metadata={n:dict(bytes=s, sha256=h) for n, (s, h) in METADATA.items()},
                source_components=components, matrices=matrices, vectors=vectors,
                counts=dict(matrices=278, vectors=262, source_components=1096,
                            packed_chunks=308, guarded_gpu_allocations=1186),
                accounting=accounting, reference_only=True, engine_or_gpu_executed=False,
                complete_model_storage_admitted=False, original_bf16_source_qualified=False,
                whole_graph_qualified=False, performance_qualified=False, phone_qualified=False, scope=SCOPE)


def validate(observation, expected):
    def ordered(value):
        result = dict(value)
        for role in ("matrices", "vectors"):
            entries = result[role]
            check(type(entries) is list and len({v["name"] for v in entries}) == len(entries),
                  "unique logical names")
            result[role] = sorted(entries, key=lambda v: v["name"])
        return result
    check(json.dumps(ordered(observation), sort_keys=True, allow_nan=False)
          == json.dumps(ordered(expected), sort_keys=True, allow_nan=False),
          "resident storage inventory differs")
    return dict(passed=True, reference_only=True, source_components=1096, packed_chunks=308,
                guarded_gpu_allocations=1186, numerical_graph_or_gpu_qualified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("observation", type=Path)
    args = parser.parse_args()
    observation = document(bounded(args.observation, 4 * 1024**2))
    print(json.dumps(validate(observation, expected_inventory(args.model, args.metadata))))


if __name__ == "__main__":
    main()
