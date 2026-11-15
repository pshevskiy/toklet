#!/usr/bin/env python3
"""One pinned Qwen3.5-4B conversion witness; no model execution or fit claim.

This reads complete authenticated files twice, then compares every retained
array with two frozen converter arithmetic witnesses. A single witness must
match the entire artifact. Names, shapes and transformations are reconstructed
here rather than accepted from a producer's correspondence table.
"""

import argparse
from contextlib import ExitStack
import hashlib
import importlib
import json
import math as scalar_math
import os
from pathlib import Path
import re
import stat
import struct
import sys
import time


VERSION = "toklet-qwen4-parent-correspondence-v1"
MATH_SHA256 = "7340a71dbdef174815cdac9c5d0a51ed7875173f2fcde8c57d42c2ef3c384c28"
PROFILES = ("cpu_rint", "metal_round")
IO_LIMIT = 65536
HEADER_LIMIT = 512 * 1024
DOCUMENT_LIMIT = 2 * 1024 * 1024
WINDOW_LIMIT = 16 * 1024 * 1024
ROW_BLOCK_MAX = 128
GROUP_BATCH_MAX = 256
EXAMPLE_LIMIT = 32
U64 = 2**64 - 1
COVERAGE_TARGET = {
    "parent_arrays": 426,
    "matrices": 249,
    "unquantized_arrays": 177,
    "child_descriptors": 924,
    "groups": 65699840,
    "matrix_row_blocks": 10820,
    "unquantized_elements": 961536,
}
PLAN_SHA256 = "50e6994b9ead0df5730da424732f73770c498d0cb29af8ea42af811379c0ee07"
PINNED_INPUTS = {
    "correspondence-api-1.json": "a5ae98d744abc59de123cab9dc801f881d1839e3f98f137d2b22ae772e434410",
    "affine-oracle-design-2.json": "538fa91ad88542d84119615b5401425051b7648b0a43c602dde860001caee2f3",
    "converter-audit-1.json": "95a38182476eb49baa1ebec754ad3a3697850836949b5d2e40ea81ead7264de1",
    "converter-source-1/actual-parent-child-map-1.json": "80f3e11875936883e9842d2136b6f21ca16a4f12c864cdddb182f3689016e0d8",
    "source-profile-contract-2.json": "ce6330415a707c99fdf080e3a26d523a51b9175d2b9a9c952526bfc6b381ace2",
    "source-auth-plan-2.json": PLAN_SHA256,
}
FILES = (
    {
        "label": "new4", "repository": "TheWirelessPhoenix/Qwen3.5-4B-abliterated_4bit",
        "revision": "916d9f7c2e27ed41e2b9ed39f3a2badccb2bbc7c",
        "name": "model.safetensors", "bytes": 2367237149,
        "sha256": "244b47c46b759664c223a996046857720826a388bdf3969620a8f12ebb6e14e0",
        "header_bytes": 119829,
        "header_sha256": "7284f68df36a7761569b48321066e92d616306e33b4a9487029673a0ff344cb2",
    },
    {
        "label": "new4-parent-reference", "repository": "huihui-ai/Huihui-Qwen3.5-4B-abliterated",
        "revision": "5581467dfd52bf338c782006a6cdce05c42594be",
        "name": "model-00001-of-00002.safetensors", "bytes": 4972947968,
        "sha256": "9d95b71dae3f683f8c7a3fc78c8ff458057dce97d5f99d37d49c0265e594560d",
        "header_bytes": 29304,
        "header_sha256": "fcc6cad7b5167728eefa04f4ff5a3bc2f1108bbcea53a9eff23f3cf1c6956cb6",
    },
    {
        "label": "new4-parent-reference", "repository": "huihui-ai/Huihui-Qwen3.5-4B-abliterated",
        "revision": "5581467dfd52bf338c782006a6cdce05c42594be",
        "name": "model-00002-of-00002.safetensors", "bytes": 4105672464,
        "sha256": "c6e0edcf7c7f454181d7378d8ed7188d24d068022bf42be30d3cbbfc2ae1dd90",
        "header_bytes": 60040,
        "header_sha256": "cc1f79b3b09725cbc5385d8d0d9df40af346232744fa858493e2d64aea6211ac",
    },
)


def need(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, minimum=0, maximum=U64):
    return type(value) is int and minimum <= value <= maximum


def _pairs(items):
    result = {}
    for key, value in items:
        need(key not in result, "Duplicate JSON key")
        result[key] = value
    return result


def _constant(_):
    raise ValueError("Nonfinite JSON number")


def _finite_float(value):
    result = float(value)
    need(scalar_math.isfinite(result), "Nonfinite JSON number")
    return result


def decode_document(raw, limit=DOCUMENT_LIMIT):
    """Decode bounded UTF-8 JSON, rejecting duplicate keys and nonfinite numbers."""
    need(type(raw) is bytes and integer(limit, 1, DOCUMENT_LIMIT), "Bounded bytes required")
    need(0 < len(raw) <= limit, "JSON document size outside bound")
    result = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs,
                        parse_constant=_constant, parse_float=_finite_float)
    need(type(result) is dict, "JSON object required")
    return result


def validate_header(raw, container_bytes):
    """Validate raw header JSON (not its 8-byte prefix) and complete payload spans."""
    need(integer(container_bytes, 9, 2**63 - 1), "Invalid container length")
    header = decode_document(raw, HEADER_LIMIT)
    need(len(header) <= 925, "Too many tensor descriptors")
    payload = container_bytes - 8 - len(raw)
    need(payload > 0, "No container payload")
    spans = []
    for name, value in header.items():
        need(type(name) is str and len(name) <= 256, "Invalid descriptor name")
        if name == "__metadata__":
            need(type(value) is dict and len(value) <= 16
                 and all(type(k) is str and type(v) is str and len(k) <= 256
                         and len(v) <= 4096 for k, v in value.items()), "Invalid header metadata")
            continue
        need(re.fullmatch(r"[A-Za-z0-9_.]+", name) is not None, "Invalid tensor name")
        need(type(value) is dict and set(value) == {"dtype", "shape", "data_offsets"},
             "Exact tensor descriptor required")
        dtype, shape, offsets = value["dtype"], value["shape"], value["data_offsets"]
        need(dtype in ("BF16", "F32", "U32"), "Unsupported source dtype")
        need(type(shape) is list and 1 <= len(shape) <= 8, "Positive ranked shape required")
        count = 1
        for dimension in shape:
            need(integer(dimension, 1, 2**31 - 1), "Invalid tensor dimension")
            count *= dimension
            need(count <= payload, "Tensor size overflow")
        need(type(offsets) is list and len(offsets) == 2
             and all(integer(n, 0, payload) for n in offsets), "Invalid tensor range")
        width = 2 if dtype == "BF16" else 4
        need(offsets[1] - offsets[0] == count * width, "Tensor shape/range mismatch")
        spans.append((offsets[0], offsets[1], name))
    need(spans, "Empty tensor header")
    cursor = 0
    for first, last, _ in sorted(spans):
        need(first == cursor, "Tensor alias, overlap or gap")
        cursor = last
    need(cursor == payload, "Incomplete payload coverage")
    return header


def _specification():
    """Independent fixed 32-layer text geometry; no producer table is consumed."""
    specs = {"model.embed_tokens.weight": ("matrix", [248320, 2560]),
             "model.norm.weight": ("norm_plus_one", [2560])}
    for layer in range(32):
        base = f"model.layers.{layer}."
        for role in ("input_layernorm", "post_attention_layernorm"):
            specs[base + role + ".weight"] = ("norm_plus_one", [2560])
        for role, shape in (("gate_proj", [9216, 2560]), ("up_proj", [9216, 2560]),
                            ("down_proj", [2560, 9216])):
            specs[base + "mlp." + role + ".weight"] = ("matrix", shape)
        if layer % 4 != 3:
            prefix = base + "linear_attn."
            for role, shape in (("in_proj_qkv", [8192, 2560]), ("in_proj_z", [4096, 2560]),
                                ("in_proj_a", [32, 2560]), ("in_proj_b", [32, 2560]),
                                ("out_proj", [2560, 4096])):
                specs[prefix + role + ".weight"] = ("matrix", shape)
            for role, shape in (("A_log", [32]), ("dt_bias", [32]), ("norm.weight", [128])):
                specs[prefix + role] = ("copy", shape)
            specs[prefix + "conv1d.weight"] = ("conv_unit_axis", [8192, 1, 4])
        else:
            prefix = base + "self_attn."
            for role, shape in (("q_proj", [8192, 2560]), ("k_proj", [1024, 2560]),
                                ("v_proj", [1024, 2560]), ("o_proj", [2560, 4096])):
                specs[prefix + role + ".weight"] = ("matrix", shape)
            for role in ("q_norm", "k_norm"):
                specs[prefix + role + ".weight"] = ("norm_plus_one", [256])
    return specs


def _elements(shape):
    count = 1
    for value in shape:
        need(integer(value, 1, 2**31 - 1), "Invalid shape")
        count *= value
        need(count <= 2**63 - 1, "Shape product overflow")
    return count


def _descriptor(value):
    need(type(value) is dict and set(value) == {"dtype", "shape", "data_offsets"}
         and value["dtype"] in ("BF16", "F32", "U32"), "Invalid mapped descriptor")
    shape, offsets = value["shape"], value["data_offsets"]
    need(type(shape) is list and 1 <= len(shape) <= 8, "Invalid mapped shape")
    elements = _elements(shape)
    need(type(offsets) is list and len(offsets) == 2
         and all(integer(n, 0, 2**63 - 1) for n in offsets)
         and offsets[1] - offsets[0] == elements * (2 if value["dtype"] == "BF16" else 4),
         "Invalid mapped range")


def derive_mapping(parent_shards, child):
    """Join all raw fixed-parent descriptors to the independently specified child."""
    need(type(parent_shards) is dict and len(parent_shards) == 2
         and type(child) is dict, "Two parent shards and child header required")
    parents = {}
    for shard, header in parent_shards.items():
        need(type(shard) is str and type(header) is dict, "Invalid parent shard")
        for name, descriptor in header.items():
            if name == "__metadata__":
                continue
            need(type(name) is str, "Invalid mapped parent name")
            _descriptor(descriptor)
            need(name not in parents, "Duplicate parent tensor across shards")
            parents[name] = (shard, descriptor)
    specs = _specification()
    wanted = {"model.language_model." + k.removeprefix("model.") for k in specs}
    excluded = set(parents) - wanted
    need(len(parents) == 723 and len(excluded) == 297
         and all(k.startswith(("vision_tower", "model.visual")) for k in excluded)
         and wanted <= parents.keys(), "Fixed parent text/vision coverage differs")
    result, used = [], set()
    for canonical, (kind, shape) in sorted(specs.items()):
        parent_name = "model.language_model." + canonical.removeprefix("model.")
        shard, descriptor = parents[parent_name]
        need(type(descriptor) is dict and descriptor.get("dtype") == "BF16"
             and descriptor.get("shape") == shape, "Parent shape/dtype differs")
        name = "language_model." + canonical
        child_specs = [(name, "BF16", shape)]
        if kind == "matrix":
            rows, columns = shape
            base = name.removesuffix(".weight")
            child_specs = [(name, "U32", [rows, columns // 8]),
                           (base + ".scales", "BF16", [rows, columns // 64]),
                           (base + ".biases", "BF16", [rows, columns // 64])]
        elif kind == "conv_unit_axis":
            child_specs = [(name, "BF16", [8192, 4, 1])]
        children = []
        for child_name, dtype, child_shape in child_specs:
            value = child.get(child_name)
            _descriptor(value)
            need(type(value) is dict and value.get("dtype") == dtype
                 and value.get("shape") == child_shape and child_name not in used,
                 "Child name/shape/dtype/alias differs")
            used.add(child_name)
            children.append({"name": child_name, **value})
        result.append({"parent_name": parent_name, "parent_shard": shard,
                       "parent_descriptor": descriptor, "kind": kind,
                       "child": children, "elements": _elements(shape)})
    need(len(result) == 426 and len(used) == 924
         and used == set(child) - {"__metadata__"}, "Complete child mapping required")
    return result


def complete_profiles(profile_mismatches, unquant_mismatches, coverage):
    """A complete witness cannot be assembled by mixing per-tensor profiles."""
    need(type(profile_mismatches) is dict and set(profile_mismatches) == set(PROFILES)
         and all(integer(n) for n in profile_mismatches.values())
         and integer(unquant_mismatches), "Strict profile mismatch counters required")
    need(type(coverage) is dict and set(coverage) == set(COVERAGE_TARGET)
         and all(type(coverage[k]) is int and coverage[k] == n
                 for k, n in COVERAGE_TARGET.items()), "Incomplete correspondence coverage")
    if unquant_mismatches:
        return []
    return [p for p in PROFILES if profile_mismatches[p] == 0]


def read_exact_at(handle, offset, count):
    """One bounded regular-file-style read; offsets do not authorize huge reads."""
    need(integer(offset, 0, 2**63 - 1) and integer(count, 0, IO_LIMIT)
         and offset + count <= 2**63 - 1, "Invalid bounded read range")
    try:
        need(handle.seekable() is True, "Seekable source required")
        position = handle.seek(offset)
        need(integer(position, 0, 2**63 - 1) and position == offset, "Source seek differs")
        result = handle.read(count)
    except (OSError, TypeError, AttributeError) as error:
        raise ValueError("Bounded source read failed") from error
    need(type(result) is bytes and len(result) == count, "Short or invalid source read")
    return result


def _span(handle, offset, count, maximum=WINDOW_LIMIT):
    need(integer(count, 0, maximum) and maximum <= WINDOW_LIMIT, "Window exceeds bound")
    parts = bytearray(count)
    for position in range(0, count, IO_LIMIT):
        part = read_exact_at(handle, offset + position, min(IO_LIMIT, count - position))
        parts[position:position + len(part)] = part
    return parts


def _endpoint(value):
    return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns]


def _open_regular(path, expected_size=None, maximum=None):
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    try:
        value = os.fstat(fd)
        need(stat.S_ISREG(value.st_mode), "Regular input required")
        need(expected_size is None or value.st_size == expected_size, "Source length differs")
        need(maximum is None or value.st_size <= maximum, "Input exceeds bound")
        return os.fdopen(fd, "rb", buffering=0)
    except BaseException:
        os.close(fd)
        raise


def _metadata(path, expected_sha=None):
    with _open_regular(path, maximum=DOCUMENT_LIMIT) as handle:
        before = _endpoint(os.fstat(handle.fileno()))
        raw = bytes(_span(handle, 0, before[2], DOCUMENT_LIMIT))
        after = _endpoint(os.fstat(handle.fileno()))
    checksum = hashlib.sha256(raw).hexdigest()
    need(before == after and (expected_sha is None or checksum == expected_sha),
         "Metadata identity or endpoint differs")
    return decode_document(raw), {"path": str(path), "bytes": len(raw),
                                  "sha256": checksum, "endpoint": before}


def _binary_pin(path):
    with _open_regular(path, maximum=256 * 1024 * 1024) as handle:
        before = _endpoint(os.fstat(handle.fileno()))
        checksum, count = hashlib.sha256(), 0
        while True:
            part = handle.read(IO_LIMIT)
            if not part:
                break
            count += len(part)
            checksum.update(part)
        need(count == before[2] and _endpoint(os.fstat(handle.fileno())) == before,
             "Runtime binary endpoint changed")
    return {"kind": "binary", "path": str(path), "bytes": count,
            "sha256": checksum.hexdigest(), "endpoint": before}


def validate_authentication(receipt, artifact_root):
    """Validate fixed identities, not a producer's success flags or chosen paths."""
    need(type(receipt) is dict and receipt.get("format") == "toklet-qwen4-source-content-authentication-v2"
         and receipt.get("completed") is True and receipt.get("passed") is True
         and receipt.get("failure") is None and receipt.get("model_executed") is False
         and receipt.get("numerical_or_quality_or_fit_qualification") is False
         and receipt.get("individual_tensor_values_interpreted") is False
         and receipt.get("whole_goal_complete") is False
         and receipt.get("plan_sha256") == PLAN_SHA256, "Qualified fixed-source authentication receipt required")
    records = receipt.get("records")
    need(type(records) is list and len(records) == 3, "All three source receipts required")
    paths = []
    for record, fixed in zip(records, FILES):
        need(type(record) is dict, "Source receipt object required")
        for key in ("repository", "revision", "name"):
            need(record.get(key) == fixed[key], "Authentication source identity differs")
        need(type(record.get("expected_bytes")) is int
             and record["expected_bytes"] == fixed["bytes"]
             and record.get("expected_sha256") == fixed["sha256"]
             and record.get("received_sha256") == fixed["sha256"]
             and type(record.get("materialized_bytes")) is int
             and record["materialized_bytes"] == fixed["bytes"]
             and record.get("passed") is True and record.get("failure") is None,
             "Complete fixed-source receipt differs")
        path = artifact_root / "content-auth-2" / fixed["label"] / fixed["name"]
        source = record.get("authenticated_file")
        need(type(source) is dict and source.get("path") == str(path)
             and type(source.get("bytes")) is int and source["bytes"] == fixed["bytes"]
             and source.get("sha256") == fixed["sha256"]
             and source.get("header_sha256") == fixed["header_sha256"],
             "Authenticated fixed path/header differs")
        before, after = source.get("before"), source.get("after")
        need(type(before) is list and len(before) == 5 and before == after
             and all(integer(n) for n in before) and before[2] == fixed["bytes"],
             "Stable authentication descriptor required")
        paths.append(path)
    return paths


def _hash_source(handle, fixed, endpoint):
    need(_endpoint(os.fstat(handle.fileno())) == endpoint, "Source endpoint changed")
    handle.seek(0)
    checksum, total = hashlib.sha256(), 0
    while True:
        part = handle.read(IO_LIMIT)
        need(type(part) is bytes and len(part) <= IO_LIMIT, "Invalid hash read")
        if not part:
            break
        total += len(part)
        need(total <= fixed["bytes"], "Source grew while hashing")
        checksum.update(part)
    need(total == fixed["bytes"] and checksum.hexdigest() == fixed["sha256"]
         and _endpoint(os.fstat(handle.fileno())) == endpoint,
         "Whole source SHA or stable endpoint differs")
    return {"bytes": total, "sha256": checksum.hexdigest(), "endpoint": endpoint}


class _Rejected(Exception):
    pass


def _compare(expected, actual, profile, owner, part, first, state, np):
    need(expected.shape == actual.shape and expected.dtype == actual.dtype,
         "Comparison dtype/shape differs")
    differing = expected != actual
    count = int(np.count_nonzero(differing))
    if count and len(state["examples"]) < EXAMPLE_LIMIT:
        indices = np.flatnonzero(differing.ravel())[:EXAMPLE_LIMIT - len(state["examples"])]
        a, b = expected.ravel(), actual.ravel()
        for index in indices:
            state["examples"].append({"profile": profile, "parent": owner, "part": part,
                                      "index": first + int(index), "expected_bits": int(a[index]),
                                      "actual_bits": int(b[index])})
    return count


def _eliminated(state):
    return state["unquant_mismatches"] > 0 or all(
        state["profiles"][p]["mismatches"] > 0 for p in PROFILES)


def _values(mapping, parent_files, origins, child_file, child_origin, math, np, early_reject, state):
    need(sys.byteorder == "little" and np.dtype("uint16").byteorder in ("=", "<"),
         "Qualified little-endian runtime required")
    for row in mapping:
        parent = parent_files[row["parent_shard"]]
        descriptor = row["parent_descriptor"]
        offset = origins[row["parent_shard"]] + descriptor["data_offsets"][0]
        owner = row["parent_name"]
        stream = {"parent": owner, "kind": row["kind"], "complete": False,
                  "parent_bytes_read": 0, "child_bytes_read": 0}
        state["streams"].append(stream)
        if row["kind"] == "matrix":
            rows, columns = descriptor["shape"]
            groups_per_row = columns // 64
            for first_row in range(0, rows, ROW_BLOCK_MAX):
                count = min(ROW_BLOCK_MAX, rows - first_row)
                raw = _span(parent, offset + first_row * columns * 2, count * columns * 2)
                source = np.frombuffer(raw, dtype="<u2").reshape(-1, 64)
                need(not bool(np.any((source & 0x7F80) == 0x7F80)), "Nonfinite parent matrix")
                actual = []
                for field, width in zip(row["child"], (columns // 8 * 4, groups_per_row * 2, groups_per_row * 2)):
                    data = _span(child_file, child_origin + field["data_offsets"][0] + first_row * width, count * width)
                    dtype = "<u4" if field["dtype"] == "U32" else "<u2"
                    actual.append(np.frombuffer(data, dtype=dtype))
                    stream["child_bytes_read"] += len(data)
                stream["parent_bytes_read"] += len(raw)
                code_actual = actual[0].reshape(-1, 8)
                for start in range(0, source.shape[0], GROUP_BATCH_MAX):
                    n = min(GROUP_BATCH_MAX, source.shape[0] - start)
                    for profile in PROFILES:
                        try:
                            result = math.quantize_groups(source[start:start + n], profile)
                        except ValueError as error:
                            state["profiles"][profile]["mismatches"] += n
                            state["profiles"][profile]["invalid_groups"] += n
                            state["profiles"][profile]["groups"] += n
                            if len(state["examples"]) < EXAMPLE_LIMIT:
                                state["examples"].append({"profile": profile, "parent": owner,
                                                          "part": "invalid_profile_arithmetic",
                                                          "groups": n, "reason": str(error)[:256]})
                            continue
                        need(type(result) is tuple and len(result) == 3, "Exact math tuple required")
                        codes, scales, biases = result
                        wanted = (code_actual[start:start + n], actual[1][start:start + n], actual[2][start:start + n])
                        for part, expected, observed, multiplier in zip(
                                ("codes", "scales", "biases"), (codes, scales, biases), wanted, (8, 1, 1)):
                            state["profiles"][profile]["mismatches"] += _compare(
                                expected, observed, profile, owner, part,
                                (first_row * groups_per_row + start) * multiplier, state, np)
                        state["profiles"][profile]["groups"] += n
                    state["coverage"]["groups"] += n
                    if early_reject and _eliminated(state):
                        raise _Rejected("Both whole-artifact canonical profiles disqualified")
                state["coverage"]["matrix_row_blocks"] += 1
            state["coverage"]["matrices"] += 1
            state["coverage"]["child_descriptors"] += 3
        else:
            field = row["child"][0]
            for first in range(0, row["elements"], IO_LIMIT // 2):
                count = min(IO_LIMIT // 2, row["elements"] - first)
                raw = read_exact_at(parent, offset + first * 2, count * 2)
                target = read_exact_at(child_file, child_origin + field["data_offsets"][0] + first * 2, count * 2)
                source = np.frombuffer(raw, dtype="<u2")
                actual = np.frombuffer(target, dtype="<u2")
                need(not bool(np.any((source & 0x7F80) == 0x7F80))
                     and not bool(np.any((actual & 0x7F80) == 0x7F80)), "Nonfinite unquantized array")
                expected = math.add_one_bf16(source) if row["kind"] == "norm_plus_one" else source
                mismatch = _compare(expected, actual, "shared", owner, row["kind"], first, state, np)
                state["unquant_mismatches"] += mismatch
                state["coverage"]["unquantized_elements"] += count
                stream["parent_bytes_read"] += len(raw)
                stream["child_bytes_read"] += len(target)
                if early_reject and _eliminated(state):
                    raise _Rejected("Shared unquantized transformation disqualified")
            state["coverage"]["unquantized_arrays"] += 1
            state["coverage"]["child_descriptors"] += 1
        stream["complete"] = True
        state["coverage"]["parent_arrays"] += 1


def run(artifact_root, authentication_receipt, authentication_sha256, output, early_reject=False):
    """Run only the fixed case; outer guard owns memory/deadline and publication."""
    need(type(early_reject) is bool, "Boolean early-reject policy required")
    need(re.fullmatch(r"[0-9a-f]{64}", authentication_sha256 or "") is not None,
         "Explicit authentication receipt SHA required")
    artifact_root = Path(artifact_root).resolve()
    authentication_receipt, output = Path(authentication_receipt), Path(output)
    need(not output.exists() and output.parent.is_dir(), "Fresh report in existing directory required")
    started = time.monotonic_ns()
    state = {"coverage": dict.fromkeys(COVERAGE_TARGET, 0), "unquant_mismatches": 0,
             "profiles": {p: {"groups": 0, "mismatches": 0, "invalid_groups": 0} for p in PROFILES},
             "examples": [], "streams": []}
    report = {"format": VERSION, "status": "incomplete", "completed": False,
              "coverage_complete": False, "source_correspondence_passed": False,
              "matching_whole_artifact_profiles": [], "failure": None,
              "early_reject_policy": early_reject, "source_before": [], "source_after": [],
              "expected_fixed_files": list(FILES),
              "input_pins": [], "runtime": {}, "observations": state,
              "historical_converter_execution": "unavailable",
              "historical_backend_dependency_FENV_FTZ": "unavailable",
              "model_executed": False, "model_numerical_qualified": False,
              "quality_qualified": False, "memory_fit_qualified": False,
              "performance_qualified": False, "adoption_qualified": False,
              "whole_goal_complete": False}
    handles, source_endpoints, metadata_pins = [], [], []
    with ExitStack() as stack:
        try:
            for relative, checksum in PINNED_INPUTS.items():
                _, pinned = _metadata(artifact_root / relative, checksum)
                metadata_pins.append(pinned)
            receipt, auth_pin = _metadata(authentication_receipt, authentication_sha256)
            metadata_pins.append(auth_pin)
            paths = validate_authentication(receipt, artifact_root)
            headers = []
            for fixed, path, prior in zip(FILES, paths, receipt["records"]):
                handle = stack.enter_context(_open_regular(path, expected_size=fixed["bytes"]))
                handles.append(handle)
                endpoint = _endpoint(os.fstat(handle.fileno()))
                source_endpoints.append(endpoint)
                need(endpoint == prior["authenticated_file"]["after"], "Source differs from authentication endpoint")
                report["source_before"].append(_hash_source(handle, fixed, endpoint))
                prefix = read_exact_at(handle, 0, 8)
                need(struct.unpack("<Q", prefix)[0] == fixed["header_bytes"], "Raw source header length differs")
                raw = bytes(_span(handle, 8, fixed["header_bytes"], HEADER_LIMIT))
                need(hashlib.sha256(raw).hexdigest() == fixed["header_sha256"], "Raw source header SHA differs")
                headers.append(validate_header(raw, fixed["bytes"]))
            parents = {FILES[i]["name"]: headers[i] for i in (1, 2)}
            mapping = derive_mapping(parents, headers[0])
            # This independent fixed source has unsanitized four-tap convolution
            # descriptors, so all81 specified RMSNorm shifts are mandatory.
            need(sum(row["kind"] == "norm_plus_one" for row in mapping) == 81
                 and sum(row["kind"] == "conv_unit_axis" for row in mapping) == 24,
                 "Exact typed sanitizer branch required")
            math_path = Path(__file__).resolve().with_name("qwen35_conversion_math.py")
            _, math_pin = _metadata_source(math_path)
            need(math_pin["sha256"] == MATH_SHA256, "Frozen math source SHA differs")
            _, driver_pin = _metadata_source(Path(__file__).resolve())
            metadata_pins.extend((math_pin, driver_pin))
            np = importlib.import_module("numpy")
            math = importlib.import_module("qwen35_conversion_math")
            need(Path(math.__file__).resolve() == math_path, "Math imported from another source path")
            need(math.MATH_VERSION == "toklet-qwen35-conversion-math-v1"
                 and tuple(math.PROFILES) == PROFILES and math.MAX_GROUPS == GROUP_BATCH_MAX
                 and math.MAX_ELEMENTS == 65536, "Frozen math API differs")
            _, numpy_pin = _metadata_source(Path(np.__file__).resolve())
            core = importlib.import_module("numpy._core._multiarray_umath")
            metadata_pins.extend((numpy_pin, _binary_pin(Path(core.__file__).resolve()),
                                  _binary_pin(Path(sys.executable).resolve())))
            report["runtime"] = {"python": sys.version, "executable": sys.executable,
                                 "numpy": np.__version__, "byteorder": sys.byteorder,
                                 "math_version": math.MATH_VERSION,
                                 "row_block_max": ROW_BLOCK_MAX, "group_batch_max": GROUP_BATCH_MAX,
                                 "owned_computation_window_limit": WINDOW_LIMIT}
            parent_files = {FILES[i]["name"]: handles[i] for i in (1, 2)}
            origins = {FILES[i]["name"]: 8 + FILES[i]["header_bytes"] for i in (1, 2)}
            _values(mapping, parent_files, origins, handles[0], 8 + FILES[0]["header_bytes"],
                    math, np, early_reject, state)
            matching = complete_profiles({p: state["profiles"][p]["mismatches"] for p in PROFILES},
                                         state["unquant_mismatches"], state["coverage"])
            need(all(state["profiles"][p]["groups"] == COVERAGE_TARGET["groups"] for p in PROFILES),
                 "Incomplete arithmetic profile coverage")
            report.update(completed=True, coverage_complete=True,
                          status="pass" if matching else "reject",
                          source_correspondence_passed=bool(matching),
                          matching_whole_artifact_profiles=matching)
        except _Rejected as error:
            report.update(status="reject", failure=str(error))
        except (Exception, KeyboardInterrupt) as error:
            report.update(status="incomplete", failure=f"{type(error).__name__}: {str(error)[:1024]}")
        finally:
            try:
                for handle, fixed, endpoint in zip(handles, FILES, source_endpoints):
                    report["source_after"].append(_hash_source(handle, fixed, endpoint))
                    path = artifact_root / "content-auth-2" / fixed["label"] / fixed["name"]
                    observed = os.stat(path, follow_symlinks=False)
                    need(stat.S_ISREG(observed.st_mode) and _endpoint(observed) == endpoint,
                         "Source path changed from stable descriptor")
                need(len(handles) == len(FILES), "All three stable sources required")
                for pinned in metadata_pins:
                    path = Path(pinned["path"])
                    if pinned.get("kind") == "binary":
                        again = _binary_pin(path)
                    else:
                        _, again = (_metadata_source(path) if path.suffix == ".py"
                                    else _metadata(path, pinned["sha256"]))
                    need(again == pinned, "Input source/metadata changed during correspondence")
            except (Exception, KeyboardInterrupt) as error:
                report.update(status="incomplete", completed=False, coverage_complete=False,
                              source_correspondence_passed=False, matching_whole_artifact_profiles=[],
                              failure=f"Endpoint closure: {type(error).__name__}: {str(error)[:1024]}")
    report["input_pins"] = metadata_pins
    report["elapsed_ns"] = time.monotonic_ns() - started
    raw = (json.dumps(report, allow_nan=False, separators=(",", ":")) + "\n").encode()
    need(len(raw) <= DOCUMENT_LIMIT, "Bounded report exceeded")
    partial = output.with_name(output.name + ".partial")
    fd = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as sink:
        sink.write(raw)
        sink.flush()
        os.fsync(sink.fileno())
    os.link(partial, output)
    partial.unlink()
    return report


def _metadata_source(path):
    """Pin executable source as text, never parse it as a JSON document."""
    with _open_regular(path, maximum=DOCUMENT_LIMIT) as handle:
        before = _endpoint(os.fstat(handle.fileno()))
        raw = bytes(_span(handle, 0, before[2], DOCUMENT_LIMIT))
        need(_endpoint(os.fstat(handle.fileno())) == before, "Executable source changed")
    raw.decode("utf-8")
    return raw, {"path": str(path), "bytes": len(raw),
                 "sha256": hashlib.sha256(raw).hexdigest(), "endpoint": before}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--authentication-receipt", type=Path, required=True)
    parser.add_argument("--authentication-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--early-reject", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = run(args.artifact_root, args.authentication_receipt,
                     args.authentication_sha256, args.output, args.early_reject)
        print(json.dumps({k: report[k] for k in ("status", "coverage_complete",
                         "source_correspondence_passed", "matching_whole_artifact_profiles")}))
        return 0 if report["source_correspondence_passed"] else 1
    except (Exception, KeyboardInterrupt) as error:
        print(f"{type(error).__name__}: {str(error)[:1024]}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
