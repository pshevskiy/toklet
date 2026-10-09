"""Explicitly build the pinned comparison worker or convert/verify FP32 GGUF."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import prepare

SOURCE = prepare.ROOT / "artifacts/llama.cpp"
MODEL = prepare.ROOT / "artifacts/llama-qwen-f32"
BUILD = prepare.ROOT / "artifacts/llama-bench-build"
TOOLS = prepare.ROOT / "artifacts/bench-tools"


def source_identity():
    pinned = prepare.load_json(prepare.SOURCES)["baselines"]["llama.cpp_commit"]
    actual = subprocess.check_output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(SOURCE), "status", "--porcelain"], text=True).strip()
    if actual != pinned or dirty:
        raise ValueError("llama.cpp source must be clean at the frozen revision")
    return pinned


def build():
    revision = source_identity()
    if BUILD.exists():
        raise ValueError("Refusing to overwrite a comparison build")
    cmake = TOOLS / "cmake/data/bin/cmake"
    ninja = TOOLS / "bin/ninja"
    if not cmake.exists() or not ninja.exists():
        raise ValueError("Install pinned CMake 4.1.3/Ninja 1.13.0 wheels into artifacts/bench-tools")
    configure = [str(cmake), "-S", str(prepare.ROOT / "tools/bench/llama"), "-B", str(BUILD),
                 "-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={ninja}", f"-DLLAMA_SOURCE={SOURCE}",
                 "-DCMAKE_BUILD_TYPE=Release"]
    start = time.perf_counter()
    subprocess.run(configure, check=True, timeout=120)
    subprocess.run([str(cmake), "--build", str(BUILD), "--target", "toklet-llama-bench", "-j", "2"], check=True, timeout=600)
    binary = BUILD / "bin/toklet-llama-bench"
    # A subdirectory can set its own output location; our target defaults here.
    if not binary.exists():
        binary = BUILD / "toklet-llama-bench"
    if not binary.exists():
        raise ValueError("Worker executable missing")
    manifest = {"llama_revision": revision, "configure_command": configure,
                "build_seconds": time.perf_counter() - start,
                "cmake": subprocess.check_output([str(cmake), "--version"], text=True).strip(),
                "ninja": subprocess.check_output([str(ninja), "--version"], text=True).strip(),
                "compiler": subprocess.check_output(["clang", "--version"], text=True).strip(),
                "binary": str(binary.relative_to(prepare.ROOT)), "binary_sha256": prepare.digest(binary),
                "worker_sha256": prepare.digest(prepare.ROOT/"tools/bench/llama/worker.cpp"),
                "cmake_source_sha256": prepare.digest(prepare.ROOT/"tools/bench/llama/CMakeLists.txt")}
    prepare.write_json(BUILD / "manifest.json", manifest)


def checked_build():
    revision = source_identity()
    manifest = prepare.load_json(BUILD / "manifest.json")
    for key, path in (("binary_sha256", prepare.ROOT / manifest["binary"]),
                      ("worker_sha256", prepare.ROOT / "tools/bench/llama/worker.cpp"),
                      ("cmake_source_sha256", prepare.ROOT / "tools/bench/llama/CMakeLists.txt")):
        if prepare.digest(path) != manifest[key]:
            raise ValueError("Comparison executable or build sources changed; rebuild explicitly")
    if manifest["llama_revision"] != revision:
        raise ValueError("Comparison build uses a different upstream revision")
    return manifest


def mapped_name(name):
    if name == "model.embed_tokens.weight": return "token_embd.weight"
    if name == "model.norm.weight": return "output_norm.weight"
    pieces = name.split(".")
    if len(pieces) not in (5, 6) or pieces[:2] != ["model", "layers"]:
        raise ValueError(f"Unsupported source tensor {name}")
    layer = int(pieces[2])
    if pieces[3] in ("input_layernorm", "post_attention_layernorm"):
        op = "attn_norm" if pieces[3] == "input_layernorm" else "ffn_norm"
    elif pieces[3] == "self_attn":
        op = {"q_proj":"attn_q", "k_proj":"attn_k", "v_proj":"attn_v", "o_proj":"attn_output"}[pieces[4]]
    elif pieces[3] == "mlp":
        op = {"gate_proj":"ffn_gate", "up_proj":"ffn_up", "down_proj":"ffn_down"}[pieces[4]]
    else:
        raise ValueError(f"Unsupported source tensor {name}")
    return f"blk.{layer}.{op}.{pieces[-1]}"


def verify_payload(path):
    import numpy as np
    sys.path.insert(0, str(SOURCE / "gguf-py"))
    from gguf import GGUFReader, GGMLQuantizationType
    source = prepare.load_json(prepare.SOURCES)["model"]["weights"]
    prepare.validate_source(source)
    reader = GGUFReader(path)
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    if len(tensors) != len(reader.tensors):
        raise ValueError("Duplicate converted tensor names")
    tensor_bytes = 0
    with (prepare.ROOT / source["file"]).open("rb") as weights:
        header_size = int.from_bytes(weights.read(8), "little")
        if not 0 < header_size <= 16*1024*1024:
            raise ValueError("Invalid source header")
        header = json.loads(weights.read(header_size))
        payload_start = 8 + header_size
        for name, item in header.items():
            if name == "__metadata__": continue
            tensor = tensors.pop(mapped_name(name))
            if item["dtype"] != "BF16" or tensor.tensor_type != GGMLQuantizationType.F32:
                raise ValueError("Expected exact BF16-source to FP32 conversion")
            if list(tensor.shape)[::-1] != item["shape"]:
                raise ValueError(f"Converted shape differs for {name}")
            flat = tensor.data.reshape(-1)
            begin, end = item["data_offsets"]
            if end - begin != flat.size * 2:
                raise ValueError("Source count mismatch")
            weights.seek(payload_start + begin)
            for offset in range(0, flat.size, 32768):
                count = min(32768, flat.size - offset)
                raw = weights.read(count * 2)
                if len(raw) != count * 2:
                    raise ValueError("Truncated source tensor")
                reconstructed = (np.frombuffer(raw, dtype="<u2").astype("<u4") << 16)
                actual = flat[offset:offset + count].view("<u4")
                if not np.array_equal(reconstructed, actual):
                    raise ValueError(f"Converted values differ for {name} at {offset}")
            tensor_bytes += flat.size * 4
    if tensors or tensor_bytes != 1976131072:
        raise ValueError("Converted tensor inventory differs from pinned source")
    return {"tensor_count": len(reader.tensors), "fp32_tensor_bytes": tensor_bytes}


def convert():
    revision = source_identity()
    if MODEL.exists():
        raise ValueError("Refusing to overwrite converted artifact")
    sources = prepare.load_json(prepare.SOURCES)
    prepare.validate_source(sources["model"]["weights"])
    for asset in sources["assets"]:
        if asset["role"] == "model_metadata_or_tokenizer": prepare.validate_source(asset)
    MODEL.mkdir(parents=True)
    path = MODEL / "model.gguf"
    converter = SOURCE / "convert_hf_to_gguf.py"
    command = [sys.executable, str(converter), str(prepare.ROOT/"models/Qwen2.5-0.5B-Instruct"), "--outtype", "f32", "--outfile", str(path)]
    start = time.perf_counter()
    subprocess.run(command, check=True, timeout=600, env={**os.environ, "OMP_NUM_THREADS":"1", "PYTHONDONTWRITEBYTECODE":"1"})
    seconds = time.perf_counter() - start
    checked = verify_payload(path)
    prepare.write_json(MODEL / "manifest.json", {"llama_revision": revision,
        "source_sha256": sources["model"]["weights"]["sha256"], "converter_sha256": prepare.digest(converter),
        "conversion_seconds": seconds, "command": command, "gguf_sha256": prepare.digest(path),
        "gguf_bytes": path.stat().st_size, **checked})


def check():
    source_identity()
    manifest = prepare.load_json(MODEL/"manifest.json")
    path = MODEL/"model.gguf"
    if prepare.digest(path) != manifest["gguf_sha256"] or path.stat().st_size != manifest["gguf_bytes"]:
        raise ValueError("Converted artifact identity changed")
    checked = verify_payload(path)
    if any(manifest[key] != value for key, value in checked.items()):
        raise ValueError("Converted payload manifest differs")
    print("Verified exact FP32 converted payload")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--build", action="store_true")
    mode.add_argument("--convert", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.build: build()
    elif args.convert: convert()
    else: check()
