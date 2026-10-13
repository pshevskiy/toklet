"""Explicit pinned Transformers FP32 forward probes and FP64 envelopes."""

import argparse
from importlib.metadata import version
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/bench"))
import prepare

# Establish each real gate from the independent reference before comparing Rust.
# Tiny-v1 remains unchanged. Each array freezes its resulting absolute limit.
POLICY_V1 = {"atol_floor": 1e-4, "envelope_multiplier": 2.0, "rtol": 1e-4,
          "absolute_limit": "max(atol_floor, envelope_multiplier * independent_FP32_FP64_max_absolute_error)",
          "reference": "pinned Transformers CPU FP32 eager; independent all-FP64 functional envelope"}
POLICY = {"atol_floor": 1e-4, "envelope_multiplier": 2.0, "rtol": 1e-4,
          "case_scope": "same named operation and layer across all frozen cases",
          "absolute_limit": "max(atol_floor, envelope_multiplier * pooled_FP32_FP64_max_absolute_error, envelope_multiplier * equivalent_prefix_FP32_max_absolute_error)",
          "reference": POLICY_V1["reference"]}
CASES = {"short": {"tokens": [9707, 11, 358, 2776], "offset": 0},
         "future_changed": {"tokens": [9707, 11, 358, 498], "offset": 0},
         "offset": {"tokens": [11, 358, 2776], "offset": 5},
         "single": {"tokens": [9707], "offset": 0}}
LAYERS = [0, 1, 23]


def check(directory, sources=None, layers=None):
    manifest = prepare.load_json(directory / "manifest.json")
    if manifest["policy"] not in [POLICY_V1, POLICY] or manifest["cases"] != CASES:
        raise ValueError("Real FP32 policy/cases changed")
    if manifest["model"] != (sources or prepare.load_json(prepare.SOURCES))["model"]:
        raise ValueError("Pinned model identity changed")
    if manifest["layers"] != (layers or LAYERS):
        raise ValueError("Selected reference layers changed")
    for name, array in manifest["arrays"].items():
        path = directory / array["file"]
        if path.stat().st_size != array["bytes"] or prepare.digest(path) != array["sha256"]:
            raise ValueError(f"Probe identity mismatch: {name}")
        if array["reference_envelope"]["maximum_tolerance_fraction"] > 1:
            raise ValueError(f"Reference envelope fails frozen real FP32 policy: {name}")
    print(f"Verified {len(manifest['arrays'])} pinned real FP32 probes", flush=True)


def export(full_directory, probes_directory, sources=None, model_directory=None, streamed=False, layers=None):
    for path in [full_directory, probes_directory]:
        if path.exists():
            raise ValueError("Refusing to overwrite a reference directory")
    sources = sources or prepare.load_json(prepare.SOURCES)
    layers = layers or LAYERS
    for package in ["transformers", "torch", "numpy"]:
        if version(package) != sources["baselines"][package]:
            raise ValueError(f"Version mismatch: {package}")
    prepare.validate_source(sources["model"]["weights"])
    for asset in sources["assets"]:
        if asset["role"] == "model_metadata_or_tokenizer":
            prepare.validate_source(asset)
    import numpy as np
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM
    from transformers.models.qwen2 import modeling_qwen2 as qwen

    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    model = AutoModelForCausalLM.from_pretrained(
        model_directory or ROOT / "models/Qwen2.5-0.5B-Instruct", dtype=torch.bfloat16 if streamed else torch.float32,
        attn_implementation="eager", local_files_only=True,
    ).eval()
    if {parameter.dtype for parameter in model.parameters()} != {torch.bfloat16 if streamed else torch.float32}:
        raise ValueError("Reference source parameter dtype differs")
    if model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr():
        raise ValueError("Reference must retain tied weights")
    parameter_scopes = None
    if streamed:
        from qwen_streamed import Parameters
        parameter_scopes = Parameters(model, torch)
    baseline = {}
    handles = []

    def hook(name, pre=False):
        def capture(module, inputs, output=None):
            baseline[name] = (inputs[0] if pre else output).detach().clone()[0]
        return capture

    handles.append(model.model.embed_tokens.register_forward_hook(hook("embedding")))
    handles.append(model.model.norm.register_forward_hook(hook("final_norm")))
    for index in layers:
        layer = model.model.layers[index]
        prefix = f"layers.{index}"
        for name, module in [("input_norm", layer.input_layernorm), ("post_attention_norm", layer.post_attention_layernorm),
                             ("q_projection", layer.self_attn.q_proj), ("k_projection", layer.self_attn.k_proj), ("v_projection", layer.self_attn.v_proj),
                             ("attention_projection", layer.self_attn.o_proj), ("gate_projection", layer.mlp.gate_proj),
                             ("up_projection", layer.mlp.up_proj), ("down_projection", layer.mlp.down_proj), ("output", layer)]:
            handles.append(module.register_forward_hook(hook(f"{prefix}.{name}")))
        handles.append(layer.post_attention_layernorm.register_forward_pre_hook(hook(f"{prefix}.attention_residual", True)))
        handles.append(layer.mlp.down_proj.register_forward_pre_hook(hook(f"{prefix}.gated", True)))

    original_attention = qwen.eager_attention_forward

    def capture_attention(module, query, key, value, attention_mask, scaling, **kwargs):
        output, probabilities = original_attention(module, query, key, value, attention_mask, scaling, **kwargs)
        if module.layer_idx in layers:
            prefix = f"layers.{module.layer_idx}"
            baseline[f"{prefix}.q_rotary"] = query[0].transpose(0, 1).contiguous().clone()
            baseline[f"{prefix}.k_rotary"] = key[0].transpose(0, 1).contiguous().clone()
            baseline[f"{prefix}.attention_scores"] = (query @ qwen.repeat_kv(key, module.num_key_value_groups).transpose(2, 3) * scaling)[0]
            baseline[f"{prefix}.attention_probabilities"] = probabilities[0].clone()
            baseline[f"{prefix}.attention_context"] = output[0].reshape(output.shape[1], -1).clone()
        return output, probabilities

    qwen.eager_attention_forward = capture_attention

    def precise(tokens, offset):
        """Independent full FP64 arithmetic over identical frozen FP32 weights."""
        c = model.config
        n, h, d = len(tokens), c.hidden_size, c.hidden_size // c.num_attention_heads
        values = {}
        x = model.model.embed_tokens(torch.tensor([tokens]))[0].double()
        values["embedding"] = x
        positions = torch.arange(offset, offset + n, dtype=torch.float64)
        frequencies = c.rope_parameters["rope_theta"] ** (-torch.arange(0, d, 2, dtype=torch.float64) / d)
        angles = positions[:, None, None] * frequencies[None, None, :]

        def norm(x, module):
            return x / torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + module.variance_epsilon) * module.weight.double()

        def linear(x, module):
            return F.linear(x, module.weight.double(), module.bias.double() if module.bias is not None else None)

        def rotate(x):
            first, second = x[..., :d // 2], x[..., d // 2:]
            return torch.cat([first * angles.cos() - second * angles.sin(), second * angles.cos() + first * angles.sin()], dim=-1)

        for index, layer in enumerate(model.model.layers):
            prefix = f"layers.{index}"
            def save(name, tensor):
                if index in layers:
                    values[f"{prefix}.{name}"] = tensor
                return tensor
            normalized = save("input_norm", norm(x, layer.input_layernorm))
            query = save("q_projection", linear(normalized, layer.self_attn.q_proj))
            key = save("k_projection", linear(normalized, layer.self_attn.k_proj))
            value = save("v_projection", linear(normalized, layer.self_attn.v_proj))
            query = save("q_rotary", rotate(query.reshape(n, c.num_attention_heads, d)))
            key = save("k_rotary", rotate(key.reshape(n, c.num_key_value_heads, d)))
            mapping = torch.arange(c.num_attention_heads) // (c.num_attention_heads // c.num_key_value_heads)
            scores = save("attention_scores", torch.einsum("thd,shd->hts", query, key[:, mapping]) / d**0.5)
            causal = torch.ones(n, n, dtype=torch.bool).tril()
            probabilities = save("attention_probabilities", torch.softmax(scores.masked_fill(~causal, -torch.inf), dim=-1))
            context = save("attention_context", torch.einsum("hts,shd->thd", probabilities, value.reshape(n, c.num_key_value_heads, d)[:, mapping]).reshape(n, h))
            attention = save("attention_projection", linear(context, layer.self_attn.o_proj))
            x = save("attention_residual", x + attention)
            normalized = save("post_attention_norm", norm(x, layer.post_attention_layernorm))
            gate = save("gate_projection", linear(normalized, layer.mlp.gate_proj))
            up = save("up_projection", linear(normalized, layer.mlp.up_proj))
            silu = save("gate_silu", gate * torch.sigmoid(gate))
            gated = save("gated", silu * up)
            down = save("down_projection", linear(gated, layer.mlp.down_proj))
            x = save("output", x + down)
        values["final_norm"] = norm(x, model.model.norm)
        values["logits"] = F.linear(values["final_norm"], model.lm_head.weight.double())
        return values

    arrays = {}
    for case, spec in CASES.items():
        baseline.clear()
        tokens, offset = spec["tokens"], spec["offset"]
        with torch.inference_mode():
            try:
                output = model(input_ids=torch.tensor([tokens]), position_ids=torch.arange(offset, offset + len(tokens))[None], use_cache=False)
            finally:
                if parameter_scopes:
                    parameter_scopes.restore()
                    parameter_scopes.assert_restored()
        baseline["logits"] = output.logits[0].detach()
        for index in layers:
            baseline[f"layers.{index}.gate_silu"] = F.silu(baseline[f"layers.{index}.gate_projection"])
        high_precision = precise(tokens, offset)
        if parameter_scopes:
            parameter_scopes.observe(f"functional_fp64_{case}")
        for name, value in baseline.items():
            actual = value.numpy().astype("<f4", copy=False)
            expected = high_precision[name].numpy().reshape(actual.shape)
            difference = np.abs(actual.astype(np.float64) - expected)
            absolute_error = float(np.max(difference))
            atol = max(POLICY["atol_floor"], POLICY["envelope_multiplier"] * absolute_error)
            fraction = float(np.max(difference / (atol + POLICY["rtol"] * np.abs(expected))))
            if not np.isfinite(actual).all():
                raise ValueError("Nonfinite reference probe")
            arrays[f"cases.{case}.{name}"] = (actual.copy(), expected.copy(), {"maximum_absolute_error": absolute_error, "maximum_tolerance_fraction": fraction, "atol": atol})
        print(f"Captured {case}: {len(baseline)} arrays across layers {layers}", flush=True)
    qwen.eager_attention_forward = original_attention
    for handle in handles:
        handle.remove()

    pooled, equivalent = {}, {}
    for key, (_, _, envelope) in arrays.items():
        name = key.split(".", 2)[2]
        pooled[name] = max(pooled.get(name, 0.0), envelope["maximum_absolute_error"])
    for first_name, first in CASES.items():
        for second_name, second in CASES.items():
            if first_name >= second_name or first["offset"] != second["offset"]:
                continue
            common = 0
            for a, b in zip(first["tokens"], second["tokens"]):
                if a != b:
                    break
                common += 1
            if not common:
                continue
            for name in pooled:
                a = arrays[f"cases.{first_name}.{name}"][0]
                b = arrays[f"cases.{second_name}.{name}"][0]
                if name.endswith((".attention_scores", ".attention_probabilities")):
                    a, b = a[:, :common, :common], b[:, :common, :common]
                else:
                    a, b = a[:common], b[:common]
                error = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
                equivalent[name] = max(equivalent.get(name, 0.0), error)
    for key, (actual, expected, envelope) in arrays.items():
        name = key.split(".", 2)[2]
        prefix_error = equivalent.get(name, 0.0)
        atol = max(POLICY["atol_floor"], POLICY["envelope_multiplier"] * pooled[name], POLICY["envelope_multiplier"] * prefix_error)
        envelope.update({"atol": atol, "pooled_fp32_fp64_max_absolute_error": pooled[name], "equivalent_prefix_max_absolute_error": prefix_error,
                         "maximum_tolerance_fraction": float(np.max(np.abs(actual.astype(np.float64) - expected) / (atol + POLICY["rtol"] * np.abs(expected))))})

    identity = {"schema_version": 1, "reference_id": "qwen-larger-forward-v1" if streamed and layers != LAYERS else "qwen-forward-v2", "model": sources["model"], "cases": CASES, "policy": POLICY, "layers": layers,
                "versions": {name: version(name) for name in ["transformers", "torch", "numpy"]}, "arrays": {},
                "exporter_sha256": prepare.digest(Path(__file__)), "dtype": "FP32", "device": "CPU", "attention": "eager", "dropout": "eval disabled", "cache": "none"}
    if parameter_scopes:
        identity["streamed_parameters"] = parameter_scopes.report()
        identity["streamed_helper_sha256"] = prepare.digest(ROOT / "tools/reference/qwen_streamed.py")
        parameter_scopes.close()
    for asset in [sources["model"]["weights"], *sources["assets"]]:
        if asset is sources["model"]["weights"] or asset["role"] == "model_metadata_or_tokenizer":
            prepare.validate_source(asset)
    manifests = [json.loads(json.dumps(identity)), json.loads(json.dumps(identity))]
    for directory in [full_directory, probes_directory]:
        directory.mkdir(parents=True)
    for name, (values, _, envelope) in sorted(arrays.items()):
        flattened = values.reshape(-1)
        # Deterministic coverage through the whole array, plus every-position
        # top logit. Full arrays stay local; small selected values are versioned.
        indices = set(np.linspace(0, flattened.size - 1, min(64, flattened.size), dtype=np.int64).tolist())
        if name.endswith(".logits"):
            width = values.shape[-1]
            for position, row in enumerate(values):
                indices.add(position * width + int(np.argmax(row)))
        indices = sorted(indices)
        for directory, manifest, array in [(full_directory, manifests[0], values), (probes_directory, manifests[1], flattened[indices])]:
            data = array.astype("<f4", copy=False).tobytes()
            filename = f"{name}.f32le"
            (directory / filename).write_bytes(data)
            metadata = {"file": filename, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "shape": list(values.shape), "reference_envelope": envelope}
            if directory == probes_directory:
                metadata["indices"] = indices
                metadata["full_array_sha256"] = hashlib.sha256(values.astype("<f4", copy=False).tobytes()).hexdigest()
                metadata["full_array_bytes"] = int(values.nbytes)
            else:
                metadata["elements"] = int(flattened.size)
            manifest["arrays"][name] = metadata
        if name.endswith(".logits"):
            case = name.split(".")[1]
            for manifest in manifests:
                manifest.setdefault("argmax", {})[case] = np.argmax(values, axis=-1).tolist()
    for directory, manifest in zip([full_directory, probes_directory], manifests):
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    check(full_directory, sources, layers)
    check(probes_directory, sources, layers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--probes-output", type=Path)
    parser.add_argument("--check", type=Path)
    parser.add_argument("--streamed", action="store_true")
    parser.add_argument("--larger", action="store_true")
    args = parser.parse_args()
    sources = prepare.load_json(ROOT / "benchmarks/m5-larger-sources-v1.json") if args.larger else None
    layers = [0, 1, 27] if args.larger else None
    if args.larger and not args.streamed:
        parser.error("larger export requires bounded streamed parameters")
    if args.check:
        check(args.check, sources, layers)
    elif args.output and args.probes_output:
        export(args.output, args.probes_output, sources,
               ROOT / "models/Qwen2.5-1.5B-Instruct" if args.larger else None,
               args.streamed, layers)
    else:
        parser.error("provide --check or both --output and --probes-output")


if __name__ == "__main__":
    main()
