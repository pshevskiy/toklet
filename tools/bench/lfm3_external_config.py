"""Prepare an authenticated, metadata-only LFM3 configuration for external engines.

No weights are copied, linked, imported or executed. The derived directory cannot
load a model. Its receipt identifies the original weights for a later guarded run.
"""
import argparse
import hashlib
import json
from pathlib import Path


CONTRACT_SHA256 = "d9a5b636fcb3e7eb472f3cd1e58848bbaa4703a6acfca4dc4a398eb963751af2"
MAX_METADATA_BYTES = 32 * 1024**2
ASSETS = {
    "config.json": "child-config.json",
    "generation_config.json": "child-generation_config.json",
    "tokenizer.json": "child-tokenizer.json",
    "tokenizer_config.json": "child-tokenizer_config.json",
    "chat_template.jinja": "child-chat_template.jinja",
    "lfm3.py": "child-lfm3.py",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def bound_bytes(path, expected):
    require(path.is_file() and path.stat().st_size <= MAX_METADATA_BYTES,
            f"Missing or oversized metadata: {path}")
    with path.open("rb") as stream:
        data = stream.read(MAX_METADATA_BYTES + 1)
    require(len(data) <= MAX_METADATA_BYTES, f"Metadata grew beyond its read bound: {path}")
    require(hashlib.sha256(data).hexdigest() == expected, f"Metadata changed: {path}")
    return data


def prepare(model, configuration, contract, output, provider):
    require(provider in ("mlx-lm", "mlx-vlm"), "Unknown provider")
    model, configuration, contract, output = (
        path.resolve() for path in (model, configuration, contract, output)
    )
    require(not output.exists() and not output.is_relative_to(model),
            "Output must be a new directory outside the checkpoint")
    spec = json.loads(bound_bytes(contract, CONTRACT_SHA256))
    pins = spec["metadata_sha256"]
    original = {name: bound_bytes(model / name, pins[key]) for name, key in ASSETS.items()}
    config_source = bound_bytes(configuration, pins["parent-configuration_lfm3.py"])
    config = json.loads(original["config.json"])
    require(config.get("model_type") == "lfm3"
            and config.get("architectures") == ["Lfm3ForCausalLM"]
            and config.get("dtype") == "bfloat16"
            and config.get("model_file") == "lfm3.py" and "auto_map" not in config,
            "Unexpected original configuration")
    derived = dict(config)
    derived["auto_map"] = {"AutoConfig": "configuration_lfm3.Lfm3Config"}
    if provider == "mlx-vlm":
        del derived["model_file"]
    changed = sorted(key for key in config.keys() | derived.keys()
                     if config.get(key) != derived.get(key))
    require(changed == (["auto_map", "model_file"] if provider == "mlx-vlm" else ["auto_map"]),
            "Unexpected configuration changes")
    weights = model / "model.safetensors"
    require(weights.is_file() and weights.stat().st_size == spec["candidate"]["bytes"]
            and digest(weights) == spec["candidate"]["sha256"], "Checkpoint changed")
    # All admission checks precede the first write. Only captured, checked metadata
    # bytes enter this directory; no hard link can mutate the source checkpoint.
    output.mkdir()
    written = {}
    for name, data in original.items():
        if name == "config.json":
            data = (json.dumps(derived, indent=2, allow_nan=False) + "\n").encode()
        if name == "lfm3.py" and provider == "mlx-vlm":
            continue
        (output / name).write_bytes(data)
        written[name] = hashlib.sha256(data).hexdigest()
    (output / "configuration_lfm3.py").write_bytes(config_source)
    written["configuration_lfm3.py"] = hashlib.sha256(config_source).hexdigest()
    require(digest(weights) == spec["candidate"]["sha256"]
            and all(bound_bytes(model / name, pins[key]) == original[name]
                    for name, key in ASSETS.items())
            and bound_bytes(configuration, pins["parent-configuration_lfm3.py"]) == config_source
            and digest(contract) == CONTRACT_SHA256, "Source changed during preparation")
    receipt = dict(
        format="toklet-lfm3-external-config-v1", provider=provider,
        contract_sha256=CONTRACT_SHA256, candidate=spec["candidate"],
        original_checkpoint=str(model), original_config_sha256=pins["child-config.json"],
        config_changes={key: {"before": config.get(key), "after": derived.get(key)} for key in changed},
        files_sha256=written, metadata_only=True, model_load_ready=False,
        source_unchanged=True, weights_loaded=False,
        execution=("publisher custom MLX-LM module" if provider == "mlx-lm"
                   else "installed MLX-VLM built-in LFM3; numerical equivalence unqualified"),
    )
    with (output / "receipt.json").open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "configuration", "contract", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--provider", choices=("mlx-lm", "mlx-vlm"), required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.model, args.configuration, args.contract, args.output, args.provider)))


if __name__ == "__main__":
    main()
