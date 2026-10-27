"""Freeze independent experimental C512 inputs; derive no gates or target data."""
import hashlib
import json
from pathlib import Path
import re
import struct

import qwen3_long_context as reference

FORMAT = "toklet-qwen3-context-input-recipe-v2"
NATIVE_STAGES = ("input_norm", "q_projection", "k_projection", "v_projection", "q_norm", "k_norm",
    "q_rotary", "k_rotary", "attention_context", "attention_projection", "attention_residual",
    "post_attention_norm", "gate_projection", "up_projection", "gate_silu", "gated", "down_projection", "output")
COUNTS = {"calibration": [32, 32, 128, 128, 512, 512],
          "validation": [127, 128, 129, 255, 256, 257, 511, 512]}
REFERENCE_PROFILE = dict(weights="original affine checkpoint reconstructed by FP64 multiply/add then FP32 cast",
    activations_and_kv="Half", logits="FP32", arithmetic=["FP32", "FP64"],
    graph="unchanged independent qwen3_fp16.forward", layers=[0, 1, 27])
NATIVE_PROFILE = dict(projection="MlxAffine", normalization="Compensated", attention="Reference", readback="FullLogits")
REJECTIONS = ["nonfinite-or-missing-value", "incomplete-case-or-state-coverage", "source-or-artifact-identity-drift",
    "any-native-scalar-outside-frozen-canonical-FP32-gate", "CPU-validation-FP32-versus-FP64-outside-frozen-envelope",
    "any-causal-mask-or-state-contract-failure", "any-native-canonical-FP32-argmax-mismatch-including-near-ties",
    "any-retuning-after-validation", "unreviewed-or-unacceptable-calibration-envelope"]
QUALITY = dict(targets=10000, tasks=100, required_capacity_at_least=1025,
    loss_increase_limit_nats_per_token=0.05, task_drop_limit_percentage_points=2,
    clipping_allowed=False, same_quantized_source_parity_is_unquantized_baseline=False)


def identity(body):
    return dict(bytes=len(body), sha256=hashlib.sha256(body).hexdigest())


def validate(recipe):
    if (recipe.get("format") != FORMAT or recipe.get("status") != "experimental-input-freeze-only"
            or type(recipe.get("context")) is not int or recipe["context"] != 512
            or type(recipe.get("vocabulary")) is not int or recipe["vocabulary"] != 151936
            or type(recipe.get("text_repetitions")) is not int or recipe["text_repetitions"] != 16):
        raise ValueError("Input recipe geometry differs")
    sources, cases = recipe["sources"], recipe["cases"]
    if (not isinstance(sources, dict) or not 1 <= len(sources) <= 32 or not isinstance(cases, list)
            or len(cases) != 14 or any(not isinstance(v, str) or not v.strip() or len(v.encode()) > 8192 for v in sources.values())):
        raise ValueError("Bounded authored source/case inventory required")
    ids, selected, source_splits = set(), {key: [] for key in COUNTS}, {}
    for case in cases:
        if (set(case) != {"id", "split", "source", "tokens"} or not isinstance(case["id"], str)
                or not re.fullmatch(r"[a-z0-9-]{1,64}", case["id"]) or case["id"] in ids
                or case["split"] not in COUNTS or case["source"] not in sources
                or type(case["tokens"]) is not int or not 1 <= case["tokens"] <= 512):
            raise ValueError("Input case identity/shape differs")
        ids.add(case["id"]);selected[case["split"]].append(case["tokens"])
        # Compare content, not source labels, so aliases cannot bridge splits.
        digest = identity(sources[case["source"]].strip().encode())["sha256"]
        if digest in source_splits and source_splits[digest] != case["split"]:
            raise ValueError("Authored source crosses calibration/validation split")
        source_splits[digest] = case["split"]
    if any(sorted(selected[key]) != counts for key, counts in COUNTS.items()):
        raise ValueError("Boundary coverage differs")
    proposal = recipe["proposal"]
    if (proposal.get("adopted") is not False or proposal.get("capacity_or_precision_changed") is not False
            or proposal.get("gate_status") != "not-derived-not-adopted"
            or proposal.get("quality", {}).get("clipping_allowed") is not False
            or proposal["quality"].get("required_capacity_at_least") != 1025
            or proposal["quality"].get("targets") != 10000 or proposal["quality"].get("tasks") != 100):
        raise ValueError("Experimental scope/complete quality demand differs")
    fixed = dict(id="toklet-qwen3-context512-experimental-v2", atol_floor=0.0001, envelope_multiplier=2, rtol=0.001,
        pooling="exact captured state name across calibration cases only; no layer or dtype pooling",
        formula="atol=max(atol_floor,2*calibration_FP32_FP64_max_abs,2*calibration_equivalent_prefix_max_abs)",
        canonical_target="independent CPU FP32 saved values; FP64 is a separately reported arithmetic diagnostic, not a second acceptance oracle",
        cpu_cross_precision_validation="abs(FP32-FP64)<=atol+rtol*abs(FP64) using the frozen calibration envelope; failure rejects calibration coverage, not a second native target oracle; report CPU argmax disagreements separately",
        near_ties="retain full row, top-two IDs and margin; any native versus canonical CPU-FP32 argmax mismatch rejects, including near ties",
        envelope_ceiling="pending independent calibration review; no universal or native-error-derived ceiling",
        freeze_order=["inputs-and-formula", "CPU-calibration-only", "review-and-immutable-gates",
                      "untouched-CPU-and-native-validation", "source-and-full-quality-before-adoption"],
        rejection_rules=REJECTIONS, quality=QUALITY)
    if (any(proposal.get(key) != value for key, value in fixed.items())
            or any(type(proposal.get(key)) not in (int, float) for key in ("atol_floor", "envelope_multiplier", "rtol"))
            or proposal["quality"].get("same_quantized_source_parity_is_unquantized_baseline") is not False
            or recipe.get("reference_profile") != REFERENCE_PROFILE or recipe.get("native_profile") != NATIVE_PROFILE
            or any(type(layer) is not int for layer in recipe["reference_profile"]["layers"])):
        raise ValueError("Frozen numerical/quality/profile semantics differ")


def build(recipe, encode):
    """Tokenization is injected; this module executes no model or numerical graph."""
    validate(recipe)
    encoded, rows = {}, []
    for case in recipe["cases"]:
        source = case["source"]
        if source not in encoded:
            text = (recipe["sources"][source] + "\n\n") * recipe["text_repetitions"]
            tokens = encode(text)
            if (not isinstance(tokens, list) or len(tokens) > 131072
                    or any(type(t) is not int or not 0 <= t < recipe["vocabulary"] for t in tokens)):
                raise ValueError("Tokenizer returned invalid/unbounded IDs")
            encoded[source] = (identity(text.encode()), tokens)
        text_identity, tokens = encoded[source]
        count = case["tokens"]
        if len(tokens) < count:
            raise ValueError("Authored text has insufficient tokens; implicit padding forbidden")
        selected = tokens[:count]
        body = struct.pack("<" + "I" * count, *selected)
        rows.append(dict(**case, source_text=text_identity, encoded_source_tokens=len(tokens), ids=selected,
                         token_array=identity(body), offset=0))
    for index, left in enumerate(rows):
        for right in rows[index + 1:]:
            if left["ids"] == right["ids"]:
                raise ValueError("Duplicate numerical input sequence")
            prefix = min(len(left["ids"]), len(right["ids"]))
            if left["split"] != right["split"] and left["ids"][:prefix] == right["ids"][:prefix]:
                raise ValueError("Cross-split token-prefix reuse")
    return rows


def write_inputs(destination, recipe, encode, profile):
    """Never replace a prior run; retain partial files and manifest on failure."""
    destination = Path(destination)
    if destination.exists():
        raise ValueError("Input destination already exists")
    rows = build(recipe, encode)
    destination.mkdir()
    names = ["embedding", "final_norm", "logits"] + [f"layers.{layer}.{stage}" for layer in (0, 1, 27) for stage in NATIVE_STAGES]
    result = dict(format="toklet-qwen3-context-inputs-v2", complete=False, primary_error=None,
        recipe=recipe, recipe_identity=identity((json.dumps(recipe, sort_keys=True, separators=(",", ":")) + "\n").encode()),
        profile=profile, rows=[], stage_pooling={name: name for name in names}, calibration_prefix_pairs=[],
        token_selection="encode((source + two newlines) * explicit 16 repetitions), then take the declared token prefix; numerical witness only",
        numerical_gates_derived=False, numerical_qualified=False, quality_qualified=False, adopted=False)
    primary = None
    try:
        for row in rows:
            body = struct.pack("<" + "I" * row["tokens"], *row["ids"])
            filename = row["id"] + ".u32le"
            with (destination / filename).open("xb") as file:
                if file.write(body) != len(body):
                    raise OSError("Incomplete token write")
            result["rows"].append({key: value for key, value in row.items() if key != "ids"} | dict(file=filename, dtype="u32le"))
        for left in rows:
            for right in rows:
                if (left["split"] == right["split"] == "calibration" and left["source"] == right["source"]
                        and left["tokens"] < right["tokens"]):
                    result["calibration_prefix_pairs"].append(dict(short=left["id"], long=right["id"],
                        positions=left["tokens"], axes="token axis 0 for every listed captured state"))
        result["complete"] = True
    except BaseException as error:
        primary = error
        result["primary_error"] = type(error).__name__ + ": " + str(error)
        raise
    finally:
        try:
            with (destination / "manifest.json").open("x") as file:
                json.dump(result, file, indent=2, allow_nan=False);file.write("\n")
        except BaseException as error:
            if primary is None:
                raise
            primary.add_note("Partial input manifest: " + type(error).__name__ + ": " + str(error))
    return result


def prepare(destination, root, recipe_path, recipe_sha):
    """Explicit offline tokenizer boundary; runtime/source admission is external."""
    from importlib.metadata import version
    from tokenizers import Tokenizer
    if version("tokenizers") != "0.22.2":
        raise ValueError("Pinned tokenizer runtime required")
    root = Path(root)
    source_manifest = root / "tests/fixtures/qwen3-context512-v1/manifest.json"
    frozen = reference.load(source_manifest, "21d54e567025f70fdda01d7f1bede1a71f730c6b7c10aec62f5aaffec52a7f08")
    recipe = reference.load(recipe_path, recipe_sha)
    folder = root / "models/MLX-Qwen3-0.6B-8bit-11de968"
    path = folder / "tokenizer.json"
    with path.open("rb") as file:
        body = file.read(16 * 1024**2 + 1)
    if len(body) > 16 * 1024**2:
        raise ValueError("Tokenizer exceeds bounded extent")
    if identity(body)["sha256"] != frozen["model_pins"]["tokenizer.json"]:
        raise ValueError("Original tokenizer differs")
    # Parse the authenticated buffer rather than reopening a potentially changed path.
    tokenizer = Tokenizer.from_str(body.decode())
    quality = root / "benchmarks/contract-v1/manifest.json"
    profile = dict(model_repo="mlx-community/Qwen3-0.6B-8bit", model_revision="11de96878523501bcaa86104e3c186de07ff9068",
        required_model_files=frozen["model_pins"], tokenizer_runtime="0.22.2", tokenizer=identity(body),
        add_special_tokens=False, chat_template_evaluated=False,
        quality_dependency_manifest=identity(quality.read_bytes()), quality_execution="pending original complete demand; no C512 substitute",
        quality_checkpoint_profile="unprepared; requires original source retokenization and complete frozen loss/task demand for this checkpoint",
        reference_profile=recipe["reference_profile"], native_profile=recipe["native_profile"])
    return write_inputs(destination, recipe, lambda text: tokenizer.encode(text, add_special_tokens=False).ids, profile)
