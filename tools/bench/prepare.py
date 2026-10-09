"""Fetch pinned sources explicitly, or freeze/verify offline benchmark inputs."""

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import struct
import subprocess


ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "benchmarks"
SOURCES = BENCH / "sources.json"
CONTRACT = BENCH / "contract-v1"
QUALITY = ROOT / "artifacts/quality-v1/tasks.jsonl"


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def load_json(path):
    return json.loads(path.read_text())


def validate_source(asset):
    path = ROOT / asset.get("path", asset.get("file", ""))
    if path.stat().st_size != asset["bytes"] or digest(path) != asset["sha256"]:
        raise ValueError(f"Source identity mismatch: {path}")


def fetch(asset):
    path = ROOT / asset.get("path", asset.get("file", ""))
    if path.exists():
        validate_source(asset)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    try:
        subprocess.run([
            "curl", "--fail", "--silent", "--show-error", "--location",
            "--retry", "2", "--connect-timeout", "15", "--max-time", "600",
            "--output", str(temporary), asset["url"],
        ], check=True)
        if temporary.stat().st_size != asset["bytes"] or digest(temporary) != asset["sha256"]:
            raise ValueError(f"Downloaded source identity mismatch: {asset['url']}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def read_tokens(path):
    data = path.read_bytes()
    if len(data) % 4:
        raise ValueError(f"Truncated u32 token file: {path}")
    return list(struct.unpack(f"<{len(data) // 4}I", data))


def token_file(directory, filename, tokens):
    path = directory / filename
    path.write_bytes(struct.pack(f"<{len(tokens)}I", *tokens))
    return {"file": filename, "dtype": "u32le", "count": len(tokens), "sha256": digest(path)}


def build(directory):
    # Imports belong only to explicit preparation, never the offline verifier.
    import pyarrow.parquet as parquet
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer
    from importlib.metadata import version

    if directory.exists():
        raise ValueError(f"Refusing to overwrite frozen contract: {directory}")
    sources = load_json(SOURCES)
    for package in ("tokenizers", "transformers"):
        if version(package) != sources["baselines"][package]:
            raise ValueError(f"Preparation requires pinned {package}")
    for asset in sources["assets"]:
        validate_source(asset)
    tokenizer = Tokenizer.from_file(str(ROOT / "models/Qwen2.5-0.5B-Instruct/tokenizer.json"))
    chat = AutoTokenizer.from_pretrained(
        str(ROOT / "models/Qwen2.5-0.5B-Instruct"),
        local_files_only=True, trust_remote_code=False,
    )

    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False).ids

    directory.mkdir(parents=True)
    authored = "\n".join(
        f"Review item {i}: inspect the Rust decoder, validate dimensions before writing, "
        "keep immutable weights separate from session scratch, and compare each output "
        "with an independent numerical fixture. The tool reports a clean diff and "
        "asks the engine to continue from the exact token prefix. Record completed "
        "prefill and decode work, preserve causal attention, and reset the cache "
        "between independent requests. Explain the result in clear sentences."
        for i in range(64)
    )
    continuation = "\n".join(
        f"Result {i}: the checked operation agrees with the saved oracle. "
        "The next task starts after review and a separate commit."
        for i in range(16)
    )
    (directory / "prompt.txt").write_text(authored + "\n")
    (directory / "continuation.txt").write_text(continuation + "\n")
    prompt_tokens, output_tokens = encode(authored), encode(continuation)[:128]
    if len(prompt_tokens) < 1921 or len(output_tokens) != 128:
        raise ValueError("Authored workload text is too short")
    arrays = {}
    for count in (128, 512, 1024, 1920, 1921):
        arrays[f"prompt_{count}"] = token_file(directory, f"prompt_{count}.u32le", prompt_tokens[:count])
    arrays["continuation_128"] = token_file(directory, "continuation_128.u32le", output_tokens)
    arrays["append_64"] = token_file(directory, "append_64.u32le", encode("Tool result: all checks passed.\n" * 32)[:64])
    changed = prompt_tokens[:512].copy()
    replacement = encode("revised")[0]
    changed[128] = replacement if replacement != changed[128] else encode("changed")[0]
    if changed[128] == prompt_tokens[128]:
        raise ValueError("Changed-prefix workload must change a token")
    arrays["changed_prompt_512"] = token_file(directory, "changed_prompt_512.u32le", changed)
    rows = parquet.read_table(ROOT / "artifacts/datasets/wikitext-2-raw-test.parquet", columns=["text"]).column("text").to_pylist()
    loss_tokens = encode("\n".join(text for text in rows if text))[:10001]
    if len(loss_tokens) != 10001:
        raise ValueError("Held-out corpus has fewer than 10,000 scored targets")
    arrays["held_out_loss"] = token_file(directory, "held_out_loss.u32le", loss_tokens)

    tasks = []
    with gzip.open(ROOT / "artifacts/datasets/HumanEval.jsonl.gz", "rt") as source:
        code_cases = [json.loads(line) for line in source]
    for case in code_cases[:40]:
        prompt = chat.apply_chat_template([{"role": "user", "content": "Complete this Python function. Return only the indented completion, without repeating the function signature or using Markdown fences.\n\n" + case["prompt"]}], tokenize=False, add_generation_prompt=True)
        tasks.append({"id": case["task_id"], "category": "code", "prompt_text": prompt, "prompt_tokens": encode(prompt), "max_output_tokens": 256, "stop_policy": "eos_or_256", "scoring": "append_completion_and_run_source_tests", "code_prefix": case["prompt"], "tests": case["test"], "entry_point": case["entry_point"]})
    with (ROOT / "artifacts/datasets/TruthfulQA.csv").open(newline="") as source:
        prose_cases = list(csv.DictReader(source))
    for index, case in enumerate(prose_cases[:30]):
        options = [case["Best Answer"]]
        for incorrect in case["Incorrect Answers"].split(";"):
            incorrect = incorrect.strip()
            if incorrect and incorrect not in options:
                options.append(incorrect)
            if len(options) == 4:
                break
        if len(options) < 2:
            raise ValueError(f"TruthfulQA row {index} has no distractor")
        rotation = index % len(options)
        options = options[rotation:] + options[:rotation]
        correct = chr(ord("A") + (-rotation) % len(options))
        question = "Choose the true answer. Reply with exactly one uppercase letter.\n" + case["Question"] + "\n" + "\n".join(f"{chr(ord('A') + i)}. {answer}" for i, answer in enumerate(options))
        prompt = chat.apply_chat_template([{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True)
        tasks.append({"id": f"truthfulqa/row-{index}", "category": "prose", "prompt_text": prompt, "prompt_tokens": encode(prompt), "max_output_tokens": 32, "stop_policy": "eos_or_32", "scoring": "trimmed_text_equals_expected_letter", "expected": correct})
    with (ROOT / "artifacts/datasets/lambada_test.jsonl").open() as source:
        continuation_cases = [json.loads(line) for line in source]
    for index, case in enumerate(continuation_cases[:30]):
        prompt, separator, expected = case["text"].rstrip().rpartition(" ")
        if not separator or not expected:
            raise ValueError(f"Invalid LAMBADA row {index}")
        tasks.append({"id": f"lambada/row-{index}", "category": "continuation", "prompt_text": prompt, "prompt_tokens": encode(prompt), "max_output_tokens": 16, "stop_policy": "eos_or_16", "scoring": "first_whitespace_delimited_word_equals_expected", "expected": expected})
    if len(tasks) != 100 or any(len(task["prompt_tokens"]) + task["max_output_tokens"] > 2048 for task in tasks):
        raise ValueError("Quality tasks must number 100 and fit the admitted capacity")
    QUALITY.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(task, sort_keys=True) + "\n" for task in tasks)
    if QUALITY.exists() and QUALITY.read_text() != payload:
        raise ValueError("Refusing to overwrite changed frozen quality tasks")
    QUALITY.write_text(payload)
    trace = [
        {"role": "system", "content": "You are a Rust coding assistant. Describe changes and validation clearly."},
        {"role": "user", "content": "Validate this rectangular linear operation and explain its shape."},
        {"role": "assistant", "content": "I will compare the operation with the frozen NumPy fixture."},
        {"role": "user", "content": "Tool result: formatting, Clippy, and all CPU tests passed. Continue with normalization."},
        {"role": "assistant", "content": "The dimensions and fixture values agree. Normalization is the next task."},
        {"role": "user", "content": "Tool result: the review has no findings. Commit the task and report its checks."},
    ]
    write_json(directory / "application_trace.json", trace)
    manifest = {
        "schema_version": 1, "contract_id": "toklet-benchmark-contract-v1",
        "sources_sha256": digest(SOURCES), "arrays": arrays,
        "python_dependencies_sha256": digest(ROOT / "tools/bench/requirements.lock"),
        "authored_text_sha256": {name: digest(directory / name) for name in ("prompt.txt", "continuation.txt", "application_trace.json")},
        "tokenization": {"tokenizers": "0.22.2", "add_special_tokens": False, "throughput_uses_chat_template": False, "quality_code_and_prose_use_pinned_chat_template": True, "eos_token_id": 151645},
        "throughput": {"prompt_lengths": [128, 512, 1024, 1920], "output_steps": 128, "stop_policy": "fixed_length_teacher_forced_ignore_eos", "warmups": 3, "repetitions": 10, "context_capacity": 2048, "sustained_seconds": 600},
        "continuation": {"base_prompt": "prompt_512", "append": "append_64", "fully_cached_evaluated_len": 512, "pending_sample": {"history_len": 513, "evaluated_len": 512, "pending_token": output_tokens[0]}, "changed_prompt": "changed_prompt_512", "changed_first_index": 128, "m2_changed_prefix_policy": "reset_and_replay", "capacity_rejection": {"prompt": "prompt_1921", "output_steps": 128}},
        "quality": {"tasks_file": str(QUALITY.relative_to(ROOT)), "tasks_sha256": digest(QUALITY), "task_ids": [task["id"] for task in tasks], "categories": {"code": 40, "prose": 30, "continuation": 30}, "case_policy": "first_source_rows_in_order_no_model_based_selection", "scored_loss_tokens": 10000, "loss_block_targets": 1024, "loss_context_policy": "independent_blocks_of_up_to_1025_tokens_one_token_overlap_score_every_target_once", "mean_nll_increase_limit_nats_per_token": 0.05, "task_pass_rate_loss_limit_percentage_points": 2.0, "aggregation": "100_equal_weight_binary_cases", "sampling": "greedy_lowest_token_id_on_ties", "calibration": "held_out_sources_and_all_selected_cases_forbidden_for_quantizer_tuning"},
    }
    write_json(directory / "manifest.json", manifest)
    verify(directory, check_quality=True)


def verify(directory, check_quality=False):
    manifest = load_json(directory / "manifest.json")
    sources = load_json(SOURCES)
    if manifest["schema_version"] != 1 or manifest["sources_sha256"] != digest(SOURCES):
        raise ValueError("Contract source identity changed")
    if manifest["python_dependencies_sha256"] != digest(ROOT / "tools/bench/requirements.lock"):
        raise ValueError("Locked Python dependency identity changed")
    for name, array in manifest["arrays"].items():
        path = directory / array["file"]
        tokens = read_tokens(path)
        if array["dtype"] != "u32le" or len(tokens) != array["count"] or digest(path) != array["sha256"]:
            raise ValueError(f"Token array mismatch: {name}")
        if any(token >= 151936 for token in tokens):
            raise ValueError(f"Token outside pinned Qwen vocabulary: {name}")
    for name, sha in manifest["authored_text_sha256"].items():
        if digest(directory / name) != sha:
            raise ValueError(f"Authored input mismatch: {name}")
    for count in (128, 512, 1024, 1920, 1921):
        if manifest["arrays"][f"prompt_{count}"]["count"] != count:
            raise ValueError("Prompt length mismatch")
    quality = manifest["quality"]
    if quality["scored_loss_tokens"] != 10000 or manifest["arrays"]["held_out_loss"]["count"] != 10001:
        raise ValueError("Held-out loss count changed")
    if len(set(quality["task_ids"])) != 100 or quality["categories"] != {"code": 40, "prose": 30, "continuation": 30}:
        raise ValueError("Quality case selection changed")
    if quality["mean_nll_increase_limit_nats_per_token"] != 0.05 or quality["task_pass_rate_loss_limit_percentage_points"] != 2.0:
        raise ValueError("Frozen quality gates changed")
    base = read_tokens(directory / manifest["arrays"]["prompt_512"]["file"])
    changed = read_tokens(directory / manifest["arrays"]["changed_prompt_512"]["file"])
    if [i for i, (a, b) in enumerate(zip(base, changed)) if a != b] != [128]:
        raise ValueError("Changed-prefix case must edit exactly token 128")
    if sources["model"]["engine_total_token_limit"] != 2048:
        raise ValueError("Engine admission limit changed")
    if check_quality:
        path = ROOT / quality["tasks_file"]
        if digest(path) != quality["tasks_sha256"]:
            raise ValueError("Prepared quality tasks changed")
        tasks = [json.loads(line) for line in path.read_text().splitlines()]
        if [task["id"] for task in tasks] != quality["task_ids"]:
            raise ValueError("Prepared task IDs changed")
    print(f"Verified {manifest['contract_id']}: fixed throughput tokens, 10,000 loss targets, 100 task identities.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fetch", action="store_true", help="download pinned public tokenizer and dataset sources")
    mode.add_argument("--fetch-weights", action="store_true", help="explicitly download the pinned 988 MB model")
    mode.add_argument("--build", type=Path, metavar="NEW_DIRECTORY", help="freeze a new contract using local sources")
    parser.add_argument("--check-quality", action="store_true", help="also verify locally prepared quality payload")
    args = parser.parse_args()
    if args.fetch or args.fetch_weights:
        sources = load_json(SOURCES)
        assets = [sources["model"]["weights"]] if args.fetch_weights else sources["assets"]
        for asset in assets:
            fetch(asset)
    elif args.build:
        build(args.build)
    else:
        verify(CONTRACT, args.check_quality)


if __name__ == "__main__":
    main()
