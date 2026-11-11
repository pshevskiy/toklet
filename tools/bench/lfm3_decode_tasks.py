"""Authenticate and decode finished task collections before literal grading."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

from lfm3_quality_inputs import EOS, INPUTS_SHA256, VOCAB, digest, read_bound, require, task_inputs
from lfm3_task_evidence import audit_native, audit_source

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference"))
from lfm3_source_tasks import RUNTIME_SHA, SOURCE_SHA, guard_phase, verify_runtime

CHECKPOINTS = {
    "stock": "91983df3e2f1ae164cb714daff3c5094cc17c8997a876f3e97b52711149bda70",
    "modified_parent": "b6ff17ac993dddf0f6f51bd872b12ae1437f4453a54a0b4878cedc7864fbc07e",
}


def dump(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"


def decode(args):
    tasks, _ = task_inputs(args.inputs, args.role)
    receipt_path = args.collection / "result.json"
    receipt = json.loads(read_bound(receipt_path, args.collection_sha256))
    first, end = receipt["first"], receipt["end"]
    require(receipt["passed"] and 0 <= first < end <= 100, "Completed guarded task collection required")
    records = [json.loads(line) for line in read_bound(
        args.collection / "tasks.jsonl", receipt["tasks_sha256"], 8 * 1024**2).splitlines()]
    require(len(records) == end - first, "Task collection coverage differs")
    if args.role == "native":
        require(receipt["format"] == "toklet-lfm3-native-task-collection-v1" and receipt["exit_code"] == 0
                and receipt["completed"] == end - first and sum(receipt["peak_rss_bytes"]) <= 6 * 1024**3
                and receipt["input_reconciliation_sha256"] == INPUTS_SHA256,
                "Native collection admission differs")
        audit = audit_native(args.collection, tasks, first, records, receipt)
    else:
        require(receipt["role"] == args.role and receipt["cases"] == end - first
                and receipt["source_checkpoint_sha256"] == CHECKPOINTS[args.role]
                and receipt["runtime_verified_files"] == 28982
                and digest(args.collection / "freeze.json") == receipt["freeze_sha256"],
                "Independent source collection admission differs")
        audit = audit_source(args.collection, tasks, records, receipt)
    source = json.loads(read_bound(args.source / "prospective-contract-1.json", SOURCE_SHA))
    asset = "stock_mlx-tokenizer.json" if args.role == "stock" else "child-tokenizer.json"
    tok_path = args.source / "metadata-1" / asset
    expected = source["metadata_sha256"][asset]
    raw = read_bound(tok_path, expected, 64 * 1024**2)
    sys.path.insert(0, str(args.tokenizers_site))
    import tokenizers
    require(tokenizers.__version__ == "0.23.1", "Pinned Tokenizers runtime required")
    tokenizer = tokenizers.Tokenizer.from_buffer(raw)
    require(tokenizer.token_to_id("<|im_end|>") == EOS, "Tokenizer stop policy differs")
    args.output.mkdir()
    text_differences = []
    data = args.output / "decoded.jsonl"
    with data.open("x") as stream:
        for index, row in enumerate(records, first):
            task = tasks[index]
            prompt = len(task["prompt_tokens"])
            cap = task["max_output_tokens"]
            visible = row["visible_token_ids"]
            require(row["index"] == index and row["id"] == task["id"] and row["category"] == task["category"]
                    and row["prompt_tokens"] == prompt and row["output_reserve"] == cap,
                    "Task identity, prompt or output reserve differs")
            require(type(visible) is list and len(visible) <= cap
                    and all(type(t) is int and 0 <= t < VOCAB and t != EOS for t in visible),
                    "Invalid visible token sequence")
            generation = row["generation"] if args.role == "native" else row
            reason = generation["reason"]
            stop = reason == {"kind": "stop_token", "token_id": EOS}
            require((stop and len(visible) < cap) or
                    (reason == {"kind": "output_limit"} and len(visible) == cap),
                    "Incomplete or invalid task stopping boundary")
            sampled = len(visible) + int(stop)
            require(generation["history_tokens"] == prompt + sampled and generation["pending_tokens"] == 1,
                    "Task history does not match its sampled IDs")
            if args.role == "native":
                counts = dict(reused_tokens=0, prefill_tokens=prompt, sampled_tokens=sampled,
                              emitted_tokens=len(visible), evaluated_tokens=prompt + sampled - 1)
                require(all(generation[k] == v for k, v in counts.items()), "Native state differs")
            else:
                require(generation["committed_tokens"] == prompt + sampled - 1, "Source prefix differs")
                samples = row["samples"]
                require([r["token_id"] for r in samples] == visible + ([EOS] if stop else [])
                        and all(r["index"] == index and len(bytes.fromhex(r["logit_sha256"])) == 32
                                and math.isfinite(r["max_logit"]) for r in samples),
                        "Source sample decisions differ from the final trajectory")
            text = tokenizer.decode(visible, skip_special_tokens=True)
            encoded = text.encode("utf-8")
            require(len(encoded) <= 65536, "Task completion exceeds the literal grading bound")
            text_sha = hashlib.sha256(encoded).hexdigest()
            if args.role == "native":
                published = row["visible_text"].encode("utf-8")
                require(hashlib.sha256(published).hexdigest() == row["visible_text_sha256"],
                        "Console visible text differs from its receipt")
                if published != encoded:
                    text_differences.append(task["id"])
            stream.write(dump(dict(id=task["id"], category=task["category"], visible_token_ids=visible,
                                   visible_text=text, visible_text_sha256=text_sha)))
    require(digest(tok_path) == expected and digest(receipt_path) == args.collection_sha256
            and digest(args.collection / "tasks.jsonl") == receipt["tasks_sha256"], "Decode input changed")
    with (args.output / "fixture.json").open("x") as stream:
        stream.write(dump(dict(passed=True, role=args.role, first=first, end=end, tasks=end-first,
                               collection_sha256=args.collection_sha256, tokenizer_sha256=expected,
                               decoded_sha256=digest(data), console_text_differences=text_differences,
                               collection_audit=audit,
                               input_reconciliation_sha256=INPUTS_SHA256)))


def run(args):
    receipt = args.collection / "result.json"
    require(digest(receipt) == args.collection_sha256, "Collection receipt changed")
    args.output.mkdir()
    runtime_count = verify_runtime(args.runtime)
    own = Path(__file__).resolve()
    paths = [own, args.driver, receipt, args.collection / "tasks.jsonl",
             args.inputs / "reconciliation.json", args.source / "prospective-contract-1.json",
             own.with_name("lfm3_quality_inputs.py"), own.with_name("lfm3_task_evidence.py"),
             own.with_name("lfm3_quality_tasks.py"), own.parents[1] / "reference/lfm3_source_tasks.py"]
    pins = {str(path): digest(path) for path in paths}
    output = args.output / "decode"
    command = [str(args.python), "-B", str(args.driver), str(own), "--worker", "--role", args.role,
               "--collection-sha256", args.collection_sha256]
    for name in ["collection", "inputs", "source", "tokenizers_site", "python", "driver", "runtime"]:
        command += ["--" + name.replace("_", "-"), str(getattr(args, name))]
    command += ["--output", str(output)]
    result = guard_phase(command, "decode", args.output, output / "fixture.json")
    require(verify_runtime(args.runtime) == runtime_count and all(digest(Path(p)) == h for p, h in pins.items()),
            "Decode runtime or input evidence changed")
    result.update(runtime_sha256=RUNTIME_SHA, runtime_files=runtime_count, pins=pins,
                  guard_sha256=digest(args.output / "decode-guard.json"))
    with (args.output / "result.json").open("x") as stream:
        stream.write(dump(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["collection", "inputs", "source", "output", "tokenizers-site", "driver", "runtime"]:
        parser.add_argument("--" + name, type=lambda p: Path(p).resolve(), required=True)
    parser.add_argument("--python", type=lambda p: Path(p).absolute(), required=True)
    parser.add_argument("--role", choices=["stock", "modified_parent", "native"], required=True)
    parser.add_argument("--collection-sha256", required=True)
    parser.add_argument("--worker", action="store_true")
    options = parser.parse_args()
    decode(options) if options.worker else run(options)
