"""Score one complete LFM3 task pair with the qualified isolated grader."""
import argparse
import hashlib
import json
from pathlib import Path

import gemma_task_results
from grading_support import require, verify_file
from lfm3_quality_inputs import INPUTS_SHA256, digest, read_bound, task_inputs
import paired_task_grading
import task_cases

ROOT = Path(__file__).resolve().parents[2]
PAIRS = [("stock", "modified_parent"), ("modified_parent", "native")]


def admit(folder, expected, role, cases, bind):
    receipt_path = folder / "result.json"
    receipt = json.loads(read_bound(receipt_path, expected))
    require(receipt["passed"] and receipt["role"] == role
            and receipt["first"] == 0 and receipt["end"] == receipt["tasks"] == 100
            and receipt["runtime_files"] == 28982 and receipt["input_reconciliation_sha256"] == INPUTS_SHA256,
            "Complete independently decoded role required")
    guard_path = folder / "decode-guard.json"
    guard = json.loads(read_bound(guard_path, receipt["guard_sha256"]))
    fixture_path = folder / "decode/fixture.json"
    fixture = json.loads(read_bound(fixture_path, guard["fixture_sha256"]))
    require(guard["passed"] and guard["exit_code"] == 0
            and sum(guard["final_peak_rss_bytes"]) <= 512 * 1024**2
            and guard["max_sampled_physical_bytes"] <= 512 * 1024**2
            and guard["minimum_free_bytes"] >= 2 * 1024**3
            and all(fixture[k] == receipt[k] for k in fixture), "Decode guard differs")
    path = folder / "decode/decoded.jsonl"
    responses = [json.loads(line) for line in read_bound(path, receipt["decoded_sha256"], 8 * 1024**2).splitlines()]
    require(len(responses) == len(cases) == 100, "All 100 literal completions are required")
    for case, response in zip(cases, responses, strict=True):
        text = response["visible_text"]
        require(response["id"] == case["id"] and response["category"] == case["category"]
                and isinstance(text, str) and len(text.encode("utf-8")) <= 65536
                and hashlib.sha256(text.encode("utf-8")).hexdigest() == response["visible_text_sha256"],
                "Literal completion differs")
    for p, h in receipt["pins"].items():
        bind(Path(p), h)
    for p, h in [(receipt_path, expected), (guard_path, receipt["guard_sha256"]),
                 (fixture_path, guard["fixture_sha256"]), (path, receipt["decoded_sha256"])]:
        bind(p, h)
    identity = dict(receipt_sha256=expected, collection_sha256=receipt["collection_sha256"],
                    decoded_tasks_sha256=receipt["decoded_sha256"])
    return responses, receipt["console_text_differences"], identity


def run(args):
    pair = tuple(args.roles)
    require(pair in PAIRS, "Unsupported LFM3 quality comparison")
    output = args.output.resolve()
    require(output.is_relative_to(ROOT / "artifacts"), "Fresh owned artifact output required")
    _, original = task_inputs(args.inputs, "stock")
    canonical = gemma_task_results.canonical_grading_cases(original)
    bindings = {}

    def bind(path, expected=None):
        path = path.resolve()
        require(path.is_relative_to(ROOT), "Grading evidence escapes the project")
        key = str(path.relative_to(ROOT))
        fingerprint = dict(bytes=path.stat().st_size, sha256=expected or digest(path))
        verify_file(path, fingerprint)
        require(key not in bindings or bindings[key] == fingerprint, "Conflicting evidence identity")
        bindings[key] = fingerprint

    def guard():
        for name, pin in bindings.items():
            verify_file(ROOT / name, pin)

    for path in [Path(__file__), Path(gemma_task_results.__file__), Path(task_cases.__file__),
                 Path(paired_task_grading.__file__), args.inputs / "cases.jsonl",
                 args.inputs / "reconciliation.json"]:
        bind(path)
    baseline, baseline_differences, baseline_identity = admit(
        args.baseline, args.baseline_sha256, pair[0], canonical, bind)
    candidate, candidate_differences, candidate_identity = admit(
        args.candidate, args.candidate_sha256, pair[1], canonical, bind)
    output.mkdir()
    record = dict(format="toklet-lfm3-literal-task-grading-v1", pair=pair, commands=[], runs=[],
                  outcomes={}, unresolved_case_ids=[], completed=False, generated_code_runs_on_host=False,
                  input_reconciliation_sha256=INPUTS_SHA256,
                  decoded_inputs={"baseline": baseline_identity, "candidate": candidate_identity},
                  console_text_differences={"baseline": baseline_differences, "candidate": candidate_differences})

    def decision(outcomes):
        return gemma_task_results.decision(canonical, outcomes)

    return paired_task_grading.grade_paired_cases(canonical, {"baseline": baseline, "native": candidate},
                                                guard, output, record, bindings, decision)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["inputs", "baseline", "candidate", "output"]:
        parser.add_argument("--" + name, required=True, type=Path)
    for name in ["baseline-sha256", "candidate-sha256"]:
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--roles", nargs=2, required=True)
    result = run(parser.parse_args())
    print(json.dumps({k: result.get(k) for k in ["completed", "pair", "decision", "failure"]}))
    raise SystemExit(0 if result["completed"] else 1)
