"""Decide both original LFM3 quality comparisons from complete guarded reports."""
import argparse
import json
import math
from pathlib import Path

from lfm3_loss_report import LIMIT, TARGETS
from lfm3_quality_inputs import INPUTS_SHA256, read_bound, require, task_inputs
from task_cases import aggregate

TASK_LIMIT = 2
PAIRS = [
    ("stock_to_modified", ["stock", "modified_parent"]),
    ("modified_to_native", ["modified_parent", "native"]),
]


def check_loss(loss):
    require(loss["format"] == "toklet-lfm3-loss-report-v1" and loss["threshold_nats_per_token"] == LIMIT
            and loss["counts"] == dict(stock=TARGETS, modified_parent=TARGETS, native=TARGETS),
            "All 10,000 loss targets per profile are required")
    for label, _ in PAIRS:
        comparison = loss[label]
        mean = comparison["mean_nll_increase"]
        require(comparison["complete"] is True and comparison["paired_targets"] == TARGETS
                and type(mean) in (float, int) and math.isfinite(mean)
                and comparison["gate_passed"] is (mean <= LIMIT),
                "Complete original loss decision required")
    return loss


def check_grades(report, pair, cases):
    require(report["format"] == "toklet-lfm3-literal-task-grading-v1" and report["pair"] == pair
            and report["input_reconciliation_sha256"] == INPUTS_SHA256 and report["completed"] is True
            and report["restored_stopped"] is True and report["pipe_lifetime_finished"] is True
            and report["generated_code_runs_on_host"] is False
            and len(report["runs"]) == 80 and report["unresolved_case_ids"] == [],
            "Both isolated 100-task results must be complete")
    identities = report.get("decoded_inputs")
    fields = {"receipt_sha256", "collection_sha256", "decoded_tasks_sha256"}
    require(type(identities) is dict and set(identities) == {"baseline", "candidate"}
            and all(type(identity) is dict and set(identity) == fields
                    and all(type(v) is str and len(v) == 64 and all(c in "0123456789abcdef" for c in v)
                            for v in identity.values()) for identity in identities.values()),
            "Decoded task identities are missing or malformed")
    expected_code = {case["id"] for case in cases if case["category"] == "code"}
    for profile in ("baseline", "native"):
        ids = [run["id"] for run in report["runs"] if run["profile"] == profile]
        require(len(ids) == len(expected_code) == 40 and set(ids) == expected_code,
                "Isolated HumanEval coverage differs")
    baseline = aggregate(cases, report["outcomes"]["baseline"])
    candidate = aggregate(cases, report["outcomes"]["native"])
    points = baseline["passed"] - candidate["passed"]
    decision = report["decision"]
    require(decision["complete"] is True and decision["baseline_tasks"] == baseline
            and decision["native_tasks"] == candidate and decision["task_limit_percentage_points"] == TASK_LIMIT
            and decision["task_pass_rate_loss_percentage_points"] == points
            and decision["task_gate_passed"] is (points <= TASK_LIMIT), "Task gate or aggregation differs")
    return decision


def check_modified_identity(stock_pair, native_pair):
    require(stock_pair["decoded_inputs"]["candidate"] == native_pair["decoded_inputs"]["baseline"],
            "Quality comparisons used different modified-reference task inputs")
    old, new = stock_pair["outcomes"]["native"], native_pair["outcomes"]["baseline"]
    require([r["passed"] for r in old] == [r["passed"] for r in new],
            "The same modified-reference tasks resolved differently across pairs")


def run(args):
    loss = check_loss(json.loads(read_bound(args.loss, args.loss_sha256)))
    _, cases = task_inputs(args.inputs, "stock")
    specifications = [(args.stock_task_grades, args.stock_task_grades_sha256),
                      (args.native_task_grades, args.native_task_grades_sha256)]
    grades = [json.loads(read_bound(path, pin, 32 * 1024**2)) for path, pin in specifications]
    decisions = [check_grades(grade, pair, cases) for grade, (_, pair) in zip(grades, PAIRS, strict=True)]
    check_modified_identity(*grades)
    results = {}
    for (label, pair), task in zip(PAIRS, decisions, strict=True):
        results[label] = dict(profiles=pair, mean_nll_increase=loss[label]["mean_nll_increase"],
                              task_pass_rate_loss_percentage_points=task["task_pass_rate_loss_percentage_points"],
                              loss_gate_passed=loss[label]["gate_passed"], task_gate_passed=task["task_gate_passed"],
                              quality_gate_passed=loss[label]["gate_passed"] and task["task_gate_passed"])
    result = dict(format="toklet-lfm3-original-quality-decision-v1", complete=True, comparisons=results,
                  quality_gate_passed=all(r["quality_gate_passed"] for r in results.values()),
                  native_console_text_differences=grades[1]["console_text_differences"]["candidate"],
                  managed_adoption_qualified=False,
                  evidence_sha256=dict(loss=args.loss_sha256, stock_task_grades=args.stock_task_grades_sha256,
                                       native_task_grades=args.native_task_grades_sha256,
                                       quality_inputs=INPUTS_SHA256))
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "loss", "stock-task-grades", "native-task-grades", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("loss-sha256", "stock-task-grades-sha256", "native-task-grades-sha256"):
        parser.add_argument("--" + name, required=True)
    run(parser.parse_args())
