"""Frozen task admission, finalized CLI responses and deterministic scoring.

This module does not execute model or generated code. HumanEval programs must
go to a separately qualified isolated executor; text graders need no executor.
"""
from collections import Counter
import hashlib
import json
import keyword
import math

import prepare

TASK_SHA = "b096b8eb40960f38ae0e9f3268484eed10bf2a6bebbc981a45a061ee461aab7c"
VOCAB = 151936
CAPACITY = 2048
EOS_IDS = (151645, 151643)
MAX_PAYLOAD_BYTES = 1024 * 1024
MAX_RESPONSE_BYTES = 65536
CATEGORIES = {"code": 40, "prose": 30, "continuation": 30}
CAPS = {"code": 256, "prose": 32, "continuation": 16}
SCORERS = {
    "code": "append_completion_and_run_source_tests",
    "prose": "trimmed_text_equals_expected_letter",
    "continuation": "first_whitespace_delimited_word_equals_expected",
}


def validate_case(case):
    if not isinstance(case, dict):
        raise ValueError("Task must be an object")
    category = case.get("category")
    if category not in CATEGORIES or not isinstance(case.get("id"), str) or not case["id"]:
        raise ValueError("Task identity/category differs")
    cap = CAPS[category]
    tokens = case.get("prompt_tokens")
    if (not isinstance(tokens, list) or not 0 < len(tokens) <= CAPACITY - cap
            or any(type(token) is not int or not 0 <= token < VOCAB for token in tokens)
            or not isinstance(case.get("prompt_text"), str)
            or type(case.get("max_output_tokens")) is not int or case["max_output_tokens"] != cap
            or case.get("stop_policy") != f"eos_or_{cap}"
            or case.get("scoring") != SCORERS[category]):
        raise ValueError("Task prompt/admission/scoring policy differs")
    if category == "code":
        entry = case.get("entry_point")
        if (not isinstance(entry, str) or not entry.isidentifier() or keyword.iskeyword(entry)
                or any(not isinstance(case.get(key), str) or not case[key] for key in ("code_prefix", "tests"))):
            raise ValueError("HumanEval source/entry point differs")
    else:
        expected = case.get("expected")
        if not isinstance(expected, str) or not expected or len(expected.split()) != 1 or expected.strip() != expected:
            raise ValueError("Task expected answer differs")
        if category == "prose" and (len(expected) != 1 or not "A" <= expected <= "Z"):
            raise ValueError("Prose expected answer must be one uppercase letter")


def validate_cases(cases, policy):
    expected_policy = {
        "tasks_sha256": TASK_SHA, "categories": CATEGORIES,
        "case_policy": "first_source_rows_in_order_no_model_based_selection",
        "aggregation": "100_equal_weight_binary_cases",
        "sampling": "greedy_lowest_token_id_on_ties",
        "task_pass_rate_loss_limit_percentage_points": 2.0,
        "calibration": "held_out_sources_and_all_selected_cases_forbidden_for_quantizer_tuning",
    }
    if not isinstance(policy, dict) or any(policy.get(key) != value for key, value in expected_policy.items()):
        raise ValueError("Frozen task quality policy differs")
    if (not isinstance(cases, list) or len(cases) != 100 or any(not isinstance(case, dict) for case in cases)
            or [case.get("id") for case in cases] != policy.get("task_ids")
            or len({case.get("id") for case in cases}) != 100
            or Counter(case.get("category") for case in cases) != CATEGORIES):
        raise ValueError("Frozen task inventory/order differs")
    for case in cases:
        validate_case(case)


def load_cases():
    manifest = prepare.load_json(prepare.CONTRACT / "manifest.json")
    policy = manifest["quality"]
    path = prepare.ROOT / policy["tasks_file"]
    if not path.resolve().is_relative_to((prepare.ROOT / "artifacts/quality-v1").resolve()):
        raise ValueError("Task payload path escapes its frozen directory")
    with path.open("rb") as stream:
        payload = stream.read(MAX_PAYLOAD_BYTES + 1)
    if len(payload) > MAX_PAYLOAD_BYTES or hashlib.sha256(payload).hexdigest() != TASK_SHA:
        raise ValueError("Frozen task payload size/identity differs")
    cases = [json.loads(line) for line in payload.decode("utf-8").splitlines()]
    validate_cases(cases, policy)
    return cases


def request(case):
    validate_case(case)
    # Prepared code/prose IDs already contain the pinned chat template. Raw
    # token input avoids a second formatting/tokenization pass.
    return {"action": "generate", "tokens": case["prompt_tokens"].copy(),
            "max_new_tokens": case["max_output_tokens"]}


def collect_response(case, events, projection):
    """Validate one empty-session generation followed by explicit reset.

    The caller supplies bounded JSON events through the reset acknowledgement.
    Finalization text belongs to this response even when it follows `done`.
    Transport/admission errors raise and cannot count as failed model answers.
    """
    validate_case(case)
    if projection not in ("scalar", "tiled"):
        raise ValueError("Task CPU projection differs")
    ids, text, elapsed = [], [], []
    final_texts = reset_texts = 0
    size = 0
    phase = "generate"
    statistics = None
    for event in events:
        if not isinstance(event, dict):
            raise ValueError("Task event must be an object")
        kind = event.get("event")
        if kind == "token" and phase == "generate":
            token = event.get("token_id")
            if (type(event.get("index")) is not int or event["index"] != len(ids)
                    or type(token) is not int or not 0 <= token < VOCAB or token in EOS_IDS
                    or len(ids) >= case["max_output_tokens"]
                    or type(event.get("elapsed_seconds")) not in (int, float)
                    or not math.isfinite(event["elapsed_seconds"]) or event["elapsed_seconds"] < 0
                    or elapsed and event["elapsed_seconds"] < elapsed[-1]):
                raise ValueError("Task visible token event differs")
            ids.append(token)
            elapsed.append(event["elapsed_seconds"])
        elif kind == "text" and phase in ("generate", "finalize", "reset"):
            if phase in ("generate", "finalize"):
                final_texts += 1
                phase = "finalize"
            else:
                reset_texts += 1
            if final_texts > 1 or reset_texts > 1:
                raise ValueError("Task repeats finalization text events")
        elif kind == "done" and phase in ("generate", "finalize"):
            statistics = event.get("statistics")
            phase = "reset"
        elif kind == "reset" and phase == "reset":
            phase = "complete"
        else:
            raise ValueError("Task generation/reset event order differs")
        if kind in ("token", "text"):
            chunk = event.get("text")
            if not isinstance(chunk, str):
                raise ValueError("Task text payload differs")
            size += len(chunk.encode("utf-8"))
            if size > MAX_RESPONSE_BYTES:
                raise ValueError("Task response exceeds transport byte bound")
            text.append(chunk)
    if phase != "complete" or not isinstance(statistics, dict) or statistics.get("cpu_projection") != projection:
        raise ValueError("Task response lacks completed reset or requested CPU mode")
    if any(type(statistics.get(key)) not in (int, float) or statistics[key] != 0
           for key in ("input_formatting_seconds", "tokenization_seconds")):
        raise ValueError("Prepared task IDs were formatted/tokenized again")
    generation = statistics.get("generation", {})
    if not isinstance(generation, dict):
        raise ValueError("Task generation statistics must be an object")
    reason = generation.get("reason", {})
    if not isinstance(reason, dict):
        raise ValueError("Task stop reason must be an object")
    cap = case["max_output_tokens"]
    if reason == {"kind": "output_limit"}:
        sampled = cap
        if len(ids) != cap:
            raise ValueError("Task output-limit coverage differs")
        if final_texts:
            raise ValueError("Output-limit text must finalize at reset")
    elif (reason.get("kind") == "stop_token" and set(reason) == {"kind", "token_id"}
          and type(reason["token_id"]) is int and reason["token_id"] in EOS_IDS):
        sampled = len(ids) + 1
        if sampled > cap:
            raise ValueError("Task sampled stop exceeds output admission")
        if reset_texts:
            raise ValueError("EOS text must finalize before done")
    else:
        raise ValueError("Task stop policy differs")
    prompt = len(case["prompt_tokens"])
    expected = {"reused_tokens": 0, "prefill_tokens": prompt, "sampled_tokens": sampled,
                "emitted_tokens": len(ids), "history_tokens": prompt + sampled,
                "evaluated_tokens": prompt + sampled - 1, "pending_tokens": 1}
    if any(type(generation.get(key)) is not int or generation[key] != value for key, value in expected.items()):
        raise ValueError("Task committed/pending/sample state differs")
    timers = ("prefill_seconds", "decode_seconds", "sampling_seconds", "total_seconds")
    first = generation.get("first_token_seconds")
    if ("first_token_seconds" not in generation
            or any(type(generation.get(key)) not in (int, float) or not math.isfinite(generation[key])
            or generation[key] < 0 for key in timers)
            or any(generation[key] > generation["total_seconds"] for key in timers[:-1])
            or elapsed and type(first) not in (int, float)
            or first != (elapsed[0] if elapsed else None)
            or elapsed and elapsed[-1] > generation["total_seconds"]):
        raise ValueError("Task generation timing evidence differs")
    visible = "".join(text)
    return {"id": case["id"], "category": case["category"], "visible_token_ids": ids,
            "visible_text": visible, "visible_text_sha256": hashlib.sha256(visible.encode()).hexdigest(),
            "stop_reason": reason.copy(), "statistics": statistics}


def grade_text(case, completion):
    validate_case(case)
    if not isinstance(completion, str):
        raise ValueError("Task completion must be text")
    if case["category"] == "prose":
        return completion.strip() == case["expected"]
    if case["category"] == "continuation":
        words = completion.split()
        return bool(words) and words[0] == case["expected"]
    raise ValueError("HumanEval requires qualified isolated source-test execution")


def human_eval_program(case, completion):
    validate_case(case)
    if case["category"] != "code" or not isinstance(completion, str):
        raise ValueError("HumanEval requires literal code completion")
    # No trimming, fence removal, signature replacement or answer repair.
    # Explicit invocation prevents treating a definition-only zero exit as pass.
    return case["code_prefix"] + completion + "\n" + case["tests"] + "\n\ncheck(" + case["entry_point"] + ")\n"


def aggregate(cases, outcomes):
    """Equal-weight binary aggregation requires all 100 resolved model cases."""
    if len(cases) != 100 or len(outcomes) != 100:
        raise ValueError("Task aggregation requires all 100 cases")
    if len({case["id"] for case in cases}) != 100 or Counter(case["category"] for case in cases) != CATEGORIES:
        raise ValueError("Task aggregation inventory differs")
    counts = {category: {"cases": total, "passed": 0} for category, total in CATEGORIES.items()}
    for case, outcome in zip(cases, outcomes):
        if outcome.get("id") != case["id"] or outcome.get("category") != case["category"]:
            raise ValueError("Task outcome order/identity differs")
        if outcome.get("status") not in ("model_pass", "model_fail"):
            raise ValueError("Unresolved task infrastructure failure cannot be aggregated")
        passed = outcome.get("passed")
        if type(passed) is not bool or passed != (outcome["status"] == "model_pass"):
            raise ValueError("Task outcome is not consistent binary evidence")
        counts[case["category"]]["passed"] += passed
    total = sum(count["passed"] for count in counts.values())
    return {"cases": 100, "passed": total, "pass_rate": total / 100,
            "aggregation": "100_equal_weight_binary_cases", "categories": counts}
