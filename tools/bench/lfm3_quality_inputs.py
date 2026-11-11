"""Admit the frozen LFM3 task payload before running any model."""
from collections import Counter
import hashlib
import json

INPUTS_SHA256 = "008745f83d7d4bd6ba06c43b95cd0ab16e9bfd2966e0c143a307f1b5e3f794d5"
CAPS = {"code": 256, "prose": 32, "continuation": 16}
COUNTS = {"code": 40, "prose": 30, "continuation": 30}
VOCAB = 128000
EOS = 124900


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_bound(path, expected, maximum=2 * 1024**2):
    with path.open("rb") as stream:
        raw = stream.read(maximum + 1)
    require(len(raw) <= maximum and hashlib.sha256(raw).hexdigest() == expected,
            f"Frozen quality input differs: {path.name}")
    return raw


def task_inputs(folder, role):
    profile = {"stock": "stock", "modified_parent": "native", "native": "native"}[role]
    reconciliation = json.loads(read_bound(folder / "reconciliation.json", INPUTS_SHA256))
    require(reconciliation["passed"] and reconciliation["scored_models"] is False,
            "Prospective input reconciliation required")
    grading = [json.loads(line) for line in read_bound(
        folder / "cases.jsonl", reconciliation["grader_cases_sha256"]).splitlines()]
    raw = read_bound(folder / profile / "tasks.jsonl", reconciliation[profile + "_tasks_sha256"])
    tasks = [json.loads(line) for line in raw.splitlines()]
    require(len(tasks) == len(grading) == 100 and len({r["id"] for r in tasks}) == 100,
            "Every original quality case is required")
    require(Counter(row["category"] for row in tasks) == COUNTS, "Task category inventory differs")
    for task, grade in zip(tasks, grading, strict=True):
        category = task["category"]
        prompt = task["prompt_tokens"]
        cap = task["max_output_tokens"]
        require(task["id"] == grade["id"] and category == grade["category"] and
                type(cap) is int and cap == grade["max_output_tokens"] == CAPS[category],
                "Task identity or output cap differs")
        require(isinstance(task["prompt"], str) and type(prompt) is list and prompt and
                all(type(token) is int and 0 <= token < VOCAB for token in prompt) and
                task["required_capacity"] == len(prompt) + cap <= 2048,
                "Task prompt or capacity differs")
    return tasks, grading
