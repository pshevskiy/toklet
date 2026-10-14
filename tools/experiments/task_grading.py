"""Literal grading after all collected model processes exit.

The runtime window admits the qualified image, machine/API and controller
source identities, and owns restoration. This runner cannot start a VM or
choose candidate commands. Any unresolved case prohibits an aggregate.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import linux_code_isolation_probe as boundary
import linux_task_controller as controller
import linux_task_evidence as evidence
import task_cases as tasks
import task_grade_admission as admission


def check_collection(collection_path, collection):
    if (boundary.digest(collection_path) != collection['collection_sha256']
            or boundary.digest(Path(collection_path).with_suffix('.raw.jsonl')) != collection['raw_sha256']):
        raise ValueError('Admitted task collection changed')


def run(api, image_id, runtime, python_version, collection_path, output,*,collection_validator=None,extra_identity_modules=()):
    output = Path(output).resolve()
    if not output.is_relative_to((boundary.ROOT / "benchmarks/raw").resolve()) or output.exists():
        raise ValueError("Fresh held-out grading path is required")
    collection = (collection_validator or admission.validate_collection)(collection_path)
    cases = collection["cases"]
    check_collection(collection_path, collection)
    source = controller.bundled_source()
    controller.admit_bootstrap(source)
    modules = (controller, evidence, evidence.child, evidence.child.isolation,
               evidence.child.isolation.authored, tasks, tasks.prepare, boundary)
    identities = {str(Path(module.__file__).resolve().relative_to(boundary.ROOT)): boundary.digest(Path(module.__file__))
                  for module in modules}
    for filename in ("linux_task_transport.py", "task_cases_probe.py", "code_isolation_probe.py"):
        identities['tools/bench/' + filename] = boundary.digest(boundary.ROOT / 'tools/bench' / filename)
    for module in (admission, admission.metal, admission.reference, admission.controlled, admission.metal_engine, admission.metal.cli, admission.reference.native_usage,
                   admission.controlled.frozen, admission.controlled.common,
                   admission.controlled.metal_prefill, admission.controlled.metal_continue):
        identities[str(Path(module.__file__).resolve().relative_to(boundary.ROOT))] = boundary.digest(module.__file__)
    for module in extra_identity_modules:
        identities[str(Path(module.__file__).resolve().relative_to(boundary.ROOT))] = boundary.digest(module.__file__)
    identities[str(Path(__file__).resolve().relative_to(boundary.ROOT))] = boundary.digest(Path(__file__))
    record = {"format": "toklet-backend-task-grading-v1", "completed": False,
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Literal frozen model response source-test and text outcomes; unresolved infrastructure prohibits aggregate. No adversarial grading attestation or timing acceptance.",
              "collection_sha256": collection["collection_sha256"],
              "identity": collection["identity"], "native_raw_sha256": collection["raw_sha256"], "task_payload_sha256": tasks.TASK_SHA,
              "source_sha256": identities, "image_id": image_id, "runtime": runtime,
              "python_version": python_version, "runs": [], "outcomes": []}
    try:
        for case, response in zip(cases, collection["responses"], strict=True):
            outcome = {"id": case["id"], "category": case["category"]}
            if case["category"] != "code":
                passed = tasks.grade_text(case, response["visible_text"])
            else:
                program = tasks.human_eval_program(case, response["visible_text"])
                payload = {"case_id": case["id"], "program": program,
                           "program_sha256": hashlib.sha256(program.encode()).hexdigest()}
                # Original dataset program stays in the ignored frozen payload;
                # published receipts retain binding hashes and literal results.
                check_collection(collection_path, collection)
                raw = controller.execute(api, image_id, source, payload)
                record["runs"].append({"id": case["id"], "raw": raw,
                                       "program_sha256": payload["program_sha256"],
                                       "completion_sha256": response["visible_text_sha256"]})
                try:
                    resolved = evidence.classify(raw, payload, runtime, python_version)
                except ValueError as error:
                    # Preserve unresolved compatibility/protocol/exit evidence.
                    # Do not continue creating containers after uncertain cleanup.
                    if (raw.get("failure") or raw.get("container_removed") is not True
                            or raw.get("container_cleanup_errors") != []
                            or raw.get("cleanup_errors") != []
                            or raw.get("surviving_group_members") != []):
                        raise
                    record["runs"][-1]["infrastructure_failure"] = str(error)
                    outcome.update(passed=None, status="infrastructure_failure")
                    record["outcomes"].append(outcome)
                    continue
                record["runs"][-1]["resolved"] = resolved
                passed = resolved["passed"]
            outcome.update(passed=passed, status="model_pass" if passed else "model_fail")
            record["outcomes"].append(outcome)
        record["unresolved_case_ids"] = [item["id"] for item in record["outcomes"]
                                        if item["status"] == "infrastructure_failure"]
        record["quality_completed"] = not record["unresolved_case_ids"]
        record["aggregate"] = (tasks.aggregate(cases, record["outcomes"])
                               if record["quality_completed"] else None)
        if identities != {name: boundary.digest(boundary.ROOT/name) for name in identities}:
            raise ValueError("Held-out grading source changed")
        if (boundary.digest(collection_path) != record["collection_sha256"]
                or boundary.digest(Path(collection_path).with_suffix(".raw.jsonl")) != record["native_raw_sha256"]
                or tasks.load_cases() != cases):
            raise ValueError("Held-out grading response/input changed")
        record["completed"] = True
    except BaseException as error:
        record["failure"] = {"type": type(error).__name__, "message": str(error)[:4096]}
        record["quality_completed"] = False
        record["aggregate"] = None
    finally:
        record["completed_utc"] = datetime.now(timezone.utc).isoformat()
        with output.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
            stream.write("\n")
    return record
