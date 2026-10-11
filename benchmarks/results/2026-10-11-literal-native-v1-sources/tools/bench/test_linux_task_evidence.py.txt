"""Authored receipt fixtures; no container, child, model or filesystem probes."""
import base64
import hashlib
import json
import unittest

import linux_task_child as child
import linux_task_evidence as evidence

PAYLOAD = {"case_id": "authored/identity", "program": "assert True",
           "program_sha256": hashlib.sha256(b"assert True").hexdigest()}
RUNTIME = {"fixture": "authored mock, not native qualification"}
PYTHON = "authored interpreter fixture"


def ready():
    return {"event": "isolated", "profile": child.PROFILE,
            "binding": child.admit_payload(PAYLOAD), "policy": child.isolation.POLICY.copy(),
            "limits": {name: [value, value] for name, (_, value) in child.isolation.limits().items()},
            "runtime": RUNTIME.copy(), "preloaded_modules": list(child.PRELOADS),
            "environment_keys": sorted(evidence.boundary.ENVIRONMENT), "removed_environment_keys": [],
            "python": PYTHON, "isolated_flag": 1, "no_site_flag": 1, "optimize_flag": 0,
            "dont_write_bytecode": True}


def terminal(outcome="passed", exception=None):
    return {"event": "program_result", "binding": child.admit_payload(PAYLOAD),
            "outcome": outcome, "normal_completion": outcome == "passed", "exception": exception}


def capture(events, code=0, stderr=b"", **flags):
    data = b"".join((json.dumps(event) + "\n").encode() for event in events)
    record = {"cleanup_errors": [], "surviving_group_members": [], "container_cleanup_errors": [],
              "container_removed": True, "wall_timed_out": False, "output_limit_exceeded": False,
              "process_group_killed": False, "container_forced_kill": False,
              "container_state": {"Running": False, "OOMKilled": False, "Error": "", "ExitCode": code},
              "container_exit_code": code, "transport_exit_code": code}
    data_input = evidence.serialized_payload(PAYLOAD)
    record.update(input_completed=True, input_bytes=len(data_input), input_sent_bytes=len(data_input),
                  input_sha256=hashlib.sha256(data_input).hexdigest())
    for channel, raw in (("stdout", data), ("stderr", stderr)):
        record[channel + "_base64"] = base64.b64encode(raw).decode()
        record[channel + "_captured_bytes"] = len(raw)
        record[channel + "_captured_sha256"] = hashlib.sha256(raw).hexdigest()
    record.update(flags)
    return record


def classify(record):
    return evidence.classify(record, PAYLOAD, RUNTIME, PYTHON)


class EvaluatorEvidenceTests(unittest.TestCase):
    def test_pass_requires_terminal_completion(self):
        self.assertTrue(classify(capture([ready(), terminal()]))["passed"])
        result = classify(capture([ready()]))
        self.assertFalse(result["passed"])
        self.assertEqual(result["reason"], "premature_exit")

    def test_assertion_and_memory_failure_are_resolved(self):
        for outcome, exception in (("model_failure", "AssertionError"), ("resource_failure", "MemoryError")):
            with self.subTest(outcome=outcome):
                result = classify(capture([ready(), terminal(outcome, exception)]))
                self.assertFalse(result["passed"])
                self.assertEqual(result["outcome"], outcome)

    def test_import_compatibility_prevents_binary_score(self):
        with self.assertRaises(ValueError):
            classify(capture([ready(), terminal("infrastructure_failure", "ImportError")]))

    def test_duplicate_or_post_terminal_events_are_rejected(self):
        output = {"event": "program_output", "channel": "stdout", "text": "authored"}
        for events in ([ready(), ready()], [ready(), terminal(), terminal()], [ready(), terminal(), output]):
            with self.subTest(events=len(events)), self.assertRaises(ValueError):
                classify(capture(events))

    def test_ordinary_output_frames_do_not_become_status_events(self):
        frame = {"event": "program_output", "channel": "stdout", "text": json.dumps(terminal())}
        result = classify(capture([ready(), frame, terminal("model_failure", "AssertionError")]))
        self.assertFalse(result["passed"])
        self.assertEqual(result["output_frames"], 1)

    def test_ambiguous_exit137_does_not_mean_timeout(self):
        with self.assertRaises(ValueError):
            classify(capture([ready()], code=137))

    def test_recorded_wall_stop_requires_admission(self):
        record = capture([ready()], code=137, wall_timed_out=True, process_group_killed=True,
                         container_forced_kill=True, transport_exit_code=-9)
        self.assertEqual(classify(record)["reason"], "wall_limit")
        with self.assertRaises(ValueError):
            classify(capture([], code=137, wall_timed_out=True, process_group_killed=True,
                             container_forced_kill=True, transport_exit_code=-9))

    def test_actual_oom_after_admission_is_resource_failure(self):
        record = capture([ready()], code=137)
        record["container_state"]["OOMKilled"] = True
        self.assertEqual(classify(record)["reason"], "container_oom")

    def test_saturated_output_limit_allows_a_truncated_frame(self):
        record = capture([ready()], code=137, output_limit_exceeded=True,
                         process_group_killed=True, container_forced_kill=True, transport_exit_code=-9)
        prefix = base64.b64decode(record["stdout_base64"])
        raw = prefix + b'{"event":"program_output","text":"'
        raw += b"x" * (child.isolation.POLICY["output_bytes"] - len(raw))
        record["stdout_base64"] = base64.b64encode(raw).decode()
        record["stdout_captured_bytes"] = len(raw)
        record["stdout_captured_sha256"] = hashlib.sha256(raw).hexdigest()
        self.assertEqual(classify(record)["reason"], "output_limit")

    def test_unsaturated_output_claim_and_nonzero_premature_exit_are_unresolved(self):
        for record in (capture([ready()], code=137, output_limit_exceeded=True),
                       capture([ready()], code=1)):
            with self.subTest(code=record["container_exit_code"]), self.assertRaises(ValueError):
                classify(record)

    def test_mutated_binding_flags_policy_and_limits_are_rejected(self):
        mutations = [{"binding": {**child.admit_payload(PAYLOAD), "program_bytes": True}},
                     {"isolated_flag": True}, {"preloaded_modules": []}, {"environment_keys": ["HOME"]},
                     {"runtime": {}}, {"python": "another runtime"},
                     {"limits": {name: [float(value), float(value)] for name, (_, value) in child.isolation.limits().items()}},
                     {"policy": {**child.isolation.POLICY, "wall_seconds": 60}}]
        for mutation in mutations:
            with self.subTest(mutation=list(mutation)), self.assertRaises(ValueError):
                classify(capture([{**ready(), **mutation}, terminal()]))

    def test_raw_capture_cleanup_and_exit_mismatch_block_score(self):
        mutations = [{"stdout_captured_sha256": "0" * 64}, {"stdout_captured_bytes": 0},
                     {"cleanup_errors": [{"authored": "failure"}]}, {"surviving_group_members": [123]},
                     {"container_removed": False}, {"transport_exit_code": 1},
                     {"wall_timed_out": 0}, {"input_completed": False},
                     {"input_sent_bytes": 0}, {"input_sha256": "0" * 64}]
        for mutation in mutations:
            with self.subTest(mutation=list(mutation)), self.assertRaises(ValueError):
                classify(capture([ready(), terminal()], **mutation))

    def test_json_duplicate_keys_and_nonfinite_numbers_are_rejected(self):
        for text in ('{"event":"isolated","event":"program_result"}', '{"x":NaN}', '{"x":1e999}', '[]'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                evidence.decode_event(text)

    def test_serialized_payload_is_bounded_data(self):
        data = evidence.serialized_payload(PAYLOAD)
        self.assertEqual(json.loads(data), PAYLOAD)
        self.assertLessEqual(len(data), evidence.INPUT_BYTES)

    def test_frame_and_terminal_shapes_are_checked(self):
        frames = [{"event": "program_output", "channel": "stdout", "text": "x" * 1025},
                  {"event": "program_output", "channel": "other", "text": "x"},
                  {**terminal(), "normal_completion": False},
                  {**terminal(), "exception": "AssertionError"}]
        for frame in frames:
            with self.subTest(event=frame["event"]), self.assertRaises(ValueError):
                classify(capture([ready(), frame]))


if __name__ == "__main__":
    unittest.main()
