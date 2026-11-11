"""Portable checks for task collection's stop and reset accounting."""
import hashlib
import json
import unittest

from lfm3_quality_decision import check_grades, check_modified_identity
from lfm3_quality_inputs import INPUTS_SHA256
from lfm3_quality_tasks import EOS, collect


class ScriptedConsole:
    def __init__(self, events):
        self.events = iter(events)
        self.requests = []

    def send(self, request):
        self.requests.append(request)

    def receive(self, _deadline):
        return next(self.events, None)


class TaskCollectionTests(unittest.TestCase):
    task = {"id": "fixture", "category": "code", "prompt_tokens": [3, 4], "max_output_tokens": 2}

    def done(self, reason, visible, sampled):
        return dict(event="done", statistics=dict(
            generation=dict(reason=reason, reused_tokens=0, prefill_tokens=2,
                            sampled_tokens=sampled, emitted_tokens=visible, history_tokens=2 + sampled,
                            evaluated_tokens=1 + sampled, pending_tokens=1),
            model_profile="lfm3-bf16", decode_evaluations=sampled - 1,
            tokenization_seconds=0, input_formatting_seconds=0,
            logical_memory={"owned": 20}, memory_admitted_bytes=20))

    def token(self, index=0, token_id=17, text="a"):
        return dict(event="token", index=index, token_id=token_id, text=text, elapsed_seconds=index + 1)

    def test_hidden_stop_counts_at_cap(self):
        stop = {"kind": "stop_token", "token_id": EOS}
        console = ScriptedConsole([self.token(), {"event": "text", "text": "b"},
                                   self.done(stop, 1, 2), {"event": "reset"}])
        record = collect(console, self.task, 0)
        self.assertEqual(record["visible_token_ids"], [17])
        self.assertEqual(record["visible_text"], "ab")
        self.assertEqual(console.requests, [
            {"action": "generate", "tokens": [3, 4], "max_new_tokens": 2}, {"action": "reset"},
        ])

    def test_immediate_stop_preserves_empty_answer(self):
        stop = {"kind": "stop_token", "token_id": EOS}
        record = collect(ScriptedConsole([self.done(stop, 0, 1), {"event": "reset"}]), self.task, 0)
        self.assertEqual(record["visible_token_ids"], [])
        self.assertEqual(record["visible_text"], "")

    def test_output_limit_text_finishes_at_reset(self):
        console = ScriptedConsole([self.token(), self.token(1, 18, "b"),
                                   self.done({"kind": "output_limit"}, 2, 2),
                                   {"event": "text", "text": "c"}, {"event": "reset"}])
        self.assertEqual(collect(console, self.task, 0)["visible_text"], "abc")

    def test_incomplete_and_wrong_counters_rejected(self):
        stop = {"kind": "stop_token", "token_id": EOS}
        bad = self.done(stop, 1, 2)
        bad["statistics"]["generation"]["reused_tokens"] = 1
        for events in [
            [self.token(), bad, {"event": "reset"}],
            [self.token(token_id=EOS), self.done(stop, 1, 2), {"event": "reset"}],
            [self.token(), self.done({"kind": "output_limit"}, 1, 1), {"event": "reset"}],
            [self.token(), self.done(stop, 1, 2)],
        ]:
            with self.subTest(events=events), self.assertRaises(ValueError):
                collect(ScriptedConsole(events), self.task, 0)


class PairedTaskEvidenceTests(unittest.TestCase):
    def test_same_modified_results_require_same_input_identity(self):
        completions = [{"id": index, "visible_text": "answer"} for index in range(100)]
        changed = completions.copy()
        changed[31] = {"id": 31, "visible_text": "different answer"}
        original_sha = hashlib.sha256(json.dumps(completions).encode()).hexdigest()
        changed_sha = hashlib.sha256(json.dumps(changed).encode()).hexdigest()
        identity = dict(receipt_sha256="1" * 64, collection_sha256="2" * 64, decoded_tasks_sha256=original_sha)
        outcomes = [{"passed": index % 2 == 0} for index in range(100)]
        stock = dict(decoded_inputs={"candidate": identity}, outcomes={"native": outcomes})
        native = dict(decoded_inputs={"baseline": identity}, outcomes={"baseline": outcomes})
        check_modified_identity(stock, native)
        variants = {"receipt_sha256": "a" * 64, "collection_sha256": "b" * 64, "decoded_tasks_sha256": changed_sha}
        for field, value in variants.items():
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "different modified-reference"):
                check_modified_identity(stock, native | {"decoded_inputs": {"baseline": identity | {field: value}}})

    def test_missing_and_malformed_decoded_provenance(self):
        pair = ["stock", "modified_parent"]
        otherwise_ready = dict(
            format="toklet-lfm3-literal-task-grading-v1", pair=pair, input_reconciliation_sha256=INPUTS_SHA256,
            completed=True, restored_stopped=True, pipe_lifetime_finished=True, generated_code_runs_on_host=False,
            runs=[None] * 80, unresolved_case_ids=[],
        )
        identity = dict(receipt_sha256="1" * 64, collection_sha256="2" * 64, decoded_tasks_sha256="3" * 64)
        for malformed in (
            {},
            {"decoded_inputs": {"candidate": identity}},
            {"decoded_inputs": {"baseline": identity, "candidate": identity | {"decoded_tasks_sha256": "g" * 64}}},
        ):
            with self.subTest(malformed=malformed), self.assertRaisesRegex(ValueError, "missing or malformed"):
                check_grades(otherwise_ready | malformed, pair, [])


if __name__ == "__main__":
    unittest.main()
