"""Offline task policy and response fixtures; no model or code execution."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import task_cases as tasks


def inventory():
    cases = []
    for category, total in tasks.CATEGORIES.items():
        for index in range(total):
            case = {"id": f"{category}/{index}", "category": category,
                    "prompt_tokens": [12, 34], "prompt_text": "already formatted",
                    "max_output_tokens": tasks.CAPS[category],
                    "stop_policy": f"eos_or_{tasks.CAPS[category]}", "scoring": tasks.SCORERS[category]}
            if category == "code":
                case.update(code_prefix="def answer():\n", tests="def check(candidate):\n    assert candidate() == 7\n", entry_point="answer")
            else:
                case["expected"] = "A" if category == "prose" else "World!"
            cases.append(case)
    policy = {"tasks_sha256": tasks.TASK_SHA, "categories": tasks.CATEGORIES.copy(),
              "case_policy": "first_source_rows_in_order_no_model_based_selection",
              "aggregation": "100_equal_weight_binary_cases", "sampling": "greedy_lowest_token_id_on_ties",
              "task_pass_rate_loss_limit_percentage_points": 2.0,
              "calibration": "held_out_sources_and_all_selected_cases_forbidden_for_quantizer_tuning",
              "task_ids": [case["id"] for case in cases]}
    return cases, policy


def events(case, count=None, eos=False, text="", tail="", projection="scalar"):
    count = case["max_output_tokens"] if count is None else count
    result = [{"event": "token", "index": index, "token_id": 200 + index,
               "elapsed_seconds": index + 0.5, "text": text if index == 0 else ""}
              for index in range(count)]
    sampled = count + int(eos)
    prompt = len(case["prompt_tokens"])
    generation = {"reason": {"kind": "stop_token", "token_id": 151645} if eos else {"kind": "output_limit"},
                  "reused_tokens": 0, "prefill_tokens": prompt, "sampled_tokens": sampled,
                  "emitted_tokens": count, "history_tokens": prompt + sampled,
                  "evaluated_tokens": prompt + sampled - 1, "pending_tokens": 1,
                  "prefill_seconds": 0.25, "decode_seconds": count / 2, "sampling_seconds": 0.01,
                  "total_seconds": count + 1, "first_token_seconds": 0.5 if count else None}
    if tail and eos:
        result.append({"event": "text", "text": tail})
    result.append({"event": "done", "statistics": {"cpu_projection": projection,
                  "input_formatting_seconds": 0.0, "tokenization_seconds": 0.0, "generation": generation}})
    if tail and not eos:
        result.append({"event": "text", "text": tail})
    result.append({"event": "reset"})
    return result


class FrozenPolicy(unittest.TestCase):
    def test_inventory_and_raw_ids_preserve_caps_and_independent_input(self):
        cases, policy = inventory()
        tasks.validate_cases(cases, policy)
        for case in cases:
            command = tasks.request(case)
            self.assertEqual(command, {"action": "generate", "tokens": [12, 34],
                                      "max_new_tokens": tasks.CAPS[case["category"]]})
            command["tokens"].append(55)
            self.assertEqual(case["prompt_tokens"], [12, 34])

    def test_inventory_order_policy_and_admission_corruption_reject(self):
        cases, policy = inventory()
        for field, value in [("sampling", "random"), ("aggregation", "category_weights"),
                             ("tasks_sha256", "changed"), ("task_pass_rate_loss_limit_percentage_points", 3)]:
            altered = {**policy, field: value}
            with self.subTest(field=field), self.assertRaises(ValueError): tasks.validate_cases(cases, altered)
        for mutate in (lambda c: c.pop(), lambda c: c.reverse(), lambda c: c[1].update(id=c[0]["id"]),
                       lambda c: c[0].update(category="prose"), lambda c: c[0].update(prompt_tokens=[True]),
                       lambda c: c[0].update(prompt_tokens=[tasks.VOCAB]), lambda c: c[0].update(prompt_tokens=[]),
                       lambda c: c[0].update(prompt_tokens=[1]*1793), lambda c: c[0].update(max_output_tokens=True),
                       lambda c: c[0].update(scoring="repair_and_run"), lambda c: c[0].update(entry_point="answer()")):
            altered = copy.deepcopy(cases); mutate(altered)
            with self.subTest(case=altered[0]), self.assertRaises(ValueError): tasks.validate_cases(altered, policy)

    def test_payload_identity_size_and_path_are_checked_before_use(self):
        cases, policy = inventory()
        payload = ("\n".join(json.dumps(case) for case in cases)+"\n").encode()
        sha = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root/"artifacts/quality-v1/tasks.jsonl"
            source.parent.mkdir(parents=True); source.write_bytes(payload)
            manifest = {"quality": {**policy, "tasks_sha256": sha, "tasks_file": "artifacts/quality-v1/tasks.jsonl"}}
            with patch.object(tasks.prepare, "ROOT", root), patch.object(tasks.prepare, "load_json", return_value=manifest), patch.object(tasks, "TASK_SHA", sha):
                self.assertEqual(tasks.load_cases(), cases)
                source.write_bytes(payload+b" ")
                with self.assertRaisesRegex(ValueError, "size/identity"): tasks.load_cases()
                source.write_bytes(payload)
                with patch.object(tasks, "MAX_PAYLOAD_BYTES", len(payload)-1), self.assertRaisesRegex(ValueError, "size/identity"): tasks.load_cases()
                manifest["quality"]["tasks_file"] = "outside.jsonl"
                with self.assertRaisesRegex(ValueError, "escapes"): tasks.load_cases()


class Response(unittest.TestCase):
    def setUp(self):
        self.case = inventory()[0][40]

    def test_output_limit_collects_reset_unicode_tail_before_grading(self):
        trace = events(self.case, text="A", tail="\ufffd")
        response = tasks.collect_response(self.case, trace, "scalar")
        self.assertEqual(response["visible_text"], "A\ufffd")
        self.assertEqual(response["visible_text_sha256"], hashlib.sha256("A\ufffd".encode()).hexdigest())
        self.assertFalse(tasks.grade_text(self.case, response["visible_text"]))
        self.assertEqual(len(response["visible_token_ids"]), 32)
        self.assertEqual(response["statistics"]["generation"]["pending_tokens"], 1)

    def test_eos_is_sampled_pending_and_absent_from_visible_ids(self):
        for count in (0, 1, 31):
            with self.subTest(count=count):
                trace = events(self.case, count=count, eos=True, text="A", tail=" ")
                response = tasks.collect_response(self.case, trace, "scalar")
                self.assertEqual(len(response["visible_token_ids"]), count)
                self.assertNotIn(151645, response["visible_token_ids"])
                self.assertEqual(response["stop_reason"], {"kind": "stop_token", "token_id": 151645})
                self.assertEqual(response["statistics"]["generation"]["sampled_tokens"], count+1)
                self.assertEqual(response["statistics"]["generation"]["evaluated_tokens"], 2+count)
                self.assertEqual(response["visible_text"], ("A" if count else "")+" ")

    def test_event_order_finalization_and_mode_errors_are_infrastructure(self):
        good = events(self.case)
        bad = [good[:-1], good+[{"event":"reset"}], [{"event":"error","message":"failed"}],
               [*good[:1], {"event":"text","text":"x"}, *good[1:]],
               events(self.case, projection="tiled")]
        for trace in bad:
            with self.subTest(trace=trace), self.assertRaises(ValueError): tasks.collect_response(self.case, trace, "scalar")
        trace = events(self.case, count=1, eos=True)
        trace.insert(-1, {"event":"text","text":"duplicated EOS finalization"})
        with self.assertRaisesRegex(ValueError, "EOS text"): tasks.collect_response(self.case, trace, "scalar")
        trace = events(self.case); trace.insert(-2, {"event":"text","text":"early output-limit tail"})
        with self.assertRaisesRegex(ValueError, "Output-limit text"): tasks.collect_response(self.case, trace, "scalar")

    def test_state_sampling_timing_and_token_corruption_reject(self):
        for field, value in [("pending_tokens",0),("evaluated_tokens",35),("sampled_tokens",31),
                             ("reused_tokens",1),("prefill_tokens",1),("emitted_tokens",True),
                             ("prefill_seconds",float("inf")),("sampling_seconds",-1),("first_token_seconds",None),
                             ("reason",{"kind":"cancelled"}),("reason",{"kind":"stop_token","token_id":1})]:
            trace=events(self.case); trace[-2]["statistics"]["generation"][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError): tasks.collect_response(self.case,trace,"scalar")
        for field,value in [("index",1),("token_id",True),("token_id",151643),("elapsed_seconds",float("nan")),("text",b"bytes")]:
            trace=events(self.case);trace[0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):tasks.collect_response(self.case,trace,"scalar")
        trace=events(self.case);trace[1]["elapsed_seconds"]=0.1
        with self.assertRaises(ValueError):tasks.collect_response(self.case,trace,"scalar")
        trace=events(self.case);trace[-2]["statistics"]["tokenization_seconds"]=0.01
        with self.assertRaisesRegex(ValueError,"formatted/tokenized"):tasks.collect_response(self.case,trace,"scalar")

    def test_response_bytes_and_extra_finalization_are_bounded(self):
        trace=events(self.case,text="\u4e2d"*21846)
        with self.assertRaisesRegex(ValueError,"byte bound"):tasks.collect_response(self.case,trace,"scalar")
        trace=events(self.case,tail="one");trace.insert(-1,{"event":"text","text":"two"})
        with self.assertRaisesRegex(ValueError,"repeats"):tasks.collect_response(self.case,trace,"scalar")


class Scoring(unittest.TestCase):
    def test_prose_requires_the_whole_trimmed_uppercase_letter(self):
        case=inventory()[0][40]
        for text in ["A"," \nA\t"]:self.assertTrue(tasks.grade_text(case,text))
        for text in ["a","(A)","A.","A because...","Answer: A","A\nA",""]:
            self.assertFalse(tasks.grade_text(case,text))

    def test_continuation_preserves_first_word_case_and_punctuation(self):
        case=inventory()[0][70]
        for text in ["World!"," \nWorld! extra words"]:self.assertTrue(tasks.grade_text(case,text))
        for text in ["world!","World","World!,","other World!",""]:
            self.assertFalse(tasks.grade_text(case,text))

    def test_code_is_literal_and_explicitly_invokes_all_source_tests(self):
        case=inventory()[0][0];completion="    return 7\n```\n"
        self.assertEqual(tasks.human_eval_program(case,completion),
                         "def answer():\n    return 7\n```\n\ndef check(candidate):\n    assert candidate() == 7\n\n\ncheck(answer)\n")
        with self.assertRaisesRegex(ValueError,"isolated"):tasks.grade_text(case,"    return 7")

    def test_equal_weight_aggregation_requires_all_resolved_cases(self):
        cases,_=inventory()
        outcomes=[{"id":case["id"],"category":case["category"],"status":"model_pass" if i<50 else "model_fail","passed":i<50}
                  for i,case in enumerate(cases)]
        result=tasks.aggregate(cases,outcomes)
        self.assertEqual((result["passed"],result["pass_rate"]),(50,0.5))
        self.assertEqual(result["categories"],{"code":{"cases":40,"passed":40},"prose":{"cases":30,"passed":10},"continuation":{"cases":30,"passed":0}})
        for altered in [outcomes[:-1],list(reversed(outcomes))]:
            with self.assertRaises(ValueError):tasks.aggregate(cases,altered)
        for field,value in [("status","infrastructure_failure"),("passed",1),("passed",False)]:
            altered=copy.deepcopy(outcomes);altered[0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):tasks.aggregate(cases,altered)


if __name__ == "__main__":
    unittest.main()
