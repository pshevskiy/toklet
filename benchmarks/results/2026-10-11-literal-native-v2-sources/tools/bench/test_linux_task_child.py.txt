"""Authored offline tests for child semantics; no model/data/runtime admission."""
import hashlib
import io
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import linux_task_child as child


def payload(program):
    return {"case_id": "authored/identity", "program": program,
            "program_sha256": hashlib.sha256(program.encode()).hexdigest()}


class LiteralChildTests(unittest.TestCase):
    def test_full_source_tests_return_normally(self):
        result = child.evaluate("def candidate(x):\n    return x + 1\ndef check(f):\n    assert f(2) == 3\ncheck(candidate)\n")
        self.assertEqual(result, {"outcome": "passed", "normal_completion": True, "exception": None})

    def test_source_test_assertion_and_missing_candidate_fail(self):
        for program, exception in (("assert False", "AssertionError"),
                                   ("check(candidate)", "NameError"),
                                   ("def broken(:", "SyntaxError")):
            with self.subTest(program=program):
                self.assertEqual(child.evaluate(program),
                                 {"outcome": "model_failure", "normal_completion": False, "exception": exception})

    def test_exit_zero_and_interrupt_do_not_pass(self):
        for program, exception in (("raise SystemExit(0)", "SystemExit"),
                                   ("raise KeyboardInterrupt", "KeyboardInterrupt"),
                                   ("def candidate():\n    raise SystemExit(0)\ncandidate()", "SystemExit")):
            with self.subTest(program=program):
                self.assertEqual(child.evaluate(program)["exception"], exception)
                self.assertFalse(child.evaluate(program)["normal_completion"])

    def test_memory_exhaustion_is_resource_failure(self):
        self.assertEqual(child.evaluate("raise MemoryError")["outcome"], "resource_failure")

    def test_import_failure_leaves_case_unresolved(self):
        self.assertEqual(child.evaluate("raise ImportError('authored')")["outcome"], "infrastructure_failure")

    def test_preloading_does_not_inject_candidate_names(self):
        self.assertEqual(child.evaluate("assert math.sqrt(4) == 2")["exception"], "NameError")
        self.assertTrue(child.evaluate("import math\nassert math.sqrt(4) == 2")["normal_completion"])

    def test_payload_binding_and_bound(self):
        self.assertEqual(child.admit_payload(payload("assert True"))["program_bytes"], 11)
        for item in (None, {}, {**payload("assert True"), "program_sha256": "0" * 64},
                     payload("x" * (child.PROGRAM_BYTES + 1)),
                     {**payload("assert True"), "case_id": "x" * 257},
                     {**payload("assert True"), "unexpected": True}):
            with self.subTest(item_type=type(item).__name__):
                with self.assertRaises(ValueError):
                    child.admit_payload(item)

    def test_preload_inventory_does_not_consult_payload(self):
        with patch.object(child.importlib, "import_module") as imported:
            self.assertEqual(child.preload(), list(child.PRELOADS))
        self.assertEqual([call.args[0] for call in imported.call_args_list], list(child.PRELOADS))

    def test_input_pipe_is_bounded_and_rejects_duplicate_fields(self):
        item = payload("assert True")
        self.assertEqual(child.read_payload(io.BytesIO(json.dumps(item).encode())), item)
        for data in (b"", b"x" * (child.INPUT_BYTES + 1),
                     b'{"case_id":"one","case_id":"two"}',
                     json.dumps(item).encode() + b" trailing", b"\xff"):
            with self.subTest(size=len(data)), self.assertRaises((ValueError, UnicodeError)):
                child.read_payload(io.BytesIO(data))

    def test_ordinary_output_is_framed_and_bounded_per_frame(self):
        records = []
        writer = child.FramedOutput(records.append, "stdout")
        text = "\u2603" * 2050
        self.assertEqual(writer.write(text), len(text))
        self.assertEqual("".join(row["text"] for row in records), text)
        self.assertTrue(all(row["event"] == "program_output" and len(row["text"]) <= 1024 for row in records))

    def test_writelines_and_unsupported_stream_features(self):
        records = []
        writer = child.FramedOutput(records.append, "stdout")
        writer.writelines(["one", "two"])
        self.assertEqual("".join(row["text"] for row in records), "onetwo")
        with self.assertRaises(io.UnsupportedOperation):
            writer.fileno()
        with self.assertRaises(io.UnsupportedOperation):
            _ = writer.buffer
        self.assertEqual(child.evaluate("import io\nraise io.UnsupportedOperation('authored')")["outcome"],
                         "infrastructure_failure")

    def test_closed_output_rejects_write_flush_and_print(self):
        records = []
        writer = child.FramedOutput(records.append, "stdout")
        writer.close()
        for operation in (lambda: writer.write("after close"), writer.flush, writer.writable, writer.isatty,
                          lambda: print("after close", file=writer)):
            with self.subTest(operation=operation), self.assertRaises(ValueError):
                operation()
        self.assertEqual(records, [])

    def run_authored(self, program):
        output = io.StringIO()
        flags = SimpleNamespace(optimize=0, isolated=1, no_site=1)
        with patch.object(child.sys, "stdout", output), patch.object(child.sys, "flags", flags), \
                patch.object(child.sys, "dont_write_bytecode", True), \
                patch.object(child, "preload", return_value=list(child.PRELOADS)), \
                patch.object(child.isolation, "normalize_environment", return_value=[]), \
                patch.object(child.isolation, "enter", return_value=({}, {"authored_mock": True})):
            code = child.main(payload(program))
        return code, [json.loads(line) for line in output.getvalue().splitlines()]

    def test_admission_precedes_candidate_output_and_completion(self):
        code, rows = self.run_authored("print('authored output')\nassert True")
        self.assertEqual(code, 0)
        self.assertEqual(rows[0]["event"], "isolated")
        self.assertEqual(rows[-1]["event"], "program_result")
        self.assertEqual(rows[-1]["outcome"], "passed")
        self.assertTrue(all(row["event"] == "program_output" for row in rows[1:-1]))
        self.assertEqual(rows[0]["binding"], rows[-1]["binding"])

    def test_setup_failure_never_evaluates_program(self):
        output = io.StringIO()
        with patch.object(child.sys, "stdout", output), \
                patch.object(child.isolation, "enter", side_effect=ValueError("authored admission")), \
                patch.object(child, "preload", return_value=list(child.PRELOADS)), \
                patch.object(child.isolation, "normalize_environment", return_value=[]), \
                patch.object(child, "evaluate") as evaluated:
            self.assertEqual(child.main(payload("assert True")), 2)
            evaluated.assert_not_called()
        self.assertEqual([json.loads(line)["event"] for line in output.getvalue().splitlines()], ["setup_error"])


if __name__ == "__main__":
    unittest.main()
