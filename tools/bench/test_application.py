"""Authored application requests retain the frozen transcript identity."""
import unittest
from unittest.mock import patch
import application
import prepare


class Transcript(unittest.TestCase):
    def test_full_authored_turns_and_output_policy(self):
        manifest = prepare.load_json(prepare.CONTRACT / "manifest.json")
        requests = application.trace_requests(manifest, 128)
        self.assertEqual([len(r["messages"]) for r in requests], [2, 4, 6])
        self.assertTrue(all(r["action"] == "generate" and r["max_new_tokens"] == 128 for r in requests))
        self.assertEqual(requests[-1]["messages"], prepare.load_json(prepare.CONTRACT / "application_trace.json"))
        self.assertEqual(requests[1]["messages"][2]["role"], "assistant")
        with patch.object(application.prepare, "digest", return_value="changed"):
            with self.assertRaisesRegex(ValueError, "transcript changed"):
                application.trace_requests(manifest, 128)


if __name__ == "__main__":
    unittest.main()
