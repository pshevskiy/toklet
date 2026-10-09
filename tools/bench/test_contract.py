"""Offline integrity and schema checks; no models, datasets or GPU needed."""

import copy
from pathlib import Path
import struct
import tempfile
import unittest

from jsonschema import Draft202012Validator, ValidationError

import prepare


class ContractTests(unittest.TestCase):
    def test_frozen_contract_verifies_without_external_sources(self):
        prepare.verify(prepare.CONTRACT)

    def test_token_reader_rejects_truncated_data(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokens"
            path.write_bytes(b"\x01\x00\x00")
            with self.assertRaises(ValueError):
                prepare.read_tokens(path)
            path.write_bytes(struct.pack("<2I", 1, 151935))
            self.assertEqual(prepare.read_tokens(path), [1, 151935])

    def test_schema_rejects_incomplete_identity_and_short_measurement_runs(self):
        schema = prepare.load_json(prepare.BENCH / "result.schema.json")
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema)
        result = {}
        self.assertTrue(list(validator.iter_errors(result)))
        # Validate the individual sample schema without inventing a result record.
        samples = copy.deepcopy(schema["properties"]["samples"])
        sample = {
            "prefill_seconds": 0.1, "ttft_seconds": 0.11,
            "decode_seconds": 0.2, "total_seconds": 0.31,
            "first_visible_text_seconds": None,
            "token_event_seconds": [0.11, 0.31], "emitted_tokens": 2,
            "process_peak_footprint_bytes": None, "gpu_peak_allocation_bytes": None,
        }
        Draft202012Validator(samples).validate([sample] * 10)
        with self.assertRaises(ValidationError):
            Draft202012Validator(samples).validate([sample] * 9)
        sample["ttft_seconds"] = -1
        with self.assertRaises(ValidationError):
            Draft202012Validator(samples).validate([sample] * 10)


if __name__ == "__main__":
    unittest.main()
