"""Portable failure controls and explicit real-inventory replay mutations."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

import capture_gemma_resident_storage as capture
import check_gemma_resident_storage as replay

ROOT = Path(__file__).resolve().parents[2]
METADATA = ROOT / "tests/fixtures/gemma4-e2b-metadata-v1"


class ShortReader(io.BytesIO):
    def read(self, count=-1):
        return super().read(min(count, 1))


class WrongSeek(io.BytesIO):
    def seek(self, offset, whence=0):
        return super().seek(offset + 1, whence)


def toy_container(payload=b"\0\x80\0\x00\x80\x3f"):
    header = json.dumps({"x": dict(dtype="BF16", shape=[3], data_offsets=[0, len(payload)])}).encode()
    raw = struct.pack("<Q", len(header)) + header + payload
    return raw, {"header.json": header}


def toy_auth(raw, metadata, reader_type=io.BytesIO):
    events = []
    with patch.object(capture, "MODEL_BYTES", len(raw)), patch.object(
            capture, "MODEL_SHA", hashlib.sha256(raw).hexdigest()):
        result = capture.phase_one(reader_type(raw), metadata, {"x"},
                                   lambda key, value: events.append((key, value)))
    return result, events


class Portable(unittest.TestCase):
    def test_one_byte_reads_authenticate_exact_header_and_component(self):
        raw, metadata = toy_container()
        result, events = toy_auth(raw, metadata, ShortReader)
        self.assertEqual(result["x"]["sha256"], hashlib.sha256(raw[-6:]).hexdigest())
        self.assertEqual([x[0] for x in events], ["COMPONENT_CANDIDATE", "AUTHENTICATED_CONTAINER"])

    def test_different_parsed_header_is_rejected(self):
        raw, metadata = toy_container()
        metadata["header.json"] = metadata["header.json"].replace(b'"x"', b'"y"')
        with self.assertRaisesRegex(ValueError, "header differs"):
            toy_auth(raw, metadata)

    def test_wrong_seek_is_rejected_before_read(self):
        raw, metadata = toy_container()
        with self.assertRaisesRegex(ValueError, "seek"):
            toy_auth(raw, metadata, WrongSeek)

    def test_truncation_retains_partial_component_progress_without_success(self):
        raw, metadata = toy_container()
        events = []
        with patch.object(capture, "MODEL_BYTES", len(raw)), patch.object(
                capture, "MODEL_SHA", hashlib.sha256(raw).hexdigest()):
            with self.assertRaisesRegex(ValueError, "truncated"):
                capture.phase_one(io.BytesIO(raw[:-1]), metadata, {"x"},
                                  lambda key, value: events.append((key, value)))
        self.assertNotIn("AUTHENTICATED_CONTAINER", [x[0] for x in events])
        self.assertEqual(events[-1][0], "FAILED_UNAUTHENTICATED_COMPONENT")
        failure = events[-1][1]
        self.assertEqual(failure["consumed_bytes"], 5)
        self.assertEqual(failure["prefix_sha256"], hashlib.sha256(raw[-6:-1]).hexdigest())
        self.assertIs(failure["authenticated"], False)

    def test_trailing_bytes_reject_after_first_component_is_retained(self):
        raw, metadata = toy_container()
        events = []
        with patch.object(capture, "MODEL_BYTES", len(raw)), patch.object(
                capture, "MODEL_SHA", hashlib.sha256(raw).hexdigest()):
            with self.assertRaisesRegex(ValueError, "EOF"):
                capture.phase_one(io.BytesIO(raw + b"x"), metadata, {"x"},
                                  lambda key, value: events.append((key, value)))
        self.assertEqual([x[0] for x in events], ["COMPONENT_CANDIDATE"])

    def test_signed_zero_vector_is_preserved_by_independent_conversions(self):
        raw = bytes.fromhex("00800000803f00bf")
        source = dict(offset=0, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        actual = capture.resident_vector(io.BytesIO(raw), dict(name="x", kind="vector", count=4),
                                         {"x": source}, lambda *_: None)
        short = capture.resident_vector(ShortReader(raw), dict(name="x", kind="vector", count=4),
                                        {"x": source}, lambda *_: None)
        self.assertEqual(short, actual)
        expected = bytes.fromhex("00000080000000000000803f000000bf")
        self.assertEqual(actual["fp32_sha256"], hashlib.sha256(expected).hexdigest())
        self.assertEqual(actual["negative_zeros"], 1)
        original, fp32, guarded, zeros = replay.converted_vector(io.BytesIO(raw), 0, len(raw))
        self.assertEqual((original, fp32, guarded, zeros),
                         (source["sha256"], actual["fp32_sha256"], actual["guarded_sha256"], 1))

    def test_nonfinite_vectors_reject_without_complete_vector_record(self):
        for tail in (bytes.fromhex("807f"), bytes.fromhex("c07f"), bytes.fromhex("80ff")):
            raw = bytes.fromhex("0080803f") + tail
            events = []
            source = dict(offset=0, bytes=6, sha256=hashlib.sha256(raw).hexdigest())
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                capture.resident_vector(io.BytesIO(raw), dict(name="x", count=3), {"x": source},
                                        lambda *args: events.append(args))
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0][0], "FAILED_UNAUTHENTICATED_COMPONENT")
            self.assertEqual(events[0][1]["offending_bf16_bits"], int.from_bytes(tail, "little"))
            self.assertEqual(events[0][1]["offending_vector_index"], 2)
            self.assertEqual(events[0][1]["converted_values"], 2)
            self.assertEqual(events[0][1]["consumed_bytes"], 6)
            self.assertEqual(events[0][1]["prefix_sha256"], source["sha256"])

    def test_packed_odd_metadata_padding_and_guard_are_distinct(self):
        raw = bytes(range(96)) + b"\0" * 12
        components = {
            "x.weight": dict(offset=0, bytes=96, sha256=hashlib.sha256(raw[:96]).hexdigest()),
            "x.scales": dict(offset=96, bytes=6, sha256=hashlib.sha256(raw[96:102]).hexdigest()),
            "x.biases": dict(offset=102, bytes=6, sha256=hashlib.sha256(raw[102:]).hexdigest()),
        }
        value = capture.packed_chunks(io.BytesIO(raw), dict(name="x.weight", rows=3, columns=64),
                                      components, lambda *_: None)
        buffers = value["chunks"][0]["buffers"]
        for role in ("scales", "biases"):
            self.assertEqual(buffers[role]["padding_bytes"], 2)
            self.assertEqual(buffers[role]["guarded_bytes"], 36)
            self.assertEqual(buffers[role]["guarded_sha256"],
                             hashlib.sha256(b"\0" * 8 + bytes.fromhex("3100175a") * 7).hexdigest())
        self.assertEqual(replay.packed_buffer(io.BytesIO(raw), 96, 6), buffers["scales"])

    def test_changed_component_between_passes_is_not_published(self):
        raw = b"\0" * 36
        components = {
            "x.weight": dict(offset=0, bytes=32, sha256=hashlib.sha256(raw[:32]).hexdigest()),
            "x.scales": dict(offset=32, bytes=2, sha256=hashlib.sha256(raw[32:34]).hexdigest()),
            "x.biases": dict(offset=34, bytes=2, sha256=hashlib.sha256(raw[34:]).hexdigest()),
        }
        events = []
        changed = bytearray(raw)
        changed[0] = 1
        with self.assertRaisesRegex(ValueError, "source changed"):
            capture.packed_chunks(io.BytesIO(changed), dict(name="x.weight", rows=1, columns=64),
                                  components, lambda *args: events.append(args))
        self.assertEqual([event[0] for event in events],
                         ["EXPECTED_PACKED_BUFFER", "FAILED_UNAUTHENTICATED_COMPONENT"])

    def test_original_mid_component_read_error_survives_secondary_diagnostic_failure(self):
        raw, metadata = toy_container()
        offset = 8 + len(metadata["header.json"])
        original = OSError("original mid-component read failure")

        class FailingReader(io.BytesIO):
            def read(self, count=-1):
                if self.tell() == offset + 3:
                    raise original
                if self.tell() >= offset:
                    count = min(count, offset + 3 - self.tell())
                return super().read(count)

        events = []

        def emit(label, value):
            events.append((label, value))
            raise OSError("secondary failed-diagnostic writer")

        secondary = io.StringIO()
        with patch.object(capture, "MODEL_BYTES", len(raw)), patch.object(
                capture, "MODEL_SHA", hashlib.sha256(raw).hexdigest()), contextlib.redirect_stderr(secondary):
            with self.assertRaises(OSError) as raised:
                capture.phase_one(FailingReader(raw), metadata, {"x"}, emit)
        self.assertIs(raised.exception, original)
        self.assertEqual(events[0][1]["consumed_bytes"], 3)
        self.assertEqual(events[0][1]["prefix_sha256"], hashlib.sha256(raw[-6:-3]).hexdigest())
        self.assertIn("SECONDARY_DIAGNOSTIC_FAILURE", secondary.getvalue())

    def test_full_small_metadata_inventory_uses_no_model_download(self):
        metadata = capture.pinned_metadata(METADATA)
        logical = capture.logical_inventory()
        physical = capture.physical_inventory(capture.parse(metadata["header.json"]), logical)
        self.assertEqual((len(logical), len(physical)), (540, 1096))
        self.assertEqual(sum(x["kind"] == "matrix" for x in logical), 278)

    def test_duplicate_json_fields_are_rejected_by_both_readers(self):
        for parse in (capture.parse, replay.document):
            with self.assertRaises(ValueError):
                parse(b'{"scope":false,"scope":true}')

    def test_existing_output_prevents_source_work(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "existing.json"
            path.write_text("original")
            with patch.object(sys, "argv", ["capture", "unused", "unused", str(path)]), patch.object(
                    capture, "capture") as source:
                with self.assertRaisesRegex(ValueError, "existing output"):
                    capture.main()
                source.assert_not_called()
            self.assertEqual(path.read_text(), "original")

    def test_dangling_output_symlink_prevents_source_work(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "dangling.json"
            path.symlink_to(Path(folder) / "missing")
            with patch.object(sys, "argv", ["capture", "unused", "unused", str(path)]), patch.object(
                    capture, "capture") as source:
                with self.assertRaisesRegex(ValueError, "existing output"):
                    capture.main()
                source.assert_not_called()

    def test_partial_publication_retains_bytes_and_original_error_without_complete(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "partial.json"
            original_open = Path.open
            failure = OSError("injected original publication failure")

            class PartialWriter:
                def __init__(self):
                    self.file = original_open(path, "xb")

                def __enter__(self):
                    return self

                def write(self, raw):
                    self.file.write(raw[:8])
                    self.file.flush()
                    raise failure

                def __exit__(self, *_):
                    self.file.close()

            output = io.StringIO()
            with patch.object(sys, "argv", ["capture", "unused", "unused", str(path)]), patch.object(
                    capture, "capture", return_value={"synthetic_reader_control": True}), patch.object(
                    Path, "open", lambda *_args, **_kwargs: PartialWriter()), contextlib.redirect_stdout(output):
                with self.assertRaises(OSError) as raised:
                    capture.main()
            self.assertIs(raised.exception, failure)
            self.assertEqual(path.stat().st_size, 8)
            self.assertNotIn("COMPLETE", output.getvalue())


def real_suite(model, metadata, inventory):
    observed = replay.document(replay.bounded(inventory, 4 * 1024**2))
    expected = replay.expected_inventory(model, metadata)
    suite = unittest.TestSuite()
    suite.addTest(unittest.FunctionTestCase(lambda: replay.validate(observed, expected), description="complete baseline"))

    def mutation(label, change):
        def run():
            value = copy.deepcopy(observed)
            change(value)
            with unittest.TestCase().assertRaises((ValueError, KeyError, TypeError)):
                replay.validate(value, expected)
        suite.addTest(unittest.FunctionTestCase(run, description=label))

    def assign(path, value):
        def change(document):
            parent = document
            for key in path[:-1]:
                parent = parent[key]
            parent[path[-1]] = value
        return change

    first_name = next(iter(observed["source_components"]))
    for label, path, value in (
        ("coherent source anchor changed", ["model", "sha256"], "0" * 64),
        ("component source changed", ["source_components", first_name, "sha256"], "0" * 64),
        ("component offset changed", ["source_components", first_name, "offset"], 0),
        ("incomplete physical inventory", ["counts", "source_components"], 1095),
        ("missing chunk accounting", ["accounting", "chunk_descriptor_bytes"], 0),
        ("source scratch undercounted", ["accounting", "maximum_three_source_with_descriptors"], 0),
        ("invented bundle admission", ["complete_model_storage_admitted"], True),
        ("invented GPU execution", ["engine_or_gpu_executed"], True),
        ("invented graph capability", ["whole_graph_qualified"], True),
        ("boolean replaced by integer", ["reference_only"], 1),
        ("scope widened", ["scope"], "current filesystem and all modalities authenticated"),
        ("chunk range changed", ["matrices", 0, "chunks", 0, "first"], 1),
        ("guard hash changed", ["matrices", 0, "chunks", 0, "buffers", "codes", "guarded_sha256"], "0" * 64),
        ("vector conversion changed", ["vectors", 0, "fp32_sha256"], "0" * 64),
        ("vector signs changed", ["vectors", 0, "negative_zeros"], -1),
    ):
        mutation(label, assign(path, value))
    mutation("duplicate logical owner", lambda d: d["matrices"].append(copy.deepcopy(d["matrices"][0])))
    mutation("incomplete vector coverage", lambda d: d["vectors"].pop())

    def coherent(d):
        for matrix in d["matrices"]:
            for chunk in matrix["chunks"]:
                for buffer in chunk["buffers"].values():
                    buffer["source_sha256"] = buffer["guarded_sha256"] = "0" * 64
        for source in d["source_components"].values():
            source["sha256"] = "0" * 64
        for vector in d["vectors"]:
            vector["fp32_sha256"] = vector["guarded_sha256"] = "0" * 64
    mutation("all source and expected data views coherently replaced", coherent)
    return suite


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "portable":
        METADATA = Path(sys.argv[2])
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(Portable)
    elif len(sys.argv) == 5 and sys.argv[1] == "actual":
        METADATA = Path(sys.argv[4])
        suite = real_suite(Path(sys.argv[2]), METADATA, Path(sys.argv[3]))
    else:
        raise SystemExit("usage: test_gemma_resident_storage.py portable METADATA | actual MODEL INVENTORY METADATA")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
