import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import q4


def inspect(data):
    """Independent unpacking used only in these tests, with scalar FP32 storage."""
    magic, version, group, rows, columns, stride, payload = struct.unpack("<8sIIQQQQ16x", data[:64])
    assert (magic, version, group, payload) == (b"TOKQ4V1\0", 1, 64, rows * stride)
    output = []
    for row in range(rows):
        values = []
        for column in range(columns):
            offset = 64 + row * stride + column // 64 * 36
            scale, bias = struct.unpack_from("<ee", data, offset)
            code = data[offset + 4 + column % 64 // 2] >> (4 * (column % 2)) & 15
            values.append(struct.unpack("<f", struct.pack("<f", scale * code + bias))[0])
        output.append(values)
    return output


class Format(unittest.TestCase):
    def test_hand_calculated_nibbles_constant_tail_and_alignment(self):
        rows = [[-2 + .5 * (i % 16) for i in range(64)] + [3.25], [0.0] * 65]
        data = q4.pack_matrix(rows, 65)
        self.assertEqual(len(data), 320)
        self.assertEqual(data[64:68], struct.pack("<ee", .5, -2))
        self.assertEqual(data[68:100], bytes([0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe]) * 4)
        self.assertEqual(data[100:104], struct.pack("<ee", 0, 3.25))
        self.assertEqual(data[104:192], bytes(88))
        self.assertEqual(inspect(data), rows)

    def test_constant_signed_zero_half_rounding_and_subnormal(self):
        for value in [0, -0.0, 1.00048828125, 65504, -65504, 2**-24]:
            data = q4.pack_matrix([[value] * 64], 64)
            self.assertEqual(data[64:66], bytes(2))
            self.assertEqual(data[68:100], bytes(32))
            self.assertEqual(inspect(data)[0], [q4.half(value)] * 64)
        data = q4.pack_matrix([[0, 2**-25]], 2)
        self.assertEqual(data[64:66], b"\x01\x00")
        self.assertEqual(data[68], 0)  # exact halfway code rounds to even zero

    def test_codes_round_ties_even_after_metadata_rounding(self):
        data = q4.pack_matrix([[0, .5, 1.5, 2.5, 14.5, 15]], 6)
        self.assertEqual(data[68:71], bytes([0x00, 0x22, 0xfe]))
        self.assertEqual(inspect(data), [[0, 0, 2, 2, 14, 15]])

    def test_extreme_finite_range_and_rejected_source_metadata(self):
        self.assertEqual(inspect(q4.pack_matrix([[-65504, 65504]], 2)), [[-65504, 65536]])
        for row in [[float("nan")], [float("inf")], [float("-inf")], [70000], [-70000, 0], [1e40]]:
            with self.assertRaises(ValueError): q4.pack_matrix([row], len(row))
        for rows, columns in [(0, 1), (1, 0), (True, 1), (1, 2**64), (2**64, 1)]:
            with self.assertRaises(ValueError): q4.header(rows, columns)
        with self.assertRaises(ValueError): q4.pack_matrix([[1]], 2)
        with self.assertRaises(ValueError): q4.pack_matrix([[1]], 1, limit=127)

    def test_frozen_artifact_exact_bytes_and_reconstruction(self):
        directory = Path(__file__).resolve().parents[2] / "tests/fixtures/q4"
        manifest = json.loads((directory / "manifest.json").read_text())
        rows = manifest["source_rows"]
        data = (directory / "rows.q4").read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(), manifest["packed_sha256"])
        self.assertEqual(q4.pack_matrix(rows, 65), data)
        self.assertEqual(inspect(data), manifest["reconstructed_rows"])

    def test_streaming_identity_existing_output_and_failure_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            src = Path(directory) / "input"; dst = Path(directory) / "output"
            raw = struct.pack("<4f", -2, 5.5, 3.25, 3.25)
            src.write_bytes(raw)
            result = q4.pack_file(src, dst, 2, 2)
            self.assertEqual(result["source_sha256"], hashlib.sha256(raw).hexdigest())
            self.assertEqual(result["packed_sha256"], hashlib.sha256(dst.read_bytes()).hexdigest())
            saved = dst.read_bytes()
            with self.assertRaises(FileExistsError): q4.pack_file(src, dst, 2, 2)
            self.assertEqual(dst.read_bytes(), saved)
            dst.unlink(); src.write_bytes(struct.pack("<f", float("nan")))
            with self.assertRaises(ValueError): q4.pack_file(src, dst, 1, 1)
            self.assertFalse(dst.exists())

    def test_final_flush_failure_removes_only_owned_output(self):
        with tempfile.TemporaryDirectory() as directory:
            src = Path(directory) / "input"; dst = Path(directory) / "output"
            src.write_bytes(struct.pack("<f", 1))
            real_open = Path.open
            class FailingExit:
                def __init__(self, file): self.file = file
                def __enter__(self): return self.file
                def __exit__(self, *args):
                    self.file.close()
                    raise OSError("final flush failed")
            def opened(path, *args, **kwargs):
                file = real_open(path, *args, **kwargs)
                return FailingExit(file) if path == dst else file
            with patch.object(Path, "open", opened):
                with self.assertRaisesRegex(OSError, "final flush failed"):
                    q4.pack_file(src, dst, 1, 1)
            self.assertFalse(dst.exists())
            with self.assertRaises(ValueError): q4.pack_file(src, dst, 2, 1)
            self.assertFalse(dst.exists())


if __name__ == "__main__": unittest.main()
