import hashlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

import numpy as np
import qwen_storage as storage


class StorageTests(unittest.TestCase):
    def test_exact_returned_bytes_are_hashed_before_parse(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            before, after = b'{"old":true}', b'{"new":true}'
            path.write_bytes(before)
            class ReplacedAfterRead(io.BytesIO):
                def __exit__(self, *args):
                    with open(path, "wb") as stream: stream.write(after)
                    return super().__exit__(*args)
            expected = hashlib.sha256(before).hexdigest()
            with mock.patch.object(Path, "open", return_value=ReplacedAfterRead(before)):
                data, identity = storage.bounded_bytes(path, 100, expected)
            self.assertEqual((data, identity), (before, expected))
            self.assertNotEqual(storage.sha(path), identity)
            with mock.patch.object(Path, "open", return_value=ReplacedAfterRead(before)):
                with self.assertRaisesRegex(ValueError, "identity"):
                    storage.bounded_bytes(path, 100, hashlib.sha256(after).hexdigest())

    def test_read_bound_and_payload_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matrix.q4"
            path.write_bytes(b"12345")
            with self.assertRaisesRegex(ValueError, "bound"):
                storage.bounded_bytes(path, 4)
            with self.assertRaisesRegex(ValueError, "identity"):
                storage.bounded_bytes(path, 5, "0" * 64)

    def test_config_endpoint_is_rechecked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "model.safetensors"
            source.write_bytes(b"source")
            config = root / "config.json"
            config.write_bytes(b"{}")
            manifest = root / "manifest.json"
            manifest.write_bytes(b"{}")
            reader = storage.Weights.__new__(storage.Weights)
            reader.path = source
            reader.expected_sha = storage.sha(source)
            reader.packed = root
            reader.config_sha = storage.sha(config)
            reader.manifest_sha = storage.sha(manifest)
            reader.entries = {}
            with source.open("rb") as reader.stream:
                reader.verify()
                config.write_bytes(b'{"changed":true}')
                with self.assertRaisesRegex(ValueError, "config endpoint"):
                    reader.verify()

    def test_open_descriptor_endpoint_survives_path_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.bin"
            source.write_bytes(b"admitted")
            reader = storage.Weights.__new__(storage.Weights)
            reader.path, reader.expected_sha, reader.packed = source, storage.sha(source), None
            with source.open("rb") as reader.stream:
                old = source.with_name("old.bin")
                source.rename(old)
                source.write_bytes(b"admitted")
                old.write_bytes(b"mutated!")
                with self.assertRaisesRegex(ValueError, "descriptor endpoint"):
                    reader.verify()

    def test_q4_oracle_and_canonical_padding(self):
        rows, columns, stride = 2, 65, 128
        data = bytearray(struct.pack("<8sIIQQQQ16x", b"TOKQ4V1\0", 1, 64, rows, columns, stride, rows*stride))
        expected = []
        for row in range(rows):
            payload = bytearray(stride)
            struct.pack_into("<ee", payload, 0, .125, -.5)
            for index in range(32):
                payload[4+index] = (2*index%16) | ((2*index+1)%16 << 4)
            struct.pack_into("<ee", payload, 36, .25, .25)
            payload[40] = 15
            data.extend(payload)
            expected.append([.125*(column%16)-.5 for column in range(64)] + [4])
        np.testing.assert_array_equal(storage.decode_q4(data, rows, columns), expected)
        for index, value in ((64+41, 1), (64+72, 1), (64+3, 0x7e)):
            altered = bytearray(data)
            altered[index] = value
            with self.assertRaises(ValueError):
                storage.decode_q4(altered, rows, columns)

    def test_source_stripes_and_tied_table_ownership(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors"
            words = np.asarray([0x3F80, 0xBF80, 0x0000, 0x3E80] * 1025, dtype="<u2")
            header = json.dumps({"model.embed_tokens.weight": {
                "dtype": "BF16", "shape": [1025, 4], "data_offsets": [0, words.nbytes]
            }}).encode()
            path.write_bytes(struct.pack("<Q", len(header)) + header + words.tobytes())
            with storage.Weights(path, storage.sha(path)) as reader:
                actual = reader["embed_tokens"]
                self.assertIs(actual, reader["embed_tokens"])
                self.assertEqual(actual.dtype, np.float16)
                np.testing.assert_array_equal(actual, np.tile([1, -1, 0, .25], (1025, 1)))
                reader.verify()
            self.assertEqual(reader.cached, {})


if __name__ == "__main__":
    unittest.main()
