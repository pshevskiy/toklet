"""Independent literal stream/header and transfer controls; no network or model."""
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import minicpm5_parent as audit


def fixture():
    # Literal BF16 bytes: norm=[1,-2], matrix=[[.5,3],[0,-0]].
    payload = bytes.fromhex('803f00c0003f404000000080')
    header = json.dumps(dict(norm=dict(dtype='BF16', shape=[2], data_offsets=[0, 4]),
                             matrix=dict(dtype='BF16', shape=[2, 2], data_offsets=[4, 12]))).encode()
    prefix = len(header).to_bytes(8, 'little')
    raw = prefix + header + payload
    return prefix, header, raw


class ShortReader:
    def __init__(self, raw):
        self.stream = io.BytesIO(raw)
        self.requests = []

    def read(self, count):
        self.requests.append(count)
        return self.stream.read(min(count, 3))


class Streams(unittest.TestCase):
    def run_audit(self, raw=None, pin=None, norms=None):
        prefix, header, expected = fixture()
        journal = io.StringIO()
        return audit.authenticate(io.BytesIO(expected if raw is None else raw), prefix, header,
                                  len(expected), hashlib.sha256(expected).hexdigest() if pin is None else pin,
                                  {'norm': expected[-12:-8]} if norms is None else norms, journal)

    def test_complete_object_tensors_blocks_and_norms_have_independent_literal_hashes(self):
        prefix, header, raw = fixture()
        reader = ShortReader(raw)
        journal = io.StringIO()
        result = audit.authenticate(reader, prefix, header, len(raw), hashlib.sha256(raw).hexdigest(),
                                    {'norm': bytes.fromhex('803f00c0')}, journal)
        self.assertEqual(result['bytes'], len(raw))
        self.assertEqual(result['sha256'], hashlib.sha256(raw).hexdigest())
        self.assertEqual(result['exact_child_norms'], 1)
        expected = [('norm', bytes.fromhex('803f00c0')), ('matrix', bytes.fromhex('003f404000000080'))]
        for item, (name, data) in zip(result['tensors'], expected):
            self.assertEqual(item['name'], name)
            self.assertEqual(item['sha256'], hashlib.sha256(data).hexdigest())
            self.assertEqual(item['bytes'], len(data))
            self.assertEqual(item['blocks'][0]['sha256'], hashlib.sha256(data).hexdigest())
        self.assertTrue(all(0 < n <= audit.BLOCK for n in reader.requests))
        rows = [json.loads(line) for line in journal.getvalue().splitlines()]
        self.assertTrue(all(row['object_authenticated'] is False for row in rows))

    def test_block_boundaries_are_exact_absolute_offsets(self):
        prefix, header, raw = fixture()
        with patch.object(audit, 'BLOCK', len(header)):
            result = self.run_audit()
        self.assertEqual(result['tensors'][0]['blocks'][0]['offset'], len(prefix + header))
        self.assertEqual(result['tensors'][1]['blocks'][0]['offset'], len(prefix + header) + 4)
        # A larger synthetic tensor actually crosses multiple bounded reads.
        data = bytes(range(256))
        h = json.dumps(dict(w=dict(dtype='BF16', shape=[128], data_offsets=[0, 256]))).encode()
        p = len(h).to_bytes(8, 'little');blob = p + h + data
        with patch.object(audit, 'BLOCK', len(h)):
            out = audit.authenticate(io.BytesIO(blob), p, h, len(blob), hashlib.sha256(blob).hexdigest(), {}, io.StringIO())
        blocks = out['tensors'][0]['blocks']
        self.assertGreater(len(blocks), 1)
        self.assertEqual(sum(b['bytes'] for b in blocks), 256)
        self.assertEqual([b['offset'] for b in blocks],
                         [len(p + h) + i for i in range(0, 256, len(h))])

    def test_truncation_trailing_mutation_and_wrong_hash_fail_whole_identity(self):
        _, _, raw = fixture()
        for data in [raw[:-1], raw + b'x', raw[:-1] + b'\1', b'\0' + raw[1:]]:
            with self.assertRaises(ValueError):
                self.run_audit(raw=data)
        with self.assertRaisesRegex(ValueError, 'SHA-256'):
            self.run_audit(pin='0' * 64)

    def test_norms_are_exact_bytes_including_negative_zero_and_unknown_controls_reject(self):
        for norms in [{'norm': b'\0' * 4}, {'norm': b'\0'}, {'missing': b'x'}, {'norm': b'x' * 65537}]:
            with self.assertRaises(ValueError):
                self.run_audit(norms=norms)

    def test_plan_rejects_holes_overlaps_sizes_types_and_unexpected_dtype(self):
        _, header, raw = fixture()
        original = json.loads(header)
        changes = [(['norm', 'dtype'], 'F16'), (['norm', 'shape'], [True]),
                   (['norm', 'shape'], [0]), (['norm', 'shape'], [131073]),
                   (['norm', 'data_offsets'], [0, 3]), (['matrix', 'data_offsets'], [5, 13]),
                   (['matrix', 'data_offsets'], [2, 10]), (['norm', 'extra'], 1)]
        for path, value in changes:
            changed = copy.deepcopy(original);changed[path[0]][path[1]] = value
            h = json.dumps(changed).encode();p = len(h).to_bytes(8, 'little')
            with self.assertRaises(ValueError):
                audit.plan(p, h, len(p + h) + 12)
        with self.assertRaises(ValueError):
            audit.plan(b'\0' * 8, header, len(raw))
        with self.assertRaises(ValueError):
            audit.plan(len(header).to_bytes(8, 'little'), header, True)

    def test_duplicate_header_fields_and_unbounded_reads_reject(self):
        h = b'{"w":{},"w":{}}'
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            audit.plan(len(h).to_bytes(8, 'little'), h, len(h) + 10)
        for count in [-1, True, audit.BLOCK + 1]:
            with self.assertRaises(ValueError):
                audit.exact(io.BytesIO(b''), count)


class Transfers(unittest.TestCase):
    def run_transfer(self, transport=None, raw=None, code=0):
        prefix, header, expected = fixture()
        status = dict(http_code=200, size_download=len(expected), ssl_verify_result=0)
        status.update({} if transport is None else transport)
        fake = SimpleNamespace(stdout=io.BytesIO(expected if raw is None else raw), pid=123, returncode=None)
        fake.poll = lambda: None
        fake.terminate = lambda: None
        fake.kill = lambda: None
        def launch(command, stdout, stderr):
            stderr.write('TOKLET_CURL_JSON' + json.dumps(status) + '\n');stderr.flush()
            return fake
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(audit.subprocess, 'Popen', side_effect=launch), \
             patch.object(audit.os, 'wait4', return_value=(123, code << 8, SimpleNamespace(ru_maxrss=4096))):
            target = Path(folder)
            shard = ('fixture', len(expected), hashlib.sha256(expected).hexdigest(), prefix, header,
                     {'norm': bytes.fromhex('803f00c0')})
            error = None
            try:
                audit.transfer(shard, target)
            except ValueError as e:
                error = e
            record = json.loads((target / 'fixture.json').read_text())
            return record, error

    def test_success_requires_whole_bytes_and_exact_closed_transfer_receipt(self):
        result, error = self.run_transfer()
        self.assertIsNone(error)
        self.assertTrue(result['passed'])
        self.assertEqual(result['final_wait4_peak_rss_bytes'], 4096)
        self.assertFalse(result['payload_retained'])

    def test_resealed_transfer_errors_never_authenticate_or_hide_primary_log(self):
        for status in [dict(http_code=206), dict(size_download=1), dict(ssl_verify_result=1),
                       dict(ssl_verify_result=False), dict(http_code=200.)]:
            result, error = self.run_transfer(transport=status)
            self.assertIsNotNone(error)
            self.assertFalse(result['passed'])
        for kwargs in [dict(raw=b'bad'), dict(code=22)]:
            result, error = self.run_transfer(**kwargs)
            self.assertIsNotNone(error)
            self.assertFalse(result['passed'])


if __name__ == '__main__':
    unittest.main()
