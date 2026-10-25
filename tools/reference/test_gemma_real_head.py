"""Post-capture mutations of complete real-head results; no synthetic GPU credit."""
from pathlib import Path
import copy
import io
import struct
import sys
import unittest
from unittest import mock
import check_gemma_real_head as check


class ReaderFailures(unittest.TestCase):
    def test_duplicate_json(self):
        with self.assertRaisesRegex(ValueError, 'duplicate JSON field'):
            check.parse(b'GEMMA_HEAD_PLAN {"a":1,"a":2}\n')

    def test_nonfinite_json(self):
        with self.assertRaisesRegex(ValueError, 'finite JSON required'):
            check.parse(b'GEMMA_HEAD_PLAN {"a":NaN}\n')

    def test_bounded_model_hash(self):
        path = mock.Mock()
        stream = io.BytesIO(b'12345')
        path.open.return_value = stream
        with self.assertRaisesRegex(ValueError, 'exceeds exact byte bound'):
            check.base.digest(path, 4)
        self.assertTrue(stream.closed)

    def test_incomplete_test_wrapper(self):
        with self.assertRaisesRegex(ValueError, 'complete exact successful test'):
            check.parse(b'running 1 test\nGEMMA_HEAD_PLAN {}\n')


class CaptureMutations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        stdout, cls.binary, model, frozen = map(Path, sys.argv[1:5])
        with stdout.open('rb') as source:
            cls.rows = check.parse(source.read(32 * 1024**2 + 1))
        cls.assets = check.Assets(model, frozen)
        try:
            cls.baseline = check.verify(cls.rows, cls.binary, cls.assets)
        except BaseException:
            cls.assets.close()
            raise

    @classmethod
    def tearDownClass(cls):
        cls.assets.close()

    def setUp(self):
        self.changed = copy.deepcopy(self.rows)
        self.overrides = {}

    def record(self, label, key=None):
        return next(body for name, body in self.changed if name == label and (key is None or body.get('key') == key))

    def reject(self):
        with self.assertRaises(ValueError):
            check.verify(self.changed, self.binary, self.assets, overrides=self.overrides)

    def replace(self, key, role, position, bits):
        entries = [body for label, body in self.changed if label == 'ARRAY' and body['key'] == key and body['role'] == role]
        self.assertEqual(len(entries), 1)
        entry = entries[0]
        with self.binary.open('rb') as source:
            source.seek(entry['offset'])
            raw = bytearray(source.read(entry['bytes']))
        if entry['offset'] in self.overrides:
            raw = bytearray(self.overrides[entry['offset']])
        raw[position*4:position*4+4] = struct.pack('<I', bits)
        raw = bytes(raw)
        self.overrides[entry['offset']] = raw
        digest = check.sha(raw)
        for label, body in self.changed:
            if label in ('ARRAY', 'ARRAY_ATTEMPT') and body['offset'] == entry['offset']:
                body['sha256'] = digest
        step = self.record('STEP', key)
        if role == 'transactional-caller':
            step['caller']['sha256'] = digest
            self.record('STEP_OUTCOME', key)['caller_sha256'] = digest
        else:
            child = int(role.split('-')[1])
            step['observed_child_outputs'][child]['sha256'] = digest
            device = self.record('STEP_OUTCOME', key)['observation']['children'][child]['device']['output']
            tokens = step['tokens']
            full = raw + check.SENTINEL*((4-tokens)*check.TILE) + check.GUARDS
            device.update(check.identity(full))

    def test_actual_complete_capture(self):
        self.assertIs(self.baseline['passed'], True)
        self.assertEqual((self.baseline['steps'], self.baseline['arrays']), (7, 123))

    def test_boolean_buffer_word_count(self):
        self.record('CONSTRUCTOR')['observation']['children'][0]['device']['codes']['words'] = True
        self.reject()

    def test_aggregate_policy_excludes_source(self):
        self.record('META')['policy']['categories']['source_cpu_loading_overlap'] = 0
        self.reject()

    def test_original_source_span(self):
        self.record('META')['source_spans'][0]['offset'] += 4
        self.reject()

    def test_original_resident_weight_digest(self):
        self.record('CONSTRUCTOR')['observation']['children'][0]['device']['codes']['sha256'] = '0'*64
        self.reject()

    def test_second_upload_identity(self):
        self.record('STEP_OUTCOME')['observation']['children'][0]['device']['codes']['allocation'] = '0x1'
        self.reject()

    def test_live_allocation_alias(self):
        device = self.record('CONSTRUCTOR')['observation']['children'][0]['device']
        device['output']['allocation'] = device['input']['allocation']
        self.reject()

    def test_device_guard(self):
        self.record('STEP_OUTCOME')['observation']['children'][0]['device']['output']['guards'][0] = 0
        self.reject()

    def test_status_digest(self):
        self.record('STEP_OUTCOME')['observation']['children'][0]['device']['status']['sha256'] = '0'*64
        self.reject()

    def test_inactive_input_capacity(self):
        self.record('STEP_OUTCOME', 'part/0')['observation']['children'][0]['device']['input']['sha256'] = '0'*64
        self.reject()

    def test_token_strided_parent_staging(self):
        self.record('STEP_OUTCOME')['observation']['staged_sha256'] = '0'*64
        self.reject()

    def test_reused_chunk_scratch(self):
        self.record('STEP_OUTCOME', 'part/0')['observation']['chunk_scratch_sha256'] = '0'*64
        self.reject()

    def test_coherent_wrong_math_beyond_gate(self):
        # Reconstruct all captured views coherently so only the independent
        # mathematical gate can reject the new full-call first output.
        key = 'repeat/0'
        self.replace(key, 'child-0-observed-output', 0, 0x447a0000)  # 1000
        self.replace(key, 'transactional-caller', 0, 0x447a0000)
        caller = self.record('STEP', key)['caller']
        outcome = self.record('STEP_OUTCOME', key)
        raw = self.overrides[caller['offset']]
        outcome['observation']['staged_sha256'] = check.sha(raw)
        with self.assertRaisesRegex(ValueError, 'GPU rounded mathematical numerical gate failed'):
            check.verify(self.changed, self.binary, self.assets, overrides=self.overrides)

    def test_completed_child_array_changed_without_caller(self):
        self.replace('repeat/0', 'child-0-observed-output', 0, 0x447a0000)
        self.reject()

    def drift(self, key):
        caller = self.record('STEP', key)['caller']
        with self.binary.open('rb') as source:
            source.seek(caller['offset'])
            bits = struct.unpack('<I', source.read(4))[0]
        adjacent = bits+1 if bits < 0x7f7fffff or 0x80000000 <= bits < 0xff7fffff else bits-1
        self.replace(key, 'child-0-observed-output', 0, adjacent)
        self.replace(key, 'transactional-caller', 0, adjacent)
        raw = self.overrides[caller['offset']]
        if self.record('STEP', key)['tokens'] == 1:
            first = self.record('STEP', 'repeat/0')['caller']
            with self.binary.open('rb') as source:
                source.seek(first['offset']+check.N*4)
                raw += source.read(3*check.N*4)
        self.record('STEP_OUTCOME', key)['observation']['staged_sha256'] = check.sha(raw)

    def test_coherent_repeat_drift_within_gate(self):
        self.drift('repeat/1')
        with self.assertRaisesRegex(ValueError, 'repeat or resume bits differ'):
            check.verify(self.changed, self.binary, self.assets, overrides=self.overrides)

    def test_coherent_partition_drift_within_gate(self):
        self.drift('part/1')
        with self.assertRaisesRegex(ValueError, 'single-token partition bits differ'):
            check.verify(self.changed, self.binary, self.assets, overrides=self.overrides)

    def test_missing_original_outcome(self):
        index = next(i for i, (label, _) in enumerate(self.changed) if label == 'STEP_OUTCOME')
        self.changed.pop(index)
        self.reject()

    def test_inflight_buffer_read_claim(self):
        self.record('STEP_OUTCOME')['observation']['children'][0]['shared_read_safe'] = False
        self.reject()

    def test_retained_source_claim(self):
        self.record('CONSTRUCTOR')['source_cpu_bytes_retained'] = 226492416
        self.reject()

    def test_raw_array_boolean_byte_count(self):
        self.record('ARRAY')['bytes'] = True
        self.reject()


if __name__ == '__main__':
    if len(sys.argv) != 5:
        raise SystemExit('native stdout, raw outputs, original checkpoint, frozen head directory')
    unittest.main(argv=[sys.argv[0]], verbosity=2)
