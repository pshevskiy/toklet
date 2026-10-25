"""Mutations of the actual complete GPU capture, with no synthetic GPU credit."""
from pathlib import Path
import copy
import io
import sys
import unittest
from unittest import mock
import check_gemma_partitioned as check


class ReaderFailures(unittest.TestCase):
    def test_duplicate_json(self):
        with self.assertRaisesRegex(ValueError, 'duplicate field'):
            check.records(b'GEMMA_PARTITION_META {"device":"one","device":"two"}\n')

    def test_nonfinite_json(self):
        with self.assertRaisesRegex(ValueError, 'finite JSON required'):
            check.records(b'GEMMA_PARTITION_META {"number":NaN}\n')

    def test_bounded_read_closes_stream(self):
        stream = io.BytesIO(b'12345')
        path = mock.Mock()
        path.open.return_value = stream
        with self.assertRaisesRegex(ValueError, 'bounded fixture'):
            check.reference.read(path, 4)
        self.assertTrue(stream.closed)

    def test_incomplete_test_wrapper(self):
        with self.assertRaisesRegex(ValueError, 'complete successful exact-test wrapper'):
            check.records(b'running 1 test\nGEMMA_PARTITION_META {}\n')


class CaptureMutations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        capture, frozen = map(Path, sys.argv[1:3])
        cls.rows = check.records(check.reference.read(capture, 32 * 1024**2))
        cls.cases = check.fixture(frozen)
        cls.baseline = check.Replay(copy.deepcopy(cls.rows)).run(cls.cases)

    def setUp(self):
        self.changed = copy.deepcopy(self.rows)

    def record(self, label, phase=None):
        return next(body for name, body in self.changed if name == label and (phase is None or body.get('phase') == phase))

    def reject(self):
        with self.assertRaises(ValueError):
            check.Replay(self.changed).run(self.cases)

    def test_actual_complete_capture(self):
        self.assertIs(self.baseline['passed'], True)
        self.assertEqual((self.baseline['owners'], self.baseline['steps']), (10, 60))

    def test_boolean_word(self):
        self.record('CONSTRUCTOR')['observation']['children'][0]['device']['codes']['words'][0] = True
        self.reject()

    def test_original_packed_byte(self):
        self.record('CONSTRUCTOR')['observation']['children'][0]['device']['codes']['words'][0] ^= 1
        self.reject()

    def test_metadata_padding(self):
        # First cold owner has row1/K128 (4 metadata bytes); find K64/max1
        # so the second BF16 lane is padding rather than another source value.
        entry = next(body for label, body in self.changed if label == 'CONSTRUCTOR' and body['key'] == 'rows11-k64/max1')
        entry['observation']['children'][0]['device']['scales']['words'][0] ^= 1 << 16
        self.reject()

    def test_device_guard(self):
        self.record('STEP')['observation']['children'][0]['device']['output']['words'][-1] = 0
        self.reject()

    def test_status(self):
        self.record('STEP')['observation']['children'][0]['device']['status']['words'][0] = 1
        self.reject()

    def test_second_upload_identity(self):
        self.record('STEP')['observation']['children'][0]['device']['codes']['allocation'] = '0x1'
        self.reject()

    def test_live_alias(self):
        entry = self.record('CONSTRUCTOR')['observation']['children'][0]['device']
        entry['output']['allocation'] = entry['input']['allocation']
        self.reject()

    def test_full_staging_token_stride(self):
        bits = self.record('STEP')['observation']['staged_bits']
        bits[19], bits[20] = bits[20], bits[19]
        self.reject()

    def test_inactive_device_capacity(self):
        self.record('STEP', 'part')['observation']['children'][0]['device']['input']['words'][128] = 0
        self.reject()

    def test_coherent_wrong_result_all_views(self):
        for label, body in self.changed:
            if label != 'STEP' or body['key'] != 'rows19-k128/max1' or body['phase'] not in ('repeat', 'resume'):
                continue
            observation = body['observation']
            observation['caller_bits'][20] ^= 1
            observation['staged_bits'][20] ^= 1
            observation['children'][1]['device']['output']['words'][1] ^= 1
        self.reject()

    def test_failure_caller_is_not_published(self):
        self.record('STEP', 'composition-failure')['observation']['caller_bits'][0] = 0
        self.reject()

    def test_failure_retains_actual_partial_stage(self):
        before = self.record('STEP', 'resume')['observation']['staged_bits']
        self.record('STEP', 'composition-failure')['observation']['staged_bits'] = copy.deepcopy(before)
        self.reject()

    def test_sticky_poison(self):
        for label in ('STEP', 'STEP_OUTCOME'):
            entry = self.record(label, 'poison-retry')
            (entry['observation'] if label == 'STEP' else entry)['poisoned'] = False
        self.reject()

    def test_later_preflight_changes_nothing(self):
        self.record('STEP', 'reject')['observation']['children'][0]['device']['input']['words'][0] = 1
        self.reject()

    def test_recreated_geometry(self):
        entry = next(body for label, body in self.changed if label == 'CONSTRUCTOR_ATTEMPT' and body['key'].endswith('/recreated'))
        entry['maximum_rows'] = 7
        self.reject()

    def test_aggregate_budget_double_count(self):
        self.record('CONSTRUCTOR')['observation']['owned_bytes'] += 240
        self.reject()

    def test_missing_completed_record(self):
        index = next(i for i, (label, _) in enumerate(self.changed) if label == 'STEP_OUTCOME')
        self.changed.pop(index)
        self.reject()


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit('actual native capture and frozen table fixture directory')
    unittest.main(argv=[sys.argv[0]], verbosity=2)
