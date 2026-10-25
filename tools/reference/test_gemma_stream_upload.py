"""Mutations of actual streaming-upload evidence; portable checks get no GPU credit."""
from pathlib import Path
import copy
import io
import sys
import unittest
from unittest import mock
import check_gemma_stream_upload as check


class ReaderFailures(unittest.TestCase):
    def test_duplicate_json(self):
        with self.assertRaisesRegex(ValueError, 'duplicate field'):
            check.records(b'GEMMA_STREAM_META {"device":"one","device":"two"}\n')

    def test_nonfinite_json(self):
        with self.assertRaisesRegex(ValueError, 'finite JSON required'):
            check.records(b'GEMMA_STREAM_META {"number":NaN}\n')

    def test_bounded_read_closes_stream(self):
        stream = io.BytesIO(b'12345')
        path = mock.Mock()
        path.open.return_value = stream
        with self.assertRaisesRegex(ValueError, 'bounded fixture'):
            check.base.reference.read(path, 4)
        self.assertTrue(stream.closed)

    def test_incomplete_test_wrapper(self):
        with self.assertRaisesRegex(ValueError, 'complete successful exact-test wrapper'):
            check.records(b'running 1 test\nGEMMA_STREAM_META {}\n')


class PreCapture(unittest.TestCase):
    def test_cold_budget_and_two_byte_bf16_tail(self):
        cases = check.base.fixture(Path(sys.argv[2]))
        self.assertEqual(check.cold(cases[0], 7)['owned_bytes'], 7864)
        state = check.cold(cases[1], 8)
        child = state['children'][-1]
        self.assertEqual(child['rows'], 3)
        for name in ('scales', 'biases'):
            words = child['device'][name]
            self.assertEqual(len(words), 9)
            self.assertEqual(words[1] >> 16, 0)
            self.assertEqual(words[2:], [check.base.SENTINEL]*7)


class CaptureMutations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        capture, frozen = map(Path, sys.argv[1:3])
        cls.rows = check.records(check.base.reference.read(capture, 32*1024**2))
        cls.cases = check.base.fixture(frozen)
        cls.baseline = check.Replay(copy.deepcopy(cls.rows)).run(cls.cases)

    def setUp(self):
        self.changed = copy.deepcopy(self.rows)

    def record(self, label, phase=None, index=None):
        return next(body for name, body in self.changed if name == label and (phase is None or body.get('phase') == phase) and (index is None or body.get('index') == index))

    def reject(self):
        with self.assertRaises(ValueError):
            check.Replay(self.changed).run(self.cases)

    def test_actual_complete_capture(self):
        self.assertIs(self.baseline['passed'], True)
        self.assertEqual((self.baseline['projection_steps'], self.baseline['successful_projection_values'], self.baseline['source_callback_calls']), (25,777,20))

    def test_working_admission(self):
        self.record('ATTEMPT')['minimum_working'] -= 72
        self.reject()

    def test_source_descriptors(self):
        self.record('META')['source_descriptors_bytes'] = 0
        self.reject()

    def test_source_active_slice(self):
        self.record('FILL_ATTEMPT')['before']['scales'].append(0)
        self.reject()

    def test_previous_source_prefix_retained(self):
        self.record('FILL_ATTEMPT', 'main', 1)['before']['codes'][0] ^= 1
        self.reject()

    def test_complete_source_byte(self):
        self.record('FILL_RESULT')['after']['codes'][0] ^= 1
        self.reject()

    def test_boolean_source_byte(self):
        self.record('FILL_RESULT')['after']['codes'][0] = True
        self.reject()

    def test_source_addresses_reused(self):
        self.record('FILL_ATTEMPT', 'main', 1)['addresses'][0] = '0x1'
        self.reject()

    def test_source_arrays_disjoint(self):
        pointers = self.record('FILL_ATTEMPT')['addresses']
        pointers[1] = pointers[0]
        self.reject()

    def test_distinct_but_overlapping_source_ranges(self):
        for label, body in self.changed:
            if label == 'FILL_ATTEMPT' and body['key'] == 'rows19-k128/max7' and body['phase'] == 'main':
                body['addresses'] = ['0x1000','0x1001','0x2000']
        self.reject()

    def test_source_range_wrap(self):
        for label, body in self.changed:
            if label == 'FILL_ATTEMPT' and body['key'] == 'rows19-k128/max7' and body['phase'] == 'main':
                body['addresses'] = ['0xffffffffffffffff','0x1000','0x2000']
        self.reject()

    def test_late_domain_bytes(self):
        self.record('FILL_RESULT', 'late-domain', 1)['after']['scales'][:2] = [0,0]
        self.reject()

    def test_budget_before_callback(self):
        self.record('OUTCOME', 'budget')['callback_calls'] = 1
        self.reject()

    def test_original_late_error(self):
        self.record('OUTCOME', 'late-source')['error'] = 'replacement error'
        self.reject()

    def test_original_panic(self):
        self.record('OUTCOME', 'panic')['panic'] = 'replacement panic'
        self.reject()

    def test_panic_warmed_owner(self):
        self.record('UNCHANGED_OWNER', 'panic')['observation']['children'][0]['device']['codes']['words'][0] ^= 1
        self.reject()

    def test_recovery_resident_alias(self):
        before = self.record('OWNER', 'main')['observation']['children'][0]['device']['codes']['allocation']
        self.record('OWNER', 'recover')['observation']['children'][0]['device']['codes']['allocation'] = before
        self.reject()

    def test_gpu_padded_metadata(self):
        owner = next(body for name, body in self.changed if name == 'OWNER' and body['key'] == 'rows11-k64/max8')
        owner['observation']['children'][-1]['device']['scales']['words'][1] ^= 1 << 16
        self.reject()

    def test_gpu_source_byte(self):
        self.record('OWNER')['observation']['children'][0]['device']['codes']['words'][0] ^= 1
        self.reject()

    def test_gpu_guard(self):
        self.record('STEP')['observation']['children'][0]['device']['output']['words'][-1] = 0
        self.reject()

    def test_gpu_status(self):
        self.record('STEP')['observation']['children'][0]['device']['status']['words'][0] = 1
        self.reject()

    def test_gpu_inactive_input(self):
        self.record('STEP', 'part')['observation']['children'][0]['device']['input']['words'][128] = 0
        self.reject()

    def test_coherent_wrong_projection(self):
        d = self.record('STEP')['observation']
        d['caller_bits'][0] ^= 1
        d['staged_bits'][0] ^= 1
        d['children'][0]['device']['output']['words'][0] ^= 1
        self.reject()

    def test_recovery_original_owner(self):
        self.record('UNCHANGED_OWNER', 'recover')['observation']['staged_bits'][0] ^= 1
        self.reject()

    def test_external_side_effects(self):
        self.record('FINAL')['callback_side_effects_rolled_back'] = True
        self.reject()

    def test_missing_callback_result(self):
        index = next(i for i, (name, _) in enumerate(self.changed) if name == 'FILL_RESULT')
        self.changed.pop(index)
        self.reject()


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit('actual native capture or --portable, and tiny frozen table fixture')
    suite, loader = unittest.TestSuite(), unittest.TestLoader()
    for kind in (ReaderFailures,PreCapture) if sys.argv[1] == '--portable' else (ReaderFailures,PreCapture,CaptureMutations):
        suite.addTests(loader.loadTestsFromTestCase(kind))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
