"""Mutate the actual full resident gather capture, never synthetic GPU credit."""
from pathlib import Path
import copy
import io
import sys
import unittest
from unittest import mock
import check_gemma_gather as check


class ReaderFailures(unittest.TestCase):
    def test_duplicate_json(self):
        with self.assertRaisesRegex(ValueError, 'duplicate field'):
            check.records(b'GEMMA_GATHER_META {"device":"one","device":"two"}\n')

    def test_nonfinite_json(self):
        with self.assertRaisesRegex(ValueError, 'finite JSON required'):
            check.records(b'GEMMA_GATHER_META {"number":NaN}\n')

    def test_bounded_read_closes_stream(self):
        stream = io.BytesIO(b'12345')
        path = mock.Mock()
        path.open.return_value = stream
        with self.assertRaisesRegex(ValueError, 'bounded fixture'):
            check.reference.read(path, 4)
        self.assertTrue(stream.closed)

    def test_incomplete_test_wrapper(self):
        with self.assertRaisesRegex(ValueError, 'complete successful exact-test wrapper'):
            check.records(b'running 1 test\nGEMMA_GATHER_META {}\n')


class PreCapture(unittest.TestCase):
    def test_first_affine_evolves_cold_words_before_learning_identities(self):
        # Portable protocol check only: no device capture or GPU credit. The
        # mathematical output is taken from the frozen independent fixture.
        case = check.base.fixture(Path(sys.argv[2]))[0]
        table = check.matrix(case, 7)
        replay = check.Replay([])
        n, k = case['rows'], case['columns']
        labels = []
        def take(label, expected=None):
            labels.append(label)
            if label != 'AFFINE':
                return expected
            observed = copy.deepcopy(table)
            counter = 1
            for child in observed['children']:
                for name, words in list(child['device'].items()):
                    self.assertIs(type(words), list)
                    child['device'][name] = dict(allocation=hex(counter), words=words)
                    counter += 1
            return dict(key='portable-only', index=0, input_bits=case['input'][k:2*k],
                        output_bits=case['expected'][n:2*n], matrix=observed)
        replay.take = take
        replay.affine(table, case, 'portable-only', 0)
        self.assertEqual(labels, ['AFFINE_ATTEMPT', 'AFFINE_OUTCOME', 'AFFINE'])
        self.assertEqual(table['staged_bits'], case['expected'][n:2*n])
        self.assertEqual(len(replay.live), 18)


class CaptureMutations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        capture, frozen = map(Path, sys.argv[1:3])
        cls.rows = check.records(check.reference.read(capture, 32*1024**2))
        cls.cases = check.base.fixture(frozen)
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
        self.assertEqual((self.baseline['steps'], self.baseline['affine_calls']), (52, 8))

    def test_boolean_word(self):
        self.record('STEP')['observation']['workspace']['ids']['words'][0] = True
        self.reject()

    def test_original_code_byte(self):
        self.record('AFFINE')['matrix']['children'][0]['device']['codes']['words'][0] ^= 1
        self.reject()

    def test_metadata_padding(self):
        entry = next(body for name, body in self.changed if name == 'AFFINE' and body['key'] == 'rows11-k64/max7')
        entry['matrix']['children'][0]['device']['scales']['words'][3] ^= 1 << 16
        self.reject()

    def test_shared_weight_identity(self):
        self.record('STEP')['observation']['matrix']['children'][0]['device']['codes']['allocation'] = '0x1'
        self.reject()

    def test_live_workspace_alias(self):
        d = self.record('CONSTRUCTOR')['observation']
        d['workspace']['ids']['allocation'] = d['matrix']['children'][0]['device']['codes']['allocation']
        self.reject()

    def test_recreated_workspace_overlap(self):
        old = self.record('CONSTRUCTOR')['observation']['workspace']['ids']['allocation']
        self.record('RECREATED')['observation']['workspace']['ids']['allocation'] = old
        self.reject()

    def test_wrong_local_id(self):
        self.record('STEP')['observation']['workspace']['ids']['words'][0] ^= 1
        self.reject()

    def test_device_guard(self):
        self.record('STEP')['observation']['workspace']['output']['words'][-1] = 0
        self.reject()

    def test_inactive_capacity(self):
        self.record('STEP')['observation']['workspace']['output']['words'][128*5] = 0
        self.reject()

    def test_status(self):
        self.record('STEP')['observation']['workspace']['status']['words'][0] = 1
        self.reject()

    def test_full_stage_stride(self):
        stage = self.record('STEP')['observation']['staged_bits']
        stage[128:256], stage[256:384] = stage[256:384], stage[128:256]
        self.reject()

    def test_negative_zero_sign(self):
        stage = self.record('STEP')['observation']['staged_bits']
        index = stage.index(0x80000000)
        stage[index] = 0
        self.reject()

    def test_coherent_bad_lookup(self):
        d = self.record('STEP')['observation']
        d['staged_bits'][0] ^= 1
        d['caller_bits'][0] ^= 1
        self.reject()

    def test_affine_output(self):
        self.record('AFFINE')['output_bits'][0] ^= 1
        self.reject()

    def test_affine_scratch_mutated_by_gather(self):
        self.record('STEP')['observation']['matrix']['chunk_bits'][0] ^= 1
        self.reject()

    def test_rejection_caller(self):
        self.record('STEP', 'matrix-reject')['observation']['caller_bits'][0] = 0
        self.reject()

    def test_rejection_workspace(self):
        self.record('STEP', 'request-reject')['observation']['workspace']['ids']['words'][0] ^= 1
        self.reject()

    def test_pending_device_read(self):
        entry = next(body for name, body in self.changed if name == 'STEP' and body['phase'] == 'matrix-reject' and body['index'] == 4)
        entry['observation']['matrix']['children'][0]['device'] = {}
        self.reject()

    def test_failure_caller_is_not_published(self):
        self.record('STEP', 'composition-failure')['observation']['caller_bits'][0] = 0
        self.reject()

    def test_failure_retains_changed_stage(self):
        before = self.record('STEP', 'request-reject')['observation']['staged_bits']
        self.record('STEP', 'composition-failure')['observation']['staged_bits'] = before[:]
        self.reject()

    def test_sticky_poison(self):
        for label in ('STEP', 'STEP_OUTCOME'):
            entry = self.record(label, 'poison-retry')
            (entry['observation'] if label == 'STEP' else entry)['poisoned'] = False
        self.reject()

    def test_aggregate_owner(self):
        self.record('CONSTRUCTOR')['observation']['admitted_bytes'] += 16*1024**2
        self.reject()

    def test_missing_outcome(self):
        index = next(i for i, (label, _) in enumerate(self.changed) if label == 'AFFINE_OUTCOME')
        self.changed.pop(index)
        self.reject()


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit('actual native capture and frozen table fixture directory')
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    for kind in (ReaderFailures, PreCapture) if sys.argv[1] == '--portable' else (ReaderFailures, PreCapture, CaptureMutations):
        suite.addTests(loader.loadTestsFromTestCase(kind))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
