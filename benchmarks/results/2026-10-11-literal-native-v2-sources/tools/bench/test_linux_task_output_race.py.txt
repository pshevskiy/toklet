"""Offline mutations for the narrow completed-output attach race."""
import base64
import hashlib
import unittest

import linux_task_evidence as candidate
import test_linux_task_evidence as fixtures


class CompletedOutputRaceTests(unittest.TestCase):
    def record(self):
        receipt = fixtures.capture([fixtures.ready()], output_limit_exceeded=True,
                                   process_group_killed=True, transport_exit_code=-9)
        raw = base64.b64decode(receipt['stdout_base64'])
        raw += b'x' * (65536 - len(raw))
        receipt.update(stdout_base64=base64.b64encode(raw).decode(),
                       stdout_captured_bytes=len(raw),
                       stdout_captured_sha256=hashlib.sha256(raw).hexdigest())
        return receipt

    def classify(self, receipt):
        return candidate.classify(receipt, fixtures.PAYLOAD, fixtures.RUNTIME, fixtures.PYTHON)

    def test_completed_remote_output_is_false_resource_result(self):
        result = self.classify(self.record())
        self.assertEqual((result['outcome'], result['passed'], result['reason']),
                         ('resource_failure', False, 'output_limit'))

    def test_missing_ownership_or_other_transport_status_stays_unresolved(self):
        for changes in ({'process_group_killed': False}, {'wall_timed_out': True},
                        {'output_limit_exceeded': False}, {'container_forced_kill': True},
                        {'input_completed': False}, {'transport_exit_code': 125},
                        {'transport_exit_code': 126}, {'transport_exit_code': 127}):
            receipt = self.record()
            receipt.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.classify(receipt)
        receipt = self.record()
        receipt['container_state']['OOMKilled'] = True
        with self.assertRaises(ValueError):
            self.classify(receipt)

    def test_unsaturated_or_unbound_capture_stays_unresolved(self):
        for rehash in (True, False):
            receipt = self.record()
            raw = base64.b64decode(receipt['stdout_base64'])[:-1]
            receipt.update(stdout_base64=base64.b64encode(raw).decode(), stdout_captured_bytes=len(raw))
            if rehash:
                receipt['stdout_captured_sha256'] = hashlib.sha256(raw).hexdigest()
            with self.subTest(rehash=rehash), self.assertRaises(ValueError):
                self.classify(receipt)

    def test_admission_and_cleanup_are_still_required(self):
        for changes in ({'container_removed': False}, {'cleanup_errors': [{'type': 'Error'}]},
                        {'surviving_group_members': [42]}):
            receipt = self.record()
            receipt.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.classify(receipt)
        receipt = self.record()
        raw = b'{}\n' + b'x' * 65533
        receipt.update(stdout_base64=base64.b64encode(raw).decode(), stdout_captured_bytes=len(raw),
                       stdout_captured_sha256=hashlib.sha256(raw).hexdigest())
        with self.assertRaises(ValueError):
            self.classify(receipt)


if __name__ == '__main__':
    unittest.main()
