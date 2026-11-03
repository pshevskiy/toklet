import copy
import json
import hashlib
from pathlib import Path
import tempfile
import unittest
import run_llama3_quality_grading as subject


class PublicationControls(unittest.TestCase):
    def test_only_clean_outer_and_restored_inner_decisions_are_usable(self):
        window=dict(exit_code=0,primary_error=None,cleanup_errors=[],independent_stopped_verification=True)
        inner=dict(completed=True,restored_stopped=True,pipe_lifetime_finished=True,unresolved_case_ids=[])
        self.assertTrue(subject.acceptable(window,inner))
        for field,value in [('exit_code',1),('primary_error','timeout'),('cleanup_errors',['live']),('independent_stopped_verification',False)]:
            wrong=copy.deepcopy(window);wrong[field]=value;self.assertFalse(subject.acceptable(wrong,inner))
        for field,value in [('completed',False),('restored_stopped',False),('pipe_lifetime_finished',False),('unresolved_case_ids',['case']),('failure',{'message':'bad'}),('restoration_failure',{'message':'bad'})]:
            wrong=copy.deepcopy(inner);wrong[field]=value;self.assertFalse(subject.acceptable(window,wrong))

    def test_inner_receipt_larger_than_journal_cap_is_bounded_and_hashed_from_same_bytes(self):
        value=dict(completed=True,padding='x'*(9*1024**2))
        with tempfile.TemporaryDirectory()as name:
            path=Path(name)/'result.json';raw=json.dumps(value).encode();path.write_bytes(raw)
            inner,pin=subject.read_inner_result(path)
            self.assertTrue(inner['completed']);self.assertGreater(pin['bytes'],8*1024**2)
            self.assertEqual(pin['sha256'],hashlib.sha256(raw).hexdigest())
