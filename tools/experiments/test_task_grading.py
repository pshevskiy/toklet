"""Literal grading fixtures; no models, generated execution or VM."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import task_grading as grader
from test_task_cases import inventory


class LiteralGrading(unittest.TestCase):
    def fixture(self):
        cases=inventory()[0]
        completions=["    return 7" if case['category']=='code' else case['expected'] for case in cases]
        collection={'identity':{'backend':'metal','model':'larger','projection_mode':'HalfCompensated'},'cases':cases,'collection_sha256':hashlib.sha256(b'{}').hexdigest(),'raw_sha256':hashlib.sha256(b'').hexdigest(),'responses':[
            {'visible_text':text,'visible_text_sha256':hashlib.sha256(text.encode()).hexdigest()} for text in completions]}
        return cases,collection

    def run_fixture(self, classify=None, raw=None):
        cases,collection=self.fixture()
        seen=[]
        def execute(api,image,source,payload):
            index=len(seen)
            expected=grader.tasks.human_eval_program(cases[index],collection['responses'][index]['visible_text'])
            self.assertEqual(payload['program'],expected)
            self.assertEqual(payload['program_sha256'],hashlib.sha256(expected.encode()).hexdigest())
            self.assertEqual(api,('fixed-api',))
            seen.append(payload)
            return raw or {'container_removed':True,'container_cleanup_errors':[],
                           'cleanup_errors':[],'surviving_group_members':[]}
        with tempfile.TemporaryDirectory(dir=grader.boundary.ROOT/'benchmarks/raw') as directory:
            path=Path(directory)/'grading.json'
            collection_path=Path(directory)/'collection.json';collection_path.write_text('{}');collection_path.with_suffix('.raw.jsonl').write_bytes(b'')
            with patch.object(grader.admission,'validate_collection',return_value=collection), \
                    patch.object(grader.tasks,'load_cases',return_value=cases), \
                    patch.object(grader.controller,'bundled_source',return_value='fixed'), \
                    patch.object(grader.controller,'admit_bootstrap'), \
                    patch.object(grader.controller,'execute',side_effect=execute), \
                    patch.object(grader.evidence,'classify',side_effect=classify or (lambda *args:{'passed':True,'outcome':'passed'})):
                result=grader.run(('fixed-api',),'fixed-image',{},'fixed-python',collection_path,path)
            self.assertEqual(json.loads(path.read_text()),result)
        return result,seen

    def test_all_100_equal_weight_cases_and_40_literal_programs(self):
        result,seen=self.run_fixture()
        self.assertTrue(result['completed'])
        self.assertEqual(result['identity']['backend'],'metal')
        self.assertEqual(result['identity']['model'],'larger')
        self.assertTrue(result['quality_completed'])
        self.assertEqual(len(seen),40)
        self.assertEqual(result['aggregate']['passed'],100)
        self.assertEqual(result['aggregate']['pass_rate'],1)
        self.assertEqual(result['unresolved_case_ids'],[])

    def test_unresolved_compatibility_never_becomes_model_failure_or_aggregate(self):
        def unresolved(*args):raise ValueError('compatibility unresolved')
        result,seen=self.run_fixture(classify=unresolved)
        self.assertTrue(result['completed'])
        self.assertFalse(result['quality_completed'])
        self.assertIsNone(result['aggregate'])
        self.assertEqual(len(seen),40)
        self.assertEqual(len(result['unresolved_case_ids']),40)
        self.assertTrue(all(item['status']=='infrastructure_failure' and item['passed'] is None
                            for item in result['outcomes'][:40]))

    def test_unverified_cleanup_stops_before_any_following_container(self):
        def unresolved(*args):raise ValueError('cleanup unresolved')
        result,seen=self.run_fixture(classify=unresolved,raw={'container_removed':False,
            'container_cleanup_errors':[{'type':'unknown'}],'cleanup_errors':[], 'surviving_group_members':[]})
        self.assertFalse(result['completed'])
        self.assertEqual(len(seen),1)
        self.assertEqual(result['outcomes'],[])
        self.assertIsNone(result['aggregate'])
        self.assertFalse(result['quality_completed'])

    def test_known_resource_failure_remains_resolved_binary_model_failure(self):
        result,_=self.run_fixture(classify=lambda *args:{'passed':False,'outcome':'resource_failure','reason':'wall_limit'})
        self.assertTrue(result['quality_completed'])
        self.assertEqual(result['aggregate']['passed'],60)
        self.assertTrue(all(item['status']=='model_fail' and item['passed'] is False for item in result['outcomes'][:40]))


    def test_replacement_after_admission_prevents_every_execution(self):
        cases,collection=self.fixture()
        with tempfile.TemporaryDirectory(dir=grader.boundary.ROOT/'benchmarks/raw') as directory:
            path=Path(directory)/'collection.json';path.write_text('{}');path.with_suffix('.raw.jsonl').write_bytes(b'')
            def replaced(_):
                path.write_text('{"replacement":true}');return collection
            with patch.object(grader.admission,'validate_collection',side_effect=replaced),patch.object(grader.controller,'execute') as execute:
                with self.assertRaisesRegex(ValueError,'Admitted task collection changed'):grader.run(('fixed-api',),'fixed-image',{},'fixed-python',path,Path(directory)/'output.json')
                execute.assert_not_called()

if __name__=='__main__':unittest.main()
