import copy
import json
from pathlib import Path
import tempfile
import unittest
import prepare_llama3_tasks as subject


class TaskInputsTests(unittest.TestCase):
    def cases(self):
        rows=[]
        for index in range(100):
            category='code'if index<40 else'prose'if index<70 else'continuation'
            name=f'HumanEval/{index}'if index<40 else f'truthfulqa/row-{index-40}'if index<70 else f'lambada/row-{index-70}'
            rows.append(dict(id=name,category=category,prompt_tokens=[1,2],max_output_tokens=subject.task_cases.CAPS[category]))
        return rows
    def test_source_order_reserves_and_per_id_category_are_required(self):
        cases=self.cases();subject.validate_cases(cases)
        swapped=copy.deepcopy(cases)
        for field in ['category','max_output_tokens']:swapped[0][field],swapped[40][field]=swapped[40][field],swapped[0][field]
        with self.assertRaises(ValueError):subject.validate_cases(swapped)
        for field,value in [('prompt_tokens',[128256]),('prompt_tokens',[True]),('prompt_tokens',[1]*311),('max_output_tokens',255)]:
            wrong=copy.deepcopy(cases);wrong[0][field]=value
            with self.assertRaises(ValueError):subject.validate_cases(wrong)
        with self.assertRaises(ValueError):subject.validate_cases(cases[:-1])
    def test_prompt_translation_preserves_user_content_and_raw_continuation(self):
        case=dict(category='code',prompt_text=subject.PREFIX+'\n def f():\n    pass \n'+subject.SUFFIX)
        expected=subject.codec_policy.render([dict(role='user',content='\n def f():\n    pass \n')],[])
        self.assertEqual(subject.prompt(case,[]),expected)
        self.assertEqual(subject.prompt(dict(category='continuation',prompt_text='  raw\n'),[]),'  raw\n')
        with self.assertRaises(ValueError):subject.prompt(dict(category='prose',prompt_text='wrong'),[])

    def test_frozen_loader_rejects_sampling_and_execution_tampering(self):
        source=subject.ROOT/'benchmarks/llama3-quality-tasks-v1'
        original=json.loads((source/'manifest.json').read_bytes())
        body=(source/'execution.jsonl').read_bytes()
        with tempfile.TemporaryDirectory()as name:
            folder=Path(name)
            def check(value,data):
                raw=json.dumps(value).encode();(folder/'manifest.json').write_bytes(raw);(folder/'execution.jsonl').write_bytes(data)
                return subject.load(folder,subject.sha(raw))
            self.assertEqual(len(check(original,body)[1]),100)
            wrong=copy.deepcopy(original);wrong['sampling']='random'
            with self.assertRaises(ValueError):check(wrong,body)
            with self.assertRaises(ValueError):check(original,body[:-1]+b'x')
            cases=[json.loads(line)for line in body.splitlines()]
            for key in ['category','max_output_tokens']:cases[0][key],cases[40][key]=cases[40][key],cases[0][key]
            changed=''.join(json.dumps(c)+'\n'for c in cases).encode()
            wrong=copy.deepcopy(original);wrong['tasks']['bytes']=len(changed);wrong['tasks']['sha256']=subject.sha(changed)
            with self.assertRaises(ValueError):check(wrong,changed)
