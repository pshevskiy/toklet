"""Literal grading fixtures; no models, generated execution or VM."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import held_out_task_grading as grader
from test_task_cases import inventory


class LiteralGrading(unittest.TestCase):
    def fixture(self):
        cases=inventory()[0]
        completions=["    return 7" if case['category']=='code' else case['expected'] for case in cases]
        collection={'projection':'scalar','run':{'cases':[
            {'response':{'visible_text':text,'visible_text_sha256':hashlib.sha256(text.encode()).hexdigest()}}
            for text in completions]}}
        return cases,collection

    def run_fixture(self, classify=None, raw=None):
        cases,collection=self.fixture()
        seen=[]
        def execute(api,image,source,payload):
            index=len(seen)
            expected=grader.tasks.human_eval_program(cases[index],collection['run']['cases'][index]['response']['visible_text'])
            self.assertEqual(payload['program'],expected)
            self.assertEqual(payload['program_sha256'],hashlib.sha256(expected.encode()).hexdigest())
            self.assertEqual(api,('fixed-api',))
            seen.append(payload)
            return raw or {'container_removed':True,'container_cleanup_errors':[],
                           'cleanup_errors':[],'surviving_group_members':[]}
        with tempfile.TemporaryDirectory(dir=grader.boundary.ROOT/'benchmarks/raw') as directory:
            path=Path(directory)/'grading.json'
            collection_path=Path(directory)/'collection.json';collection_path.write_text('{}')
            with patch.object(grader,'validate_collection',return_value=(collection,cases)), \
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
        self.assertNotIn('aggregate',result)

    def test_known_resource_failure_remains_resolved_binary_model_failure(self):
        result,_=self.run_fixture(classify=lambda *args:{'passed':False,'outcome':'resource_failure','reason':'wall_limit'})
        self.assertTrue(result['quality_completed'])
        self.assertEqual(result['aggregate']['passed'],60)
        self.assertTrue(all(item['status']=='model_fail' and item['passed'] is False for item in result['outcomes'][:40]))


class CollectionProvenance(unittest.TestCase):
    def fixture(self, directory):
        from test_task_cases import events
        cases=inventory()[0]
        sha=hashlib.sha256(b'frozen').hexdigest()
        binary=Path(directory)/'qualified-cli'
        fields=('weights_bytes','kv_bytes','scratch_bytes','history_bytes','logits_bytes',
                'metadata_bytes','metadata_allowance_bytes','tokenizer_application_allowance_bytes')
        ready={'event':'ready','backend':'cpu','context_capacity':2048,'cpu_projection':'scalar',
               'memory_admitted_bytes':800,'logical_memory':{name:100 for name in fields}}
        helper_names=('task_cases.py','prepare.py','cpu_cli.py','cpu.py','mlx_baseline.py',
            'task_cases_probe.py','quality.py','linux_code_isolation_probe.py','linux_task_evidence.py',
            'linux_task_child.py','linux_code_isolation_child.py','code_isolation_child.py','held_out_tasks.py')
        qualification={'source_sha256':{'src/lib.rs':sha},'compiler':'authored',
                       'runs':{'scalar':{'ready':copy.deepcopy(ready)}}}
        source={'model':{'weights':{'path':'authored-model','sha256':sha}},'assets':[]}
        run_cases=[];wrappers=[{'projection':'scalar','case_id':None,'line':json.dumps(ready)+'\n'}]
        for case in cases:
            stream=events(case,count=0,eos=True,projection='scalar')
            response=grader.tasks.collect_response(case,stream,'scalar')
            run_cases.append({'request':grader.tasks.request(case),'semantic_events':stream,'response':response,
                              'cli_physical_peak_bytes':1000,'driver_physical_peak_bytes':1000,
                              'conservative_joint_physical_peak_bytes':2000})
            wrappers.extend({'projection':'scalar','case_id':case['id'],'line':json.dumps(event)+'\n'} for event in stream)
        path=Path(directory)/'collection.json'
        raw=Path(str(path)+'.native.jsonl')
        raw.write_text(''.join(json.dumps(wrapper)+'\n' for wrapper in wrappers))
        errors=Path(str(path)+'.stderr.txt');errors.write_bytes(b'')
        command=[str(binary),'--model',str(grader.boundary.ROOT/'models/Qwen2.5-0.5B-Instruct'),
                 '--interactive','--json','--context-capacity','2048','--prefill-chunk','128','--cpu-projection','scalar']
        record={'format':'toklet-held-out-task-collection-v1','completed':True,'working_tree_dirty':False,
                'task_payload_sha256':grader.tasks.TASK_SHA,'projection':'scalar','engine_revision':'a'*40,
                'cli_qualification_sha256':sha,'binary_sha256':sha,'binary_path':str(binary),
                'compiled_source_sha256':qualification['source_sha256'],'compiler':'authored',
                'sources_manifest_sha256':sha,'contract_manifest_sha256':sha,'model':source['model'],
                'validated_local_model_assets':{'authored-model':sha},
                'helper_sha256':{'tools/bench/'+name:sha for name in helper_names},
                'case_inventory':[{'id':case['id'],'category':case['category'],
                                   'prompt_token_count':len(case['prompt_tokens'])} for case in cases],
                'native_raw_sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),
                'stderr_sha256':hashlib.sha256(b'').hexdigest(),
                'run':{'projection':'scalar','passed':True,'exit_code':0,'watchdog_killed':False,
                       'process_group_killed':False,'cleanup_errors':[],'surviving_group_members':[],
                       'ready':ready,'command':command,'cases':run_cases,
                       'stdout_sha256':hashlib.sha256(''.join(wrapper['line'] for wrapper in wrappers).encode()).hexdigest()}}
        def digest(value):
            value=Path(value)
            if value in (raw,errors):return hashlib.sha256(value.read_bytes()).hexdigest()
            return sha
        return record,path,cases,source,binary,qualification,digest

    def validate(self, mutate=None):
        with tempfile.TemporaryDirectory() as directory:
            record,path,cases,source,binary,qualification,digest=self.fixture(directory)
            if mutate:mutate(record)
            path.write_text(json.dumps(record))
            with patch.object(grader.tasks,'load_cases',return_value=cases), \
                    patch.object(grader.cpu_cli,'qualified_cli',return_value=(binary,qualification)), \
                    patch.object(grader.tasks.prepare,'digest',side_effect=digest), \
                    patch.object(grader.tasks.prepare,'load_json',return_value=source), \
                    patch.object(grader.tasks.prepare,'validate_source'), \
                    patch.object(grader.subprocess,'check_output',return_value=b'frozen'):
                return grader.validate_collection(path)

    def test_complete_100_case_semantic_and_metadata_fixture_replays(self):
        record,cases=self.validate()
        self.assertEqual(len(cases),100)
        self.assertEqual(len(record['run']['cases']),100)

    def test_completed_looking_metadata_mutations_rejected(self):
        def bad_ready(record):record['run']['ready']['backend']='gpu'
        def bad_command(record):record['run']['command'].append('--no-eos')
        def bad_binary(record):record['binary_sha256']='0'*64
        def bad_sources(record):record['compiled_source_sha256']={}
        def bad_model(record):record['model']={}
        def bad_contract(record):record['contract_manifest_sha256']='0'*64
        def missing_helper(record):record['helper_sha256'].pop('tools/bench/held_out_tasks.py')
        def bad_helper(record):record['helper_sha256']['tools/bench/held_out_tasks.py']='0'*64
        def bad_memory(record):record['run']['ready']['memory_admitted_bytes']=799
        def bad_peak(record):record['run']['cases'][0]['conservative_joint_physical_peak_bytes']=1999
        def bad_dirty(record):record['working_tree_dirty']=True
        for mutate in (bad_ready,bad_command,bad_binary,bad_sources,bad_model,bad_contract,
                       missing_helper,bad_helper,bad_memory,bad_peak,bad_dirty):
            with self.subTest(mutate=mutate.__name__):
                with self.assertRaises(ValueError):self.validate(mutate)


if __name__=='__main__':unittest.main()
