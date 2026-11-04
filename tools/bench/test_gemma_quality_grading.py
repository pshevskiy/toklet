"""A failed/live model collection must not decode or start the isolated grader."""
import tempfile
import copy
import hashlib
import json
import test_gemma_task_results as fixtures
from pathlib import Path
from unittest.mock import patch
import unittest
import gemma_quality_grading as target


class Controls(unittest.TestCase):
    def test_failed_worker_blocks_decode_and_vm(self):
        collection=dict(format='toklet-gemma-paired-task-collection-v1',completed=True,cases=100,
            manifest_sha256=target.MANIFEST_SHA,input_directory='artifacts/fake-inputs',
            outputs={role:dict(worker_receipt=f'artifacts/{role}.json',worker_receipt_sha256='0'*64)for role in ['original','native']})
        failed=dict(passed=False,cases=100,command=dict(pid=10,exit_code=1,primary_error='unverified',cleanup_errors=[]))
        complete=dict(passed=True,cases=100,command=dict(pid=20,exit_code=0,primary_error=None,cleanup_errors=[]))
        values=[(collection,b''),(dict(quality=dict(tasks_file='artifacts/fake-canonical.jsonl')),b''),(failed,b''),(complete,b'')]
        with tempfile.TemporaryDirectory(dir=target.ROOT/'artifacts')as folder:
            output=Path(folder)/'grade'
            with patch.object(target,'bind'),patch.object(target,'read_json',side_effect=values),\
                 patch.object(target.inputs,'structure',return_value=(dict(assets={}),[{}]*100,[],{})),\
                 patch.object(target.results,'canonical_grading_cases',return_value=[{}]*100),\
                 patch.object(target.results,'decode')as decode,\
                 patch.object(target.isolated,'grade_paired_cases')as execute:
                result=target.run(Path(folder)/'collection.json','0'*64,output)
            self.assertFalse(result['completed'])
            self.assertIn('model-worker exits',result['failure']['message'])
            decode.assert_not_called();execute.assert_not_called()

    def test_literal_adapter_calls_shared_runner_without_changing_prompts(self):
        canonical=fixtures.cases();gemma=copy.deepcopy(canonical)
        for case in gemma:case['prompt_tokens']=[200000]
        source=fixtures.original(gemma);native=[]
        for case in gemma:
            reason=dict(kind='stop_token',token_id=106)
            native.append(dict(id=case['id'],category=case['category'],prompt_tokens=1,
                output_reserve=case['max_output_tokens'],visible_token_ids=[],stop_reason=reason,
                generation=dict(reason=reason,reused_tokens=0,prefill_tokens=1,sampled_tokens=1,
                    emitted_tokens=0,evaluated_tokens=1,history_tokens=2,pending_tokens=1)))
        raw=b'{}';spec=dict(bytes=2,sha256=hashlib.sha256(raw).hexdigest())
        manifest=dict(assets={},model_assets={'tokenizer.json':spec},unquantized_baseline=dict(tokenizer_sha256=spec['sha256']))
        with tempfile.TemporaryDirectory(dir=target.ROOT/'artifacts')as folder:
            folder=Path(folder);relative=str(folder.relative_to(target.ROOT));(folder/'tokenizer.json').write_bytes(raw)
            outputs={}
            for role,records in [('original',source),('native',native)]:
                path=folder/(role+'.jsonl');path.write_text(''.join(json.dumps(r)+'\n'for r in records))
                outputs[role]=dict(worker_receipt=relative+'/'+role+'.json',worker_receipt_sha256='0'*64,
                    journal=str(path.relative_to(target.ROOT)),journal_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),evidence_bindings={})
            collection=dict(format='toklet-gemma-paired-task-collection-v1',completed=True,cases=100,
                manifest_sha256=target.MANIFEST_SHA,input_directory=relative,tokenizer_directory=relative,outputs=outputs)
            worker=lambda pid,role:dict(passed=True,cases=100,tasks_sha256=outputs[role]['journal_sha256'],command=dict(pid=pid,exit_code=0,primary_error=None,cleanup_errors=[]))
            values=[(collection,b''),(dict(quality=dict(tasks_file='artifacts/fake-canonical.jsonl')),b''),(worker(10,'original'),b''),(worker(20,'native'),b'')]
            decoded=[dict(id=c['id'],category=c['category'],visible_token_ids=[],visible_text='',visible_text_sha256=hashlib.sha256(b'').hexdigest())for c in gemma]
            with patch.object(target,'bind'),patch.object(target,'read_json',side_effect=values),                 patch.object(target.inputs,'structure',return_value=(manifest,gemma,[],{})),                 patch.object(target.results.task_cases,'load_cases',return_value=canonical),                 patch.object(target.results,'require_idle',return_value=[10,20]),                 patch.object(target.results,'decode',return_value=decoded)as decode,                 patch.object(target.isolated,'grade_paired_cases',return_value={'completed':True})as execute:
                report=target.run(folder/'collection.json','0'*64,folder/'grade')
            self.assertTrue(report['completed']);self.assertEqual(decode.call_count,2)
            self.assertIs(execute.call_args.args[0],canonical)
            self.assertEqual(gemma[0]['prompt_tokens'],[200000])
            self.assertEqual(canonical[0]['prompt_tokens'],[1])
            alternate=copy.deepcopy(source)
            alternate[0]['generation']['stop_reason']['token_id']=50
            alternate[0]['generation']['samples'][0]['token_id']=50
            alternate[0]['trajectory']['rows'][0].update(expected_id=50,stateless_id=50)
            self.assertEqual(len(target.results.collection(gemma,alternate,'original')),100)
            alternate_path=folder/'alternate-valid-source.jsonl'
            alternate_path.write_text(''.join(json.dumps(r)+'\n'for r in alternate))
            swapped=copy.deepcopy(collection)
            swapped['outputs']['original'].update(journal=str(alternate_path.relative_to(target.ROOT)),
                journal_sha256=hashlib.sha256(alternate_path.read_bytes()).hexdigest())
            values=[(swapped,b''),(dict(quality=dict(tasks_file='artifacts/fake-canonical.jsonl')),b''),(worker(10,'original'),b''),(worker(20,'native'),b'')]
            with patch.object(target,'bind'),patch.object(target,'read_json',side_effect=values),                 patch.object(target.inputs,'structure',return_value=(manifest,gemma,[],{})),                 patch.object(target.results.task_cases,'load_cases',return_value=canonical),                 patch.object(target.results,'require_idle',return_value=[10,20]),                 patch.object(target.results,'decode')as decode,                 patch.object(target.isolated,'grade_paired_cases')as execute:
                rejected=target.run(folder/'collection.json','0'*64,folder/'wrong-journal')
            self.assertFalse(rejected['completed'])
            self.assertIn('exited worker output',rejected['failure']['message'])
            decode.assert_not_called();execute.assert_not_called()



if __name__=='__main__':unittest.main()
