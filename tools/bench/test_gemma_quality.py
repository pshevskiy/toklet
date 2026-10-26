"""Synthetic recipe controls and explicit coherent workload corruptions."""
from pathlib import Path
import copy
import hashlib
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import ModuleType
import prepare_gemma_quality as prepare
import check_gemma_quality as check

def recipes():
    code=[dict(task_id=f'HumanEval/{i}',prompt=f'def f{i}():\n    ',test=f'assert f{i}()',entry_point=f'f{i}') for i in range(40)]
    prose=[{'Best Answer':'correct','Incorrect Answers':'wrong; correct; another; fourth','Question':f'Question {i}?'} for i in range(30)]
    continuation=[dict(text=f'A synthetic passage ending answer{i}  \n') for i in range(30)]
    return code,prose,continuation

class Recipe(unittest.TestCase):
    def test_extra_parser_packages_must_be_cold(self):
        for name in ['numpy','pyarrow']:
            with patch.dict(sys.modules,{name:ModuleType(name)}):
                with self.assertRaisesRegex(ValueError,'already imported'):prepare.gemma_quality_runtime.admit(Path('absent'),'0'*64,[])
    def test_all_original_source_counts_order_and_caps(self):
        tasks=prepare.make_tasks(*recipes());self.assertEqual(len(tasks),100)
        self.assertEqual([tasks[i]['max_output_tokens'] for i in (0,39,40,69,70,99)],[256,256,32,32,16,16])
        self.assertEqual(tasks[0]['code_prefix'],'def f0():\n    ')
    def test_option_rotation_has_no_model_based_selection(self):
        tasks=prepare.make_tasks(*recipes());self.assertEqual(tasks[40]['expected'],'A');self.assertEqual(tasks[41]['expected'],'D')
        self.assertEqual(len(tasks[41]['options']),4);self.assertEqual(tasks[41]['option_rotation'],1)
    def test_continuation_preserves_exact_expected_word(self):
        code,prose,continuation=recipes();continuation[0]['text']='Keep CASE and punctuation Last!  \n'
        task=prepare.make_tasks(code,prose,continuation)[70];self.assertEqual(task['raw_prompt'],'Keep CASE and punctuation');self.assertEqual(task['expected'],'Last!')
    def test_missing_rows_or_reordered_source_ids_reject(self):
        code,prose,continuation=recipes()
        with self.assertRaises(ValueError):prepare.make_tasks(code,prose[:29],continuation)
        code[0],code[1]=code[1],code[0]
        with self.assertRaises(ValueError):prepare.make_tasks(code,prose,continuation)
    def test_loss_coverage_and_only_previous_token_overlap(self):
        blocks=prepare.blocks();self.assertEqual(len(blocks),10);self.assertEqual(sum(v['targets'] for v in blocks),10000)
        self.assertEqual(blocks[-1]['targets'],784);self.assertEqual(blocks[-1]['input_end'],10001)
        for a,b in zip(blocks,blocks[1:]):self.assertEqual(a['input_end']-1,b['input_start'])
        with self.assertRaises(ValueError):prepare.blocks(9999)
    def test_empty_distractors_and_missing_last_word_reject(self):
        code,prose,continuation=recipes();prose[0]['Incorrect Answers']='correct; '
        with self.assertRaises(ValueError):prepare.make_tasks(code,prose,continuation)
        code,prose,continuation=recipes();continuation[0]['text']='NoSpace'
        with self.assertRaises(ValueError):prepare.make_tasks(code,prose,continuation)

class Captured(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=Path(sys.argv[-1]);cls.raw={p.name:p.read_bytes() for p in cls.root.iterdir() if p.is_file()}
        cls.manifest=check.safe.document(cls.raw['manifest.json']);cls.payload=[check.safe.document(line) for line in cls.raw['tasks.jsonl'].splitlines()]
    def reject(self,manifest=None,payload=None,files=None):
        m=copy.deepcopy(self.manifest) if manifest is None else manifest;rows=copy.deepcopy(self.payload) if payload is None else payload
        data={k:v for k,v in self.raw.items() if k!='manifest.json'}
        data['tasks.jsonl']=b''.join(check.encoded(v) for v in rows)
        if files:data.update(files)
        for name,raw in data.items():m['assets'][name]=dict(file=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        m['cases']=[dict(id=v['id'],category=v['category'],prompt_tokens=len(v['prompt_tokens']),max_output_tokens=v['max_output_tokens'],required_capacity=v['required_capacity'],payload_sha256=hashlib.sha256(check.encoded(v)).hexdigest()) for v in rows]
        raw=check.encoded(m)
        with tempfile.TemporaryDirectory()as folder:
            root=Path(folder)
            for name,value in data.items():(root/name).write_bytes(value)
            (root/'manifest.json').write_bytes(raw)
            with self.assertRaises((ValueError,KeyError,TypeError,IndexError)):check.structure(root,hashlib.sha256(raw).hexdigest())
    def test_complete_manifest_structure(self):
        m,rows,ids,_=check.structure(self.root,hashlib.sha256(self.raw['manifest.json']).hexdigest());self.assertEqual(len(rows),100);self.assertEqual(len(ids),10001);self.assertGreaterEqual(m['capacity']['full_suite_required'],1025)
    def test_loss_tail_and_overlapping_block_labels(self):
        self.reject(files={'held_out_loss.u32le':self.raw['held_out_loss.u32le'][:-4]})
        m=copy.deepcopy(self.manifest);m['blocks'][1]['input_start']+=1;self.reject(manifest=m)
    def test_no_task_pruning_or_output_cap_reduction(self):
        self.reject(payload=copy.deepcopy(self.payload[:-1]))
        rows=copy.deepcopy(self.payload);rows[0]['max_output_tokens']=32;rows[0]['required_capacity']=len(rows[0]['prompt_tokens'])+32;self.reject(payload=rows)
    def test_category_and_unknown_payload_field(self):
        rows=copy.deepcopy(self.payload);rows[0]['category']='prose';self.reject(payload=rows)
        rows=copy.deepcopy(self.payload);rows[0]['extra']='ignored?';self.reject(payload=rows)
    def test_bos_vocabulary_and_typed_token_ids(self):
        for value in [2,262144,False]:
            rows=copy.deepcopy(self.payload);rows[0]['prompt_tokens'][1]=value;self.reject(payload=rows)
    def test_original_quality_thresholds_and_scorers(self):
        m=copy.deepcopy(self.manifest);m['policy']['mean_nll_increase_limit_nats_per_token']=0.1;self.reject(manifest=m)
        m=copy.deepcopy(self.manifest);m['scoring']['code']='remove_fences_then_grade';self.reject(manifest=m)
    def test_real_unquantized_baseline_not_same_four_bit_model(self):
        m=copy.deepcopy(self.manifest);m['unquantized_baseline']['repository']='mlx-community/gemma-4-e2b-it-4bit';self.reject(manifest=m)
    def test_no_quality_claim_from_input_preparation(self):
        m=copy.deepcopy(self.manifest);m['quality_scored']=True;self.reject(manifest=m)
        m=copy.deepcopy(self.manifest);m['capacity']['current_engine_satisfies_contract']=True;self.reject(manifest=m)

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Recipe if sys.argv[1]=='--recipe' else Captured)
    result=unittest.TextTestRunner().run(suite);sys.exit(0 if result.wasSuccessful() else 1)
