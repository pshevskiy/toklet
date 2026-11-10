"""Portable full-coverage, resource/capacity and qualification rejection controls."""
import copy
import json
import struct
import sys
import unittest

import check_minicpm5_quality_inputs as reader
import prepare_minicpm5_quality as producer


def fixture():
    frozen=dict(candidate={},modified_parent={},original_stock={},comparisons=['stock','parent','candidate'],
                quality=dict(loss_targets=10000,tasks=100))
    tasks=[]
    for cat,count,cap in [('code',40,256),('prose',30,32),('continuation',30,16)]:
        for i in range(count):
            name=f'HumanEval/{i}' if cat=='code' else f'truthfulqa/row-{i}' if cat=='prose' else f'lambada/row-{i}'
            tasks.append(dict(id=name,category=cat,source_row=i,max_output_tokens=cap,
                prompt='fixture',prompt_tokens=[0,7,13],required_capacity=3+cap))
    blobs={'held_out_loss.u32le':struct.pack('<10001I',*range(10001)),
           'tasks.jsonl':b''.join(producer.encoded(row) for row in tasks),
           'loss_verification_prefix.txt':b'a'*65536}
    manifest=dict(format='toklet-minicpm5-complete-quality-inputs-v1',contract_sha256='a'*64,
        candidate={},modified_parent={},original_stock={},comparison_order=frozen['comparisons'],
        quality=frozen['quality'],scoring=producer.policy.SCORING,blocks=producer.policy.blocks(),
        datasets={},complete_corpus_tokens=20000,loss_targets=10000,task_cases=100,
        cases=[dict(id=r['id'],category=r['category'],prompt_tokens=3,max_output_tokens=r['max_output_tokens'],
                    required_capacity=r['required_capacity'],payload_sha256=producer.digest(producer.encoded(r))) for r in tasks],
        assets={},files={n:dict(bytes=len(raw),sha256=producer.digest(raw)) for n,raw in blobs.items()},
        capacity=dict(loss_context_required=1025,maximum_task_required=259,full_suite_required=1025,
            current_engine_capacity=512,current_engine_satisfies_contract=False,no_workload_truncation=True),
        input_equivalence=dict(complete_corpus_token_ids=True,all_100_task_rendered_strings=True,
            all_100_task_token_ids=True,tokenizer_core_structure_equal=True,
            padding_difference='stock None vs candidate BatchLongest-left; single sequence retains identical IDs',wrapper_defaults_used=False),
        stops=dict(stop_ids=[1,130073]),model_executed=False,quality_scored=False,
        generated_code_executed=False,parent_weights_downloaded=False,managed_adoption_qualified=False,
        performance_or_overhead_qualified=False)
    blobs['manifest.json']=producer.encoded(manifest)
    return blobs,frozen,manifest,tasks


def altered(manifest,blobs):
    result=dict(blobs);result['manifest.json']=producer.encoded(manifest);return result


class QualityInputs(unittest.TestCase):
    def test_import_does_not_load_parser_tokenizer_or_numerical_packages(self):
        for root in ['numpy','pyarrow','tokenizers','mlx.core']:
            self.assertNotIn(root,sys.modules)

    def test_full_fixture_and_exact_disjoint_loss_target_coverage(self):
        blobs,frozen,manifest,tasks=fixture()
        self.assertEqual(reader.validate(blobs,frozen,'a'*64),manifest)
        targets=[]
        for b in manifest['blocks']:
            self.assertEqual(b['input_end']-b['input_start'],b['targets']+1)
            targets.extend(range(b['input_start']+1,b['input_end']))
        self.assertEqual(targets,list(range(1,10001)))
        self.assertEqual(manifest['blocks'][-1]['targets'],784)

    def test_shortened_duplicated_or_reordered_loss_blocks_reject(self):
        blobs,frozen,manifest,tasks=fixture()
        for blocks in [manifest['blocks'][:-1],manifest['blocks']*2,list(reversed(manifest['blocks']))]:
            m=dict(manifest,blocks=blocks)
            with self.assertRaises(ValueError):reader.validate(altered(m,blobs),frozen,'a'*64)

    def test_manifest_types_promotion_and_full_capacity_are_strict(self):
        blobs,frozen,manifest,tasks=fixture()
        for change in [dict(loss_targets=True),dict(task_cases=99),dict(complete_corpus_tokens=10000),
                       dict(quality_scored=True),dict(model_executed=0),dict(extra=True)]:
            with self.assertRaises(ValueError):reader.validate(altered(dict(manifest,**change),blobs),frozen,'a'*64)
        for change in [dict(full_suite_required=512),dict(current_engine_satisfies_contract=True),
                       dict(no_workload_truncation=False),dict(loss_context_required=True)]:
            m=dict(manifest,capacity=dict(manifest['capacity'],**change))
            with self.assertRaises(ValueError):reader.validate(altered(m,blobs),frozen,'a'*64)

    def test_complete_payload_extents_hashes_stop_ids_and_original_caps_reject(self):
        blobs,frozen,manifest,tasks=fixture()
        for name in reader.FILES:
            changed=dict(blobs);changed[name]=changed[name][:-1]
            with self.assertRaises(ValueError):reader.validate(changed,frozen,'a'*64)
        for index in [0,40,70]:
            changed=copy.deepcopy(tasks);changed[index]['max_output_tokens']-=1
            b=dict(blobs);b['tasks.jsonl']=b''.join(producer.encoded(r) for r in changed)
            m=copy.deepcopy(manifest);m['files']['tasks.jsonl']=dict(bytes=len(b['tasks.jsonl']),sha256=producer.digest(b['tasks.jsonl']))
            with self.assertRaises(ValueError):reader.validate(altered(m,b),frozen,'a'*64)
        with self.assertRaises(ValueError):reader.validate(altered(dict(manifest,stops=dict(stop_ids=[1])),blobs),frozen,'a'*64)

    def test_duplicate_unknown_json_fields_and_nonfinite_values_reject(self):
        for raw in [b'{"a":1,"a":2}',b'{"x":NaN}',b'{"x":Infinity}']:
            with self.assertRaises(ValueError):producer.document(raw)

    def test_utf8_verification_prefix_only_trims_incomplete_last_codepoint(self):
        raw=producer.prefix_bytes('a'*65535+'🙂'+'b'*10)
        self.assertEqual(raw,b'a'*65535)
        self.assertEqual(producer.prefix_bytes('a'*65532+'🙂'+'b'*10),b'a'*65532+'🙂'.encode())


if __name__=='__main__':unittest.main()
