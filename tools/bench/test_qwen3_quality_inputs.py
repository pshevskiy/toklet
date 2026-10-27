"""Full-workload selection/boundary controls; no inference or downloads."""
import unittest
from pathlib import Path
import copy
import hashlib
import json
import shutil
import prepare_qwen3_quality as subject
import quality_contract_policy as policy


def recipes():
    return ([dict(task_id=f'HumanEval/{i}', prompt=f'def f{i}():\n    ', test='pass', entry_point=f'f{i}') for i in range(40)],
        [dict(**{'Best Answer':'correct', 'Incorrect Answers':'wrong; correct; third; fourth', 'Question':f'Question {i}?'}) for i in range(30)],
        [dict(text='Keep CASE and punctuation Last!  \n') for _ in range(30)])


class FullInputs(unittest.TestCase):
    def test_original_case_counts_order_caps_and_expected_answers(self):
        tasks = policy.make_tasks(*recipes())
        self.assertEqual(len(tasks), 100)
        self.assertEqual([tasks[i]['max_output_tokens'] for i in (0,39,40,69,70,99)], [256,256,32,32,16,16])
        self.assertEqual(tasks[0]['code_prefix'], 'def f0():\n    ')
        self.assertEqual(tasks[41]['expected'], 'D')
        self.assertEqual(tasks[70]['expected'], 'Last!')
        self.assertEqual(tasks[70]['raw_prompt'], 'Keep CASE and punctuation')

    def test_missing_or_reordered_sources_cannot_shrink_the_suite(self):
        code, prose, continuation = recipes()
        with self.assertRaises(ValueError):policy.make_tasks(code, prose[:29], continuation)
        code[0], code[1] = code[1], code[0]
        with self.assertRaises(ValueError):policy.make_tasks(code, prose, continuation)
        with self.assertRaises(ValueError):policy.blocks(9999)

    def test_every_loss_target_once_with_preceding_token_only_overlap(self):
        blocks = policy.blocks()
        self.assertEqual(len(blocks), 10)
        self.assertEqual(sum(b['targets'] for b in blocks), 10000)
        self.assertEqual(blocks[-1]['targets'], 784)
        self.assertEqual(blocks[-1]['input_end'], 10001)
        for a,b in zip(blocks,blocks[1:]):self.assertEqual(a['input_end']-1,b['input_start'])

    def test_plain_qwen3_template_keeps_empty_thinking_branch_and_rejects_controls(self):
        self.assertEqual(subject.render('Hello', ['<|im_end|>']),
            '<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n')
        for content in ('', '<|im_end|>', '<tool_call>bad'):
            with self.assertRaises(ValueError):subject.render(content, ['<|im_end|>', '<tool_call>'])


def corruption_controls(directory, output):
    """Coherent changed metadata cannot bypass independent source replay."""
    import check_qwen3_quality as checker
    root = Path(directory);output = Path(output);output.mkdir()
    original = json.loads((root / 'manifest.json').read_text())
    raw = {name:(root/name).read_bytes() for name in original['files']}
    findings = []
    for name in ('loss_ids','prompt_ids','code_tests','prose_expected','continuation_expected','output_cap',
                 'quality_limit','candidate_capacity','candidate_precision','baseline_descriptor',
                 'baseline_config_asset','baseline_tokenizer_asset'):
        m = copy.deepcopy(original);files = dict(raw);rows = [json.loads(v) for v in files['tasks.jsonl'].splitlines()]
        baseline = None
        if name == 'loss_ids':
            body = bytearray(files['held_out_loss.u32le']);body[:4] = (0).to_bytes(4,'little');files['held_out_loss.u32le'] = bytes(body)
        elif name == 'prompt_ids':rows[0]['prompt_tokens'][0] = 0
        elif name == 'code_tests':rows[0]['tests'] = 'pass\n'
        elif name == 'prose_expected':rows[40]['expected'] = 'Z'
        elif name == 'continuation_expected':rows[70]['expected'] += '!changed'
        elif name == 'output_cap':
            rows[0]['max_output_tokens'] -= 1;rows[0]['required_capacity'] -= 1
        elif name == 'quality_limit':m['quality_policy']['mean_nll_increase_limit_nats_per_token'] = 0.1
        elif name == 'candidate_capacity':m['candidate']['diagnostic_capacity'] = 512
        elif name == 'candidate_precision':m['candidate']['precision'] = 'BF16'
        elif name == 'baseline_descriptor':m['unquantized_baseline']['config_sha256'] = '0'*64
        else:
            baseline = output/(name+'-baseline');baseline.mkdir()
            for asset in ('config.json','tokenizer.json'):shutil.copyfile(subject.BASELINE/asset,baseline/asset)
            asset = baseline/('config.json' if name == 'baseline_config_asset' else 'tokenizer.json')
            with asset.open('ab') as f:f.write(b'\n')
        files['tasks.jsonl'] = b''.join(subject.encoded(v) for v in rows)
        files['execution.jsonl'] = b''.join(subject.encoded({k:v[k] for k in ('id','category','prompt_tokens','max_output_tokens')}) for v in rows)
        m['tasks'] = [dict(id=t['id'],category=t['category'],prompt_tokens=len(t['prompt_tokens']),
            max_output_tokens=t['max_output_tokens'],required_capacity=t['required_capacity'],
            payload_sha256=hashlib.sha256(subject.encoded(t)).hexdigest()) for t in rows]
        m['maximum_task_capacity'] = max(t['required_capacity'] for t in rows)
        m['files'] = {n:dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest()) for n,b in files.items()}
        target = output/name;target.mkdir()
        for n,b in {**files,'manifest.json':subject.encoded(m)}.items():(target/n).write_bytes(b)
        try:checker.check(target,hashlib.sha256((target/'manifest.json').read_bytes()).hexdigest(),baseline)
        except ValueError as error:findings.append(dict(control=name,rejected=True,error=str(error)))
        else:raise AssertionError('coherent quality corruption accepted: '+name)
    return findings


if __name__ == '__main__':unittest.main()
