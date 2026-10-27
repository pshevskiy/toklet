"""Independent full-source/tokenization replay; no inference or generated code."""
from pathlib import Path
import csv
import gzip
import hashlib
import io
import json
import struct

import prepare_qwen3_quality as identity
safe = identity.safe


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def check(directory, expected_sha, baseline=None):
    import numpy
    import pyarrow as pa
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    from jinja2 import Environment, StrictUndefined
    directory = Path(directory)
    m = safe.document(identity.checked(directory / 'manifest.json', expected_sha, 2 * 1024**2))
    safe.same(set(m), {'format','candidate','unquantized_baseline','source_assets','tokenizer_assets',
        'tokenization','quality_policy','scoring','stops','blocks','loss_targets','files','tasks',
        'maximum_task_capacity','full_suite_capacity','quality_scored','generated_code_executed','full_model_window_qualified'})
    safe.same(m['format'], 'toklet-qwen3-quality-contract-v1')
    for flag in ('quality_scored', 'generated_code_executed', 'full_model_window_qualified'):
        safe.same(m[flag], False)
    safe.same(m['loss_targets'], 10000);safe.same(m['full_suite_capacity'], 1025);safe.same(m['stops'], [151645])
    safe.same(m['quality_policy'], dict(mean_nll_increase_limit_nats_per_token=0.05,
        task_pass_rate_loss_limit_percentage_points=2.0, source_manifest_sha256=identity.SOURCES_SHA,
        original_contract_sha256=identity.OLD_CONTRACT_SHA))
    safe.same(m['candidate'], dict(repository='mlx-community/Qwen3-0.6B-8bit', revision=identity.REVISION,
        weight_sha256=identity.WEIGHT_SHA, numerical_source_sha256=identity.candidate_sources(),
        precision='packed8/group64 weights, Half activations/KV, Compensated normalization, Reference attention',
        numerical_gate_status='C512 rejected unchanged', diagnostic_capacity=1025, app_limits_changed=False))
    safe.same(m['scoring'], identity.policy.SCORING)
    safe.same(m['tokenization'], dict(add_special_tokens=False, code_prose='pinned_plain_enable_thinking_false',
        loss_continuation='raw_no_added_BOS', corpus_join='newline_join_nonempty_original_rows',
        generation_defaults_are_hard_limits=False))
    safe.same(m['tokenizer_assets'], {name:dict(bytes=size,sha256=pin) for name,(size,pin) in identity.PINS.items()})
    safe.same(m['candidate']['repository'], 'mlx-community/Qwen3-0.6B-8bit')
    safe.same(m['candidate']['revision'], identity.REVISION)
    safe.same(m['unquantized_baseline'], dict(repository='Qwen/Qwen3-0.6B', revision=identity.BASE_REVISION,
        weight_sha256=identity.BASE_SHA, config_sha256=identity.BASE_CONFIG_SHA,
        config_equal_excluding_quantization=True, tokenizer_sha256=identity.PINS['tokenizer.json'][1],
        identical_frozen_input_ids=True,
        lineage='contemporary official revision; converter records repository but no source revision'))
    base = identity.BASELINE if baseline is None else Path(baseline)
    original_config = safe.document(identity.checked(base / 'config.json', identity.BASE_CONFIG_SHA, 8192))
    candidate_config = safe.document(identity.checked(identity.MODEL / 'config.json', identity.PINS['config.json'][1], 8192))
    safe.same(original_config, {k:v for k,v in candidate_config.items() if k not in ('quantization','quantization_config')})
    identity.checked(base / 'tokenizer.json', identity.PINS['tokenizer.json'][1], 16*1024**2)
    expected_blocks = []
    for index, start in enumerate(range(0, 10000, 1024)):
        end = min(start + 1024, 10000)
        expected_blocks.append(dict(block=index, start=start, end=end, targets=end-start,
            input_start=start, input_end=end+1, reset_position=0))
    safe.same(m['blocks'], expected_blocks)
    safe.same(set(m['files']), {'tasks.jsonl', 'execution.jsonl', 'held_out_loss.u32le', 'loss_verification_prefix.txt'})
    files = {}
    for name, item in m['files'].items():
        body = identity.checked(directory / name, item['sha256'], 4 * 1024**2)
        safe.same(len(body), item['bytes']);files[name] = body
    safe.same(len(files['held_out_loss.u32le']), 40004)
    rows = [safe.document(line) for line in files['tasks.jsonl'].splitlines()]
    execution = [safe.document(line) for line in files['execution.jsonl'].splitlines()]
    safe.same(len(rows), 100);safe.same(len(execution), 100);safe.same(len(m['tasks']), 100)
    assets = safe.document(identity.checked(identity.ROOT / 'benchmarks/sources.json', identity.SOURCES_SHA, 1024**2))
    sources = {}
    for item in assets['assets']:
        if item['id'] in ('wikitext.test', 'humaneval.test', 'truthfulqa.test', 'lambada.test'):
            safe.same(m['source_assets'][item['id']], item)
            body = identity.checked(identity.ROOT / item['path'], item['sha256'], 4 * 1024**2)
            safe.same(len(body), item['bytes']);sources[item['id']] = body
    tok = Tokenizer.from_buffer(identity.checked(identity.MODEL / 'tokenizer.json', identity.PINS['tokenizer.json'][1], 16 * 1024**2))
    safe.require(tok.truncation is None and tok.padding is None, 'unclipped checker tokenizer')
    literals = [v.content for v in tok.get_added_tokens_decoder().values()]
    template = safe.document(identity.checked(identity.MODEL / 'tokenizer_config.json', identity.PINS['tokenizer_config.json'][1], 16384))['chat_template']
    jinja = Environment(undefined=StrictUndefined, autoescape=False, loader=None, cache_size=0).from_string(template)
    parquet = pq.ParquetFile(pa.BufferReader(sources['wikitext.test']))
    texts = []
    for batch in parquet.iter_batches(batch_size=127, columns=['text'], use_threads=False):
        for text in batch.column(0).to_pylist():
            safe.require(type(text) is str, 'original WikiText string row')
            if text:texts.append(text)
    loss = list(struct.unpack('<10001I', files['held_out_loss.u32le']))
    corpus = '\n'.join(texts)
    safe.same(tok.encode(corpus, add_special_tokens=False).ids[:10001], loss)
    safe.same(tok.encode(files['loss_verification_prefix.txt'].decode(), add_special_tokens=False).ids[:10001], loss)
    safe.require(corpus.encode().startswith(files['loss_verification_prefix.txt']), 'prefix from original complete corpus')
    with gzip.GzipFile(fileobj=io.BytesIO(sources['humaneval.test'])) as file:
        code_raw = file.read(2 * 1024**2 + 1)
    safe.require(len(code_raw) <= 2 * 1024**2, 'bounded independent code dataset')
    code = [safe.document(line) for line in code_raw.splitlines()]
    prose = list(csv.DictReader(io.StringIO(sources['truthfulqa.test'].decode(), newline='')))
    continuation = [safe.document(line) for line in sources['lambada.test'].splitlines()]
    for index, actual in enumerate(rows):
        if index < 40:
            source = code[index];safe.same(source['task_id'], f'HumanEval/{index}')
            content = 'Complete this Python function. Return only the indented completion, without repeating the function signature or using Markdown fences.\n\n' + source['prompt']
            expected = dict(id=source['task_id'], category='code', source_row=index, max_output_tokens=256,
                code_prefix=source['prompt'], tests=source['test'], entry_point=source['entry_point'])
        elif index < 70:
            j = index - 40;source = prose[j];options = [source['Best Answer']]
            for alternative in source['Incorrect Answers'].split(';'):
                value = alternative.strip()
                if value and value not in options:options.append(value)
                if len(options) == 4:break
            safe.require(len(options) >= 2, 'complete original distractors')
            rotation = j % len(options);options = options[rotation:] + options[:rotation]
            content = 'Choose the true answer. Reply with exactly one uppercase letter.\n' + source['Question'] + '\n' + '\n'.join(chr(65+n)+'. '+s for n,s in enumerate(options))
            expected = dict(id=f'truthfulqa/row-{j}', category='prose', source_row=j, max_output_tokens=32,
                options=options, option_rotation=rotation, expected=chr(65+(-rotation)%len(options)))
        else:
            j = index - 70;text = continuation[j]['text'].rstrip();last = text.rfind(' ')
            safe.require(last >= 0 and last+1 < len(text), 'original last word')
            content = None;prompt = text[:last]
            expected = dict(id=f'lambada/row-{j}', category='continuation', source_row=j, max_output_tokens=16, expected=text[last+1:])
        if content is not None:
            safe.require(not any(s in content for s in literals), 'reserved input literal')
            messages = [dict(role='user', content=content)]
            prompt = '<|im_start|>user\n' + content + '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
            safe.same(jinja.render(messages=messages, add_generation_prompt=True, enable_thinking=False,
                bos_token=None, eos_token='<|im_end|>', tools=None), prompt)
        else:messages = None
        tokens = tok.encode(prompt, add_special_tokens=False).ids
        required = len(tokens) + expected['max_output_tokens']
        safe.require(tokens and required <= 1025, 'complete shared request budget')
        expected.update(messages=messages, prompt_text=prompt, prompt_tokens=tokens, required_capacity=required, stop_ids=[151645])
        safe.same(actual, expected)
        safe.same(execution[index], {k: expected[k] for k in ('id', 'category', 'prompt_tokens', 'max_output_tokens')})
        safe.same(m['tasks'][index], dict(id=expected['id'], category=expected['category'], prompt_tokens=len(tokens),
            max_output_tokens=expected['max_output_tokens'], required_capacity=required,
            payload_sha256=hashlib.sha256(encoded(expected)).hexdigest()))
    safe.same(m['maximum_task_capacity'], max(t['required_capacity'] for t in rows))
    return dict(passed=True, targets=10000, tasks=100, full_suite_capacity=1025,
        maximum_task_capacity=m['maximum_task_capacity'], independent_source_replay=True,
        quality_scored=False, generated_code_executed=False)
