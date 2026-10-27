"""Freeze complete Qwen3 inputs before scoring; no model or generated code runs."""
from pathlib import Path
import csv
import gzip
import hashlib
from importlib.metadata import version
import io
import json
import struct
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/reference'))
import check_gemma_hybrid_tiny as safe
import text_reference_runtime as runtime
import gemma_quality_runtime
import quality_contract_policy as policy

MODEL = ROOT / 'models/MLX-Qwen3-0.6B-8bit-11de968'
BASELINE = ROOT / 'models/Qwen3-0.6B-a9c98e6'
REVISION = '11de96878523501bcaa86104e3c186de07ff9068'
BASE_REVISION = 'a9c98e602b9d36d2a2f7ba1eb0f5f31e4e8e5143'
PINS = {
    'config.json': (937, '9f387dcb1bf045cadb34a36e67569abd73c00e5322603df2b19d0c0f8cc9d240'),
    'tokenizer.json': (11422654, 'aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4'),
    'tokenizer_config.json': (10219, '8e7dd4250db58b2a29792562d727b2715dc0fe79a94b446caa4c07db39eceba6'),
}
WEIGHT_SHA = '3ad5c96e0a476d4eee48e9f525d0ee0a3f1830b3e5b8e17b441de0661e921397'
BASE_SHA = 'f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b'
BASE_CONFIG_SHA = '660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd'
SOURCES_SHA = 'dc2f82fb50b3f1f763c4757a992429b9122f3628c049709bdd7cb2252c69ea28'
OLD_CONTRACT_SHA = '3f09dcf70f6d9620ce5172dcbaa6428f3487a3edf412f2bfce57937faab72c66'


def sha(body):
    return hashlib.sha256(body).hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def checked(path, expected, maximum):
    body = safe.read(path, maximum)
    safe.require(sha(body) == expected, 'input identity changed: ' + str(path))
    return body


def sources():
    return gemma_quality_runtime.sources([
        Path(__file__).resolve(), Path(policy.__file__).resolve(), Path(safe.__file__).resolve(),
        ROOT / 'tools/bench/test_qwen3_quality_inputs.py',
        ROOT / 'tools/bench/check_qwen3_quality.py',
    ])


def render(content, literals):
    safe.require(type(content) is str and content and not any(s in content for s in literals),
                 'plain user text without reserved literals required')
    return '<|im_start|>user\n' + content + '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'


def candidate_sources():
    # Freeze the actual numerical implementation and worker, independently of
    # the later commit ID. A run must authenticate these bytes before scoring.
    files = [ROOT / 'Cargo.toml', ROOT / 'Cargo.lock', ROOT / 'examples/qwen3_quality.rs',
             ROOT / 'examples/support/loss.rs', *sorted((ROOT / 'src').rglob('*.rs')),
             *sorted((ROOT / 'src').rglob('*.metal'))]
    return {str(p.relative_to(ROOT)): sha(safe.read(p, 4 * 1024**2)) for p in files}


def produce(output):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    from jinja2 import Environment, StrictUndefined
    safe.same({name: version(name) for name in ('numpy', 'pyarrow')}, {'numpy': '2.3.2', 'pyarrow': '20.0.0'})
    safe.same(np.__version__, '2.3.2')
    source_manifest = safe.document(checked(ROOT / 'benchmarks/sources.json', SOURCES_SHA, 1024**2))
    old = safe.document(checked(ROOT / 'benchmarks/contract-v1/manifest.json', OLD_CONTRACT_SHA, 1024**2))
    safe.same(old['quality']['mean_nll_increase_limit_nats_per_token'], 0.05)
    safe.same(old['quality']['task_pass_rate_loss_limit_percentage_points'], 2.0)
    raw = {}
    source_pins = {}
    for item in source_manifest['assets']:
        if item['id'] in ('wikitext.test', 'humaneval.test', 'truthfulqa.test', 'lambada.test'):
            body = checked(ROOT / item['path'], item['sha256'], 4 * 1024**2)
            safe.same(len(body), item['bytes']);raw[item['id']] = body;source_pins[item['id']] = item
    safe.require(len(raw) == 4, 'all four original quality datasets required')
    assets = {name: checked(MODEL / name, pin, size) for name, (size, pin) in PINS.items()}
    config = safe.document(assets['config.json'])
    base_config = safe.document(checked(BASELINE / 'config.json', BASE_CONFIG_SHA, 8192))
    safe.same({k: v for k, v in config.items() if k not in ('quantization', 'quantization_config')}, base_config)
    checked(BASELINE / 'tokenizer.json', PINS['tokenizer.json'][1], PINS['tokenizer.json'][0])
    before = candidate_sources()
    parquet = pq.ParquetFile(pa.BufferReader(raw['wikitext.test']))
    text_rows = parquet.read(columns=['text'], use_threads=False).column(0).to_pylist()
    safe.require(all(type(t) is str for t in text_rows), 'WikiText string rows required')
    corpus = '\n'.join(t for t in text_rows if t)
    safe.require(len(corpus.encode()) <= 16 * 1024**2, 'bounded complete corpus required')
    with gzip.GzipFile(fileobj=io.BytesIO(raw['humaneval.test'])) as file:
        code = file.read(2 * 1024**2 + 1)
    safe.require(len(code) <= 2 * 1024**2, 'bounded HumanEval decompression')
    tasks = policy.make_tasks([safe.document(line) for line in code.splitlines()],
        list(csv.DictReader(io.StringIO(raw['truthfulqa.test'].decode(), newline=''))),
        [safe.document(line) for line in raw['lambada.test'].splitlines()])
    tokenizer = Tokenizer.from_buffer(assets['tokenizer.json'])
    safe.require(tokenizer.truncation is None and tokenizer.padding is None, 'unclipped tokenization required')
    literals = [v.content for v in tokenizer.get_added_tokens_decoder().values()]
    safe.same(len(literals), 26)
    template_text = safe.document(assets['tokenizer_config.json'])['chat_template']
    template = Environment(undefined=StrictUndefined, autoescape=False, loader=None, cache_size=0).from_string(template_text)
    ids = tokenizer.encode(corpus, add_special_tokens=False).ids[:10001]
    safe.same(len(ids), 10001)
    prefix = corpus.encode()[:65536]
    while True:
        try:
            prefix_text = prefix.decode();break
        except UnicodeDecodeError as error:
            safe.require(error.end == len(prefix), 'only final UTF8 boundary may be incomplete');prefix = prefix[:error.start]
    safe.same(tokenizer.encode(prefix_text, add_special_tokens=False).ids[:10001], ids)
    payload = []
    execution = []
    for task in tasks:
        task = dict(task)
        if task['category'] == 'continuation':
            messages = None;prompt = task.pop('raw_prompt')
        else:
            content = task.pop('user_content');messages = [dict(role='user', content=content)]
            prompt = render(content, literals)
            safe.same(template.render(messages=messages, add_generation_prompt=True, enable_thinking=False,
                                      bos_token=None, eos_token='<|im_end|>', tools=None), prompt)
        tokens = tokenizer.encode(prompt, add_special_tokens=False).ids
        required = len(tokens) + task['max_output_tokens']
        safe.require(tokens and all(0 <= t < 151936 for t in tokens) and required <= 1025,
                     'complete untruncated task must fit diagnostic window')
        task.update(messages=messages, prompt_text=prompt, prompt_tokens=tokens, required_capacity=required, stop_ids=[151645])
        payload.append(task)
        execution.append({k: task[k] for k in ('id', 'category', 'prompt_tokens', 'max_output_tokens')})
    safe.same(candidate_sources(), before)
    files = {'tasks.jsonl': b''.join(encoded(v) for v in payload),
             'execution.jsonl': b''.join(encoded(v) for v in execution),
             'held_out_loss.u32le': struct.pack('<10001I', *ids),
             'loss_verification_prefix.txt': prefix}
    safe.require(all(len(body) <= 4 * 1024**2 for body in files.values()), 'bounded full workload')
    record = dict(format='toklet-qwen3-quality-contract-v1', candidate=dict(repository='mlx-community/Qwen3-0.6B-8bit',
        revision=REVISION, weight_sha256=WEIGHT_SHA, numerical_source_sha256=before,
        precision='packed8/group64 weights, Half activations/KV, Compensated normalization, Reference attention',
        numerical_gate_status='C512 rejected unchanged', diagnostic_capacity=1025, app_limits_changed=False),
        unquantized_baseline=dict(repository='Qwen/Qwen3-0.6B', revision=BASE_REVISION, weight_sha256=BASE_SHA,
            config_sha256=BASE_CONFIG_SHA, config_equal_excluding_quantization=True,
            tokenizer_sha256=PINS['tokenizer.json'][1], identical_frozen_input_ids=True,
            lineage='contemporary official revision; converter records repository but no source revision'),
        source_assets=source_pins, tokenizer_assets={n: dict(bytes=s, sha256=h) for n, (s, h) in PINS.items()},
        tokenization=dict(add_special_tokens=False, code_prose='pinned_plain_enable_thinking_false',
            loss_continuation='raw_no_added_BOS', corpus_join='newline_join_nonempty_original_rows',
            generation_defaults_are_hard_limits=False),
        quality_policy=dict(mean_nll_increase_limit_nats_per_token=0.05, task_pass_rate_loss_limit_percentage_points=2.0,
            source_manifest_sha256=SOURCES_SHA, original_contract_sha256=OLD_CONTRACT_SHA),
        scoring=policy.SCORING, stops=[151645], blocks=policy.blocks(), loss_targets=10000,
        files={name: dict(bytes=len(body), sha256=sha(body)) for name, body in files.items()},
        tasks=[dict(id=t['id'], category=t['category'], prompt_tokens=len(t['prompt_tokens']),
            max_output_tokens=t['max_output_tokens'], required_capacity=t['required_capacity'],
            payload_sha256=sha(encoded(t))) for t in payload],
        maximum_task_capacity=max(t['required_capacity'] for t in payload), full_suite_capacity=1025,
        quality_scored=False, generated_code_executed=False, full_model_window_qualified=False)
    output.mkdir()
    for name, body in {**files, 'manifest.json': encoded(record)}.items():
        with (output / name).open('xb') as file:
            if file.write(body) != len(body):raise OSError('short quality contract write')
    return record
