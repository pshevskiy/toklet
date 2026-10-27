"""Original unquantized Qwen3 graph under the frozen external quality policy.

No source arithmetic is patched. Benchmark greedy selection consumes raw source
logits, widened exactly to FP32, with lowest-ID ties. This avoids introducing
additional ties through a reduced-precision log-softmax sampling wrapper.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import struct
from importlib.metadata import version

import qwen3_quantized_source as source

ROOT = source.calibration.reference.ROOT
CONTRACT_SHA = '908e037de5b399d9e9a8dd09aa9fcc568f67abb714b06508672deac6bb8ac529'
BASELINE = ROOT / 'models/Qwen3-0.6B-a9c98e6'
PINS = {
    'config.json': '660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd',
    'model.safetensors': 'f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b',
    'tokenizer.json': 'aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4',
    'tokenizer_config.json': 'bbc2c089e3bef8753f63348c10595767be8998bc6ca7c7ebcaada346d401fba8',
}


def load_inputs(folder, mode):
    manifest = source.calibration.reference.load(Path(folder) / 'manifest.json', CONTRACT_SHA)
    if (manifest['format'] != 'toklet-qwen3-quality-contract-v1' or manifest['loss_targets'] != 10000
            or manifest['full_suite_capacity'] != 1025 or manifest['stops'] != [151645]
            or manifest['unquantized_baseline']['weight_sha256'] != PINS['model.safetensors']):
        raise ValueError('Exact frozen full-quality contract required')
    if mode in ('probe', 'task_probe'):
        values = [9707, 11, 1879, 0] if mode == 'probe' else [dict(id='authored-probe',category='probe',
            prompt_tokens=[9707,11,1879,0],max_output_tokens=4)]
        return manifest, values, None
    name = {'loss': 'held_out_loss.u32le', 'tasks': 'execution.jsonl'}.get(mode)
    if name is None:
        raise ValueError('Explicit loss/tasks/probe mode required')
    item = manifest['files'][name]
    body = source.runtime.safe.read(Path(folder) / name, 4 * 1024**2)
    if len(body) != item['bytes'] or hashlib.sha256(body).hexdigest() != item['sha256']:
        raise ValueError('Frozen complete quality input changed')
    if mode == 'loss':
        if len(body) != 40004:
            raise ValueError('All10000 targets required')
        ids = list(struct.unpack('<10001I', body))
        if any(t >= 151936 for t in ids):
            raise ValueError('Loss vocabulary differs')
        return manifest, ids, item['sha256']
    cases = [source.runtime.safe.document(line) for line in body.splitlines()]
    expected = [f'HumanEval/{i}' for i in range(40)] + [f'truthfulqa/row-{i}' for i in range(30)] + [f'lambada/row-{i}' for i in range(30)]
    if len(cases) != 100 or [c['id'] for c in cases] != expected:
        raise ValueError('All100 frozen tasks in order required')
    for index, case in enumerate(cases):
        category, cap = ('code', 256) if index < 40 else ('prose', 32) if index < 70 else ('continuation', 16)
        if (set(case) != {'id', 'category', 'prompt_tokens', 'max_output_tokens'} or case['category'] != category
                or case['max_output_tokens'] != cap or not case['prompt_tokens']
                or len(case['prompt_tokens']) + cap > 1025
                or any(type(i) is not int or not 0 <= i < 151936 for i in case['prompt_tokens'])):
            raise ValueError('Complete task shape/capacity differs')
    return manifest, cases, item['sha256']


def memory_plan():
    # Separate loading/inference phases, conditional worst-case reserves rather
    # than a claim that mapped/shared allocations are physically duplicated.
    weight_file = 1503300328
    initialization = 2384199680  # Same static tied595M parameter constructor.
    cache_capacity = 1280       # Frozen original KVCache.step256, cap1025 rounded.
    cache = 2 * 28 * 8 * cache_capacity * 128 * 2
    load = dict(original_storage=weight_file, fp32_constructor=initialization,
        conditional_storage_copy=weight_file, python_frameworks=512 * 1024**2)
    inference = dict(original_storage=weight_file, graph_workspace=2 * 1024**3,
        full_output_max_fp32=1025 * 151936 * 4, cache_and_growth_overlap=2 * cache,
        host_tiles_and_journal=160 * 1024**2, python_frameworks=512 * 1024**2)
    admitted = max(sum(load.values()), sum(inference.values()))
    if admitted > 6 * 1024**3:
        raise ValueError('Unquantized quality admission exceeds6GiB')
    return dict(load=load, inference=inference, admitted=admitted, limit=6 * 1024**3,
        phases_are_separate=True, conditional_reserves_are_not_measured_duplicates=True)


def row_statistics(row, target=None):
    import numpy as np
    if (row.dtype != np.dtype('<f4') or len(row.shape) != 1 or not 2 <= row.size <= 151936
            or not np.isfinite(row).all()
            or target is not None and (type(target) is not int or not 0 <= target < row.size)):
        raise ValueError('Finite source distribution and valid target required')
    x = row.astype(np.float64);greedy = int(np.argmax(x))
    second = x.copy();second[greedy] = -np.inf;runner_up = int(np.argmax(second))
    shifted = x - x[greedy]
    nll = None if target is None else float(-shifted[target] + math.log(float(np.sum(np.exp(shifted), dtype=np.float64))))
    return dict(greedy=greedy, second=runner_up, margin=float(x[greedy]-x[runner_up]),
        negative_log_likelihood=nll, target_logit_f32_bits=None if target is None else int(row.view('<u4')[target]),
        logits_sha256=hashlib.sha256(row.tobytes()).hexdigest())


def require_output(shape, dtype, rows):
    if list(shape) != [1, rows, 151936] or dtype.removeprefix('mlx.core.') != 'bfloat16':
        raise ValueError('Exact original BF16 [1,input_count,151936] output required')


def retain_failed_output(folder, rows, pieces, report):
    folder = Path(folder);folder.mkdir()
    records = report.setdefault('failed_outputs', {})
    sizes = {'source-bits.bin': rows*151936*2, 'widened.f32le.bin': rows*151936*4}
    for name, size in sizes.items():
        records[str(folder/name)] = dict(bytes=0, intended_bytes=size, sha256=hashlib.sha256().hexdigest())
    for name in sizes:
        record = records[str(folder/name)];digest = hashlib.sha256()
        with (folder/name).open('xb') as file:
            for body in pieces(name):source.write_piece(file,body,record,digest)


def score_block(pieces, block, start, end, ids, emit):
    losses = [];seen = 0
    for first, raw, widened in pieces:
        if first != seen or widened.ndim != 2 or widened.shape[1] != 151936 or len(raw) != len(widened)*151936*2:
            raise ValueError('Contiguous complete source tile required')
        for local, row in enumerate(widened):
            position = first+local
            if position == end-start:continue
            if position > end-start:raise ValueError('Too many source output rows')
            index = start+position;target = ids[index+1];statistics = row_statistics(row,target)
            losses.append(statistics['negative_log_likelihood'])
            emit(dict(event='loss',target_index=index,block=block,position=position,target_token=target,
                source_bits_sha256=hashlib.sha256(raw[local*151936*2:(local+1)*151936*2]).hexdigest(),**statistics))
        seen += len(widened)
    if seen != end-start+1 or len(losses) != end-start:
        raise ValueError('Complete input rows and every target required before block completion')
    emit(dict(event='block_done',block=block,target_start=start,target_end=end,targets=end-start,
        evaluated_tokens=seen,pending_tokens=0,total_negative_log_likelihood=math.fsum(losses)))
    return losses


def generate_task(case, distribution, commit, emit):
    visible = [];sampled = 0;committed = len(case['prompt_tokens']);reason = dict(kind='output_limit')
    for step in range(case['max_output_tokens']):
        statistics = distribution();token = statistics['greedy'];sampled += 1
        emit(dict(event='token',id=case['id'],index=step,token_id=token,visible=token!=151645,**statistics))
        if token == 151645:reason = dict(kind='stop_token',token_id=token);break
        visible.append(token)
        if step+1 < case['max_output_tokens']:commit(token);committed += 1
    return dict(reason=reason,sampled_tokens=sampled,emitted_tokens=len(visible),history_tokens=len(case['prompt_tokens'])+sampled,
        evaluated_tokens=committed,pending_tokens=1), visible


def execute(folder, mode, journal, report, endpoint):
    """External caller admits the cold runtime, retains report even on failure."""
    if any(os.environ.get(k) != v for k,v in source.ENVIRONMENT.items()):
        raise ValueError('Frozen offline source environment required')
    availability = source.isolation_profile()
    if {name:version(name) for name in source.VERSIONS} != source.VERSIONS:
        raise ValueError('Pinned source runtime required')
    contract, inputs, input_sha = load_inputs(folder, mode)
    source.model_inventory(BASELINE, PINS)
    config = source.calibration.reference.load(BASELINE / 'config.json', PINS['config.json'])
    if config.get('model_type') != 'qwen3' or not config.get('tie_word_embeddings') or any(k in config for k in ('quantization','quantization_config','model_file')):
        raise ValueError('Original unquantized tied Qwen3 required')
    admission = memory_plan()
    import mlx.core as mx
    import numpy as np
    from mlx.utils import tree_flatten
    from mlx_lm.models.qwen3 import Model, ModelArgs
    from mlx_lm.models.cache import KVCache, make_prompt_cache
    from mlx_lm.utils import load_model
    import native_usage
    endpoint()
    report.update(complete=False, primary_error=None, cleanup_errors=[], mode=mode,
        full_quality_workload=mode in ('loss','tasks'), quality_qualified=False,
        input_manifest_sha256=CONTRACT_SHA, input_sha256=input_sha, model_pins=PINS,
        availability_profile=availability, memory_admission=admission, versions=source.VERSIONS,
        source_profile=dict(dtype_override=False, strict=True, lazy=False, trust_remote_code=False,
            loss_batch=1, loss_cache=None, loss_full_blocks=True, task_prefill_chunk=128,
            task_cache='original KVCache step256 BF16', kv_quantization=False,
            greedy='host FP64 comparison of exact-widened raw source logits, lowest ID ties',
            stop_ids=[151645], final_sample_pending=True), memory=[], journal=dict(bytes=0, events=0))
    model = output = cache = tile = original = widened = raw = wide = row = None
    digest = hashlib.sha256();primary = None

    def usage(phase):
        observation = dict(phase=phase, mlx=source.memory(mx), native=native_usage.usage())
        report['memory'].append(observation);source.admit_observed_memory(observation)

    def tiles(value):
        if value.ndim != 3 or value.shape[0] != 1 or value.shape[2] != 151936 or value.dtype != mx.bfloat16:
            raise ValueError('Original BF16 source output shape/dtype differs')
        for start in range(0, value.shape[1], 32):
            tile = value[0, start:start+32, :]
            original = np.array(tile.view(mx.uint16), copy=True)
            widened = np.array(tile.astype(mx.float32), copy=True)
            raw, wide = original.astype('<u2', copy=False).tobytes(), widened.astype('<f4', copy=False).tobytes()
            check = source.verify_widening(raw, wide, 'bfloat16')
            if check['nonfinite_values'] or check['finite_widening_mismatches']:
                # Keep the complete returned output before rejecting. Re-reading
                # GPU data is confined to this failure path, with the same tile bound.
                failures = Path(journal).parent / ('nonfinite-output-'+str(report['journal']['events']))
                def pieces(raw_name):
                    for first in range(0,value.shape[1],32):
                        part = value[0,first:first+32,:]
                        yield np.array(part.view(mx.uint16) if raw_name=='source-bits.bin' else part.astype(mx.float32),copy=True).tobytes()
                retain_failed_output(failures,value.shape[1],pieces,report)
                raise ValueError('Source output is nonfinite or widening differs')
            yield start, raw, widened

    try:
        mx.set_default_device(mx.gpu);report['previous_cache_limit'] = mx.set_cache_limit(0);mx.random.seed(0)
        model, loaded = load_model(BASELINE, lazy=False, strict=True, model_config=None,
            get_model_classes=lambda config: (Model, ModelArgs), trust_remote_code=False)
        if loaded != config or not isinstance(model, Model) or not model.args.tie_word_embeddings or KVCache.step != 256:
            raise ValueError('Original loaded class/config/cache policy differs')
        model.eval()
        parameters = [dict(name=n,shape=list(v.shape),dtype=str(v.dtype),bytes=v.nbytes) for n,v in tree_flatten(model.parameters())]
        if (not parameters or sum(p['bytes'] for p in parameters) != 1192099840
                or any(p['dtype'] != 'mlx.core.bfloat16' for p in parameters)
                or any(hasattr(v,'bits') for _,v in model.named_modules())):
            raise ValueError('Entire baseline parameter storage must be original BF16 and unquantized')
        report['parameters'] = parameters
        report['tied_output'] = 'original sanitize omits duplicate lm_head; Model uses embedding.as_linear'
        report['device'] = mx.device_info();usage('loaded')
        with Path(journal).open('xb') as stream:
            def emit(event):
                body = (json.dumps(event,separators=(',',':'),allow_nan=False)+'\n').encode()
                source.write_piece(stream,body,report['journal'],digest);report['journal']['events'] += 1
            emit(dict(event='ready',format='toklet-qwen3-bf16-quality-events-v1',mode=mode,
                full_quality_workload=mode in ('loss','tasks'),input_manifest_sha256=CONTRACT_SHA,
                targets=len(inputs)-1 if mode not in ('tasks','task_probe') else 0,
                tasks=len(inputs) if mode in ('tasks','task_probe') else 0))
            if mode not in ('tasks','task_probe'):
                targets = len(inputs)-1;losses = []
                for block,start in enumerate(range(0,targets,1024)):
                    end = min(start+1024,targets)
                    report['active_block'] = dict(block=block,target_start=start,target_end=end,completed=False)
                    output = model(mx.array([inputs[start:end+1]],dtype=mx.uint32),cache=None)
                    require_output(output.shape,str(output.dtype),end-start+1);mx.eval(output)
                    losses.extend(score_block(tiles(output),block,start,end,inputs,emit))
                    report['active_block']['completed'] = True
                    stream.flush();output = row = widened = raw = None;mx.clear_cache();usage('block-'+str(block))
                if len(losses) != targets or mode == 'loss' and targets != 10000:
                    raise ValueError('All declared loss targets required before completion')
                emit(dict(event='done',mode=mode,full_quality_workload=mode=='loss',targets=targets,tasks=0,
                    total_negative_log_likelihood=math.fsum(losses),mean_negative_log_likelihood=math.fsum(losses)/targets))
            else:
                for case in inputs:
                    emit(dict(event='task_started',id=case['id']))
                    cache = make_prompt_cache(model)
                    if len(cache)!=28 or any(type(c) is not KVCache or c.offset!=0 for c in cache):
                        raise ValueError('Fresh original per-layer cache required')
                    committed = 0
                    def call(ids):
                        nonlocal committed
                        value = model(mx.array([ids],dtype=mx.uint32),cache=cache)
                        require_output(value.shape,str(value.dtype),len(ids));mx.eval(value)
                        committed += len(ids)
                        if any(c.offset!=committed or c.keys.dtype!=mx.bfloat16 or c.values.dtype!=mx.bfloat16
                               or c.keys.shape[2]>1280 or c.values.shape[2]>1280 for c in cache):
                            raise ValueError('Source BF16 cache position/storage/admission differs')
                        return value
                    prompt = case['prompt_tokens']
                    for first in range(0,len(prompt),128):output = call(prompt[first:first+128])
                    def distribution():
                        nonlocal output,row,widened,raw
                        _, raw, widened = next(tiles(output[:,-1:,:]));row = widened[0]
                        return row_statistics(row)
                    def commit(token):
                        nonlocal output,row,widened,raw
                        output = row = widened = raw = None;output = call([token])
                    task_report,visible = generate_task(case,distribution,commit,emit)
                    if committed != task_report['evaluated_tokens']:
                        raise ValueError('Source pending/visible output counts differ')
                    emit(dict(event='task_done',id=case['id'],category=case['category'],prompt_tokens=len(prompt),
                        requested_output_tokens=case['max_output_tokens'],visible_token_ids=visible,
                        report=task_report))
                    stream.flush();output = cache = row = widened = raw = None;mx.clear_cache();usage(case['id'])
                emit(dict(event='done',mode=mode,full_quality_workload=mode=='tasks',targets=0,tasks=len(inputs)))
        report['complete'] = True
    except BaseException as error:
        primary = error;report['primary_error'] = type(error).__name__+': '+str(error)
    finally:
        output = cache = model = tile = original = widened = raw = wide = row = None
        try:mx.synchronize();mx.clear_cache()
        except BaseException as error:report['cleanup_errors'].append(type(error).__name__+': '+str(error))
        report['journal']['sha256'] = digest.hexdigest()
        try:source.model_inventory(BASELINE,PINS);endpoint();source.isolation_profile();usage('cleanup')
        except BaseException as error:report['cleanup_errors'].append(type(error).__name__+': '+str(error))
        if report['cleanup_errors']:report['complete'] = False
    if primary is not None:raise primary
    if report['cleanup_errors']:raise ValueError('Original source cleanup failed')
    return report
