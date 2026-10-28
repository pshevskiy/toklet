"""Prospective8192 original-BF16 cache profile; explicit execution only.

Frozen4096 source is imported unchanged for its numerical graph and cold helpers.
The finite orchestration below retains its old pass structure, but owns8192-only
selection, admission and receipt identity. Outer owner supplies runtime closure,
timeouts, measured6GiB cap and persistence of report on all failures.
"""
import gc
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import re
import struct

import qwen3_context4096_source as original
import qwen3_context8192_profile as profile

baseline = original.baseline
source = original.source
CAPACITY, VOCAB, PROFILE = profile.CAPACITY, profile.VOCAB, profile.PROFILE
ENDPOINTS, PROBE_RANGES, QUALITY_RANGES = profile.ENDPOINTS, profile.PROBE_RANGES, profile.QUALITY_RANGES
original_model_classes = original.original_model_classes
scored = original.scored
adapted_forward = original.adapted_forward
stock_capacity_after_update = original.stock_capacity_after_update
cache_shapes = original.cache_shapes
retain_output = original.retain_output
release_pass = original.release_pass


def dependencies():
    """Checked original helpers plus actual new profile identity; no model read."""
    pinned = ((original, 'f267bcbbac72bd7d34db14f4dff14a8bb07b8982bd6e2d4790b0294c348b8b08'),
              (baseline, 'f622fda4d218e6ba121d31ddf9f60bc3cbe842b63669e76dd00e1af5f0b7ab34'),
              (source, '67570e2f0b34685e74bd4e815b848381d91ab18eba65c04d73b51e9b4bc4cc70'))
    result = {}
    for module, expected in pinned:
        actual = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError('Frozen original source helper differs: ' + module.__name__)
        result[module.__name__] = actual
    result[profile.__name__] = hashlib.sha256(Path(profile.__file__).read_bytes()).hexdigest()
    return result


def schedule(mode):
    if mode not in ('equivalence', 'quality'):
        raise ValueError('Explicit equivalence or quality mode required')
    start = 0
    for end in ENDPOINTS if mode == 'equivalence' else (CAPACITY,):
        while start < end:
            next_position = min(start + 8, end)
            yield start, next_position
            start = next_position


def memory_plan():
    load = dict(original_storage=1503300328, fp32_constructor=2384199680,
                conditional_storage_copy=1503300328, python_frameworks=512 * 1024**2)
    inference = dict(original_storage=1503300328, graph_workspace=2 * 1024**3,
                     cache_and_update_overlap=2 * 2 * 28 * 8 * CAPACITY * 128 * 2,
                     chunk_output_max_fp32=8 * VOCAB * 4,
                     host_tiles_and_journal=160 * 1024**2, python_frameworks=512 * 1024**2)
    admitted = max(sum(load.values()), sum(inference.values()))
    if (sum(inference.values()) != 6239337192 or sum(load.values()) != 5927671248
            or source.LIMIT != 6442450944 or admitted > source.LIMIT):
        raise ValueError('Source phase admission exceeds 6 GiB')
    return dict(load=load, inference=inference, admitted=admitted, limit=source.LIMIT,
                phases_are_separate=True, cache_passes_are_sequential=True,
                conditional_reserves_are_not_measured_duplicates=True)


def load_inputs(folder, mode, manifest_sha256):
    if not profile.digest(manifest_sha256) or mode not in ('equivalence', 'quality'):
        raise ValueError('Frozen manifest and explicit finite mode required')
    folder = Path(folder)
    manifest = source.calibration.reference.load(folder / 'manifest.json', manifest_sha256)
    profile.validate(manifest)
    name = 'A.u32le' if mode == 'equivalence' else 'Q.u32le'
    item = manifest['files'][name]
    raw = source.runtime.safe.read(folder / name, CAPACITY * 4)
    if len(raw) != item['bytes'] or hashlib.sha256(raw).hexdigest() != item['sha256']:
        raise ValueError('Complete frozen8192 input bytes required')
    ids = list(struct.unpack('<8192I', raw))
    profile.validate_ids(ids, CAPACITY)
    if name == 'A.u32le' and hashlib.sha256(raw[:16384]).hexdigest() != profile.F2_A:
        raise ValueError('Actual historical A4096 token prefix differs')
    return manifest, ids[:1025] if mode == 'equivalence' else ids, item['sha256']



def adapted_cache_class(mx, stock_class):
    """One ordinary nonrotating cache, with evaluated full state before views."""
    class FixedCache(stock_class):
        def __init__(self):
            super().__init__()
            self.step = CAPACITY  # Instance only; stock class remains step256.

        def update_and_fetch(self, keys, values):
            rows = keys.shape[2]
            if not 1 <= rows <= 8 or self.offset + rows > CAPACITY:
                raise ValueError('Fixed BF16 cache capacity/chunk exceeded before update')
            if (keys.shape != values.shape or list(keys.shape[:2]) != [1, 8]
                    or keys.shape[3] != 128 or keys.dtype != mx.bfloat16 or values.dtype != mx.bfloat16):
                raise ValueError('Fixed BF16 cache input geometry/dtype differs')
            # Do not retain the lazy prefix views returned by the original method.
            super().update_and_fetch(keys, values)
            mx.eval(self.keys, self.values)
            mx.synchronize()
            return self.keys_and_values()

    return FixedCache


def validate_cache(cache, offset, adapted, bf16, capacity, before=None, emit=None):
    valid = (len(cache) == 28 and 0 < capacity <= CAPACITY
             and (not adapted or capacity == CAPACITY)
             and all(state.offset == offset and state.step == (CAPACITY if adapted else 256)
                     and all(value is not None and list(value.shape) == [1, 8, capacity, 128]
                             and value.dtype == bf16 for value in (state.keys, state.values)) for state in cache))
    if not valid:
        if emit is not None:
            emit(dict(event='cache_validation_failed', expected_offset=offset, expected_capacity=capacity,
                      expected_step=CAPACITY if adapted else 256, expected_dtype=str(bf16),
                      before=before, after=cache_shapes(cache)))
        raise ValueError('Source cache position/storage/dtype differs')


def require_equivalence(path, digest, manifest_sha256, producer_sha256):
    if path is None or not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest):
        raise ValueError('Authenticated completed equivalence receipt required before quality')
    receipt = source.calibration.reference.load(Path(path), digest)
    if (receipt.get('complete') is not True or receipt.get('mode') != 'equivalence'
            or receipt.get('profile') != PROFILE or receipt.get('model_pins') != baseline.PINS
            or receipt.get('input_manifest_sha256') != manifest_sha256
            or receipt.get('producer_sha256') != producer_sha256
            or receipt.get('dependencies') != dependencies()
            or receipt.get('equivalence_passed') is not True or receipt.get('cleanup_errors') != []):
        raise ValueError('Equivalence receipt candidate/profile/completion differs')


def consume_rows(bits, widened, start, end, mode, label, masked, ids, expected, emit, retain):
    """Bounded host readback, fixed selection, and failure evidence before rejection."""
    import numpy as np
    if (mode not in ('equivalence', 'quality') or mode == 'quality' and not masked
            or not 0 <= start < end <= len(ids)
            or end - start > 8 or bits.dtype != np.dtype('<u2') or widened.dtype != np.dtype('<f4')
            or bits.shape != (end - start, VOCAB) or widened.shape != bits.shape):
        emit(dict(event='invalid_readback', label=label, start=start, end=end,
                  bits_shape=list(bits.shape), widened_shape=list(widened.shape)))
        raise ValueError('Complete bounded BF16/FP32 chunk required')
    check = source.verify_widening(bits.tobytes(), widened.tobytes(), 'bfloat16')
    if check['nonfinite_values'] or check['finite_widening_mismatches']:
        retain(label + '-invalid-' + str(start), bits.tobytes(), widened.tobytes())
        emit(dict(event='invalid_output', label=label, start=start, **check))
        raise ValueError('Source finite exact widening failed')
    ranges = PROBE_RANGES if mode == 'equivalence' else QUALITY_RANGES
    rows, losses = {}, []
    for local, position in enumerate(range(start, end)):
        selected = scored(position, ranges)
        if masked and not selected:
            continue
        raw, wide = bits[local].tobytes(), widened[local].tobytes()
        target = ids[position + 1] if mode == 'quality' else None
        statistics = baseline.row_statistics(widened[local], target)
        row_hash = hashlib.sha256(raw).hexdigest()
        emit(dict(event='row', label=label, position=position, scored=selected,
                  source_bits_sha256=row_hash, target_token=target,
                  target_bf16_bits=None if target is None else int(bits[local, target]),
                  exact_widening=True, **statistics))
        if mode == 'equivalence' and selected:
            retain(label + '-row-' + str(position), raw, wide)
        if expected is not None and expected.get(position) != row_hash:
            if not selected:
                retain(label + '-mismatch-' + str(position), raw, wide)
            emit(dict(event='row_mismatch', label=label, position=position,
                      expected=expected.get(position), actual=row_hash))
            raise ValueError('Exact BF16 source profile row mismatch')
        rows[position] = row_hash
        if target is not None:
            losses.append(statistics['negative_log_likelihood'])
    return rows, losses


def execute(folder, mode, journal, report, endpoint, manifest_sha256,
            equivalence_receipt=None, equivalence_sha256=None):
    """Retain pre-runtime admission failures as well as execution failures."""
    try:
        return _execute(folder, mode, journal, report, endpoint, manifest_sha256,
                        equivalence_receipt, equivalence_sha256)
    except BaseException as error:
        report['complete'] = False
        report['primary_error'] = type(error).__name__ + ': ' + str(error)
        raise


def _execute(folder, mode, journal, report, endpoint, manifest_sha256,
             equivalence_receipt, equivalence_sha256):
    """Run one finite mode. `journal` must be a new file in an existing folder."""
    producer_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report.update(complete=False, primary_error=None, cleanup_errors=[], mode=mode,
                  profile=PROFILE, producer_sha256=producer_sha, dependencies=dependencies(), model_pins=baseline.PINS,
                  input_manifest_sha256=manifest_sha256, equivalence_passed=False,
                  full_quality_workload=False, quality_qualified=False,
                  memory_admission=memory_plan(), memory=[], outputs={}, journal=dict(bytes=0, events=0))
    if any(os.environ.get(key) != value for key, value in source.ENVIRONMENT.items()):
        raise ValueError('Frozen offline source environment required')
    report['availability_profile'] = source.isolation_profile()
    if {name: version(name) for name in source.VERSIONS} != source.VERSIONS:
        raise ValueError('Pinned source runtime required')
    _, ids, input_sha = load_inputs(folder, mode, manifest_sha256)
    report['input_sha256'] = input_sha
    if mode == 'quality':
        require_equivalence(equivalence_receipt, equivalence_sha256, manifest_sha256, producer_sha)
        report['equivalence_receipt_sha256'] = equivalence_sha256
    source.model_inventory(baseline.BASELINE, baseline.PINS)
    config = source.calibration.reference.load(baseline.BASELINE / 'config.json', baseline.PINS['config.json'])
    if (config.get('model_type') != 'qwen3' or not config.get('tie_word_embeddings')
            or any(key in config for key in ('quantization', 'quantization_config', 'model_file'))):
        raise ValueError('Original tied unquantized Qwen3 required')
    import mlx.core as mx
    import numpy as np
    from mlx.utils import tree_flatten
    from mlx_lm.models.qwen3 import Model, create_attention_mask
    from mlx_lm.models.cache import KVCache
    from mlx_lm.utils import load_model
    import native_usage

    model = None
    primary = None
    journal_digest = hashlib.sha256()
    root = Path(journal).parent

    def usage(phase):
        observation = dict(phase=phase, mlx=source.memory(mx), native=native_usage.usage())
        report['memory'].append(observation)
        source.admit_observed_memory(observation)

    def retain(name, raw, wide):
        retain_output(root, report['outputs'], name, raw, wide)

    def cache_record(cache, position):
        values = []
        for layer, state in enumerate(cache):
            record = dict(layer=layer, position=position)
            for name in ('keys', 'values'):
                # At most 1025 * 8 * 128 BF16 values per host read (2.01 MiB).
                raw = np.array(getattr(state, name)[:, :, :position, :].view(mx.uint16), copy=True)
                record[name + '_sha256'] = hashlib.sha256(raw.astype('<u2', copy=False).tobytes()).hexdigest()
            values.append(record)
        return values

    try:
        endpoint()
        mx.set_default_device(mx.gpu)
        report['previous_cache_limit'] = mx.set_cache_limit(0)
        mx.random.seed(0)
        model, loaded = load_model(baseline.BASELINE, lazy=False, strict=True, model_config=None,
                                   get_model_classes=original_model_classes, trust_remote_code=False)
        if loaded != config or not isinstance(model, Model) or not model.args.tie_word_embeddings or KVCache.step != 256:
            raise ValueError('Original model/config/cache policy differs')
        model.eval()
        parameters = [dict(name=name, shape=list(value.shape), dtype=str(value.dtype), bytes=value.nbytes)
                      for name, value in tree_flatten(model.parameters())]
        if (sum(item['bytes'] for item in parameters) != 1192099840
                or any(item['dtype'] != 'mlx.core.bfloat16' for item in parameters)
                or any(hasattr(module, 'bits') for _, module in model.named_modules())):
            raise ValueError('Entire baseline must retain original BF16 unquantized parameters')
        report.update(parameters=parameters, device=mx.device_info(), versions=source.VERSIONS,
                      source_profile=dict(batch=1, chunk=8, dtype_override=False, kv_quantization=False,
                                          adapted_cache_step=CAPACITY, stock_cache_step=256,
                                          mask='original once before layer0 update', layer_barriers=True,
                                          full_head_for_partially_scored_chunks=True))
        mx.synchronize()
        usage('loaded')
        fixed_class = adapted_cache_class(mx, KVCache)
        with Path(journal).open('xb') as stream:
            def emit(event):
                body = (json.dumps(event, separators=(',', ':'), allow_nan=False) + '\n').encode()
                source.write_piece(stream, body, report['journal'], journal_digest)
                report['journal']['events'] += 1
                stream.flush()

            def run_pass(label, adapted, masked, expected=None):
                cache = output = None
                pass_error = None
                expected_capacity = 0
                rows, caches, losses = {}, {}, []
                ranges = PROBE_RANGES if mode == 'equivalence' else QUALITY_RANGES
                try:
                    cache = [fixed_class() if adapted else KVCache() for _ in range(28)]
                    emit(dict(event='pass_started', label=label, ranges=ranges, masked=masked))
                    for start, end in schedule(mode):
                        before = cache_shapes(cache)
                        expected_capacity = (CAPACITY if adapted else
                                             stock_capacity_after_update(expected_capacity, start, end - start))
                        project = not masked or any(scored(position, ranges) for position in range(start, end))
                        inputs = mx.array([ids[start:end]], dtype=mx.uint32)
                        output = (adapted_forward(model, inputs, cache, mx, create_attention_mask, project)
                                  if adapted else model(inputs, cache=cache))
                        if output is not None:
                            emit(dict(event='output_returned', label=label, start=start, end=end,
                                      shape=list(output.shape), dtype=str(output.dtype)))
                            baseline.require_output(output.shape, str(output.dtype), end - start)
                            mx.eval(output)
                        mx.synchronize()
                        validate_cache(cache, end, adapted, mx.bfloat16, expected_capacity, before, emit)
                        emit(dict(event='chunk_completed', label=label, start=start, end=end, projected=project,
                                  cache_capacity=expected_capacity))
                        if output is not None:
                            bits = np.array(output.view(mx.uint16), copy=True).reshape(end - start, VOCAB).astype('<u2', copy=False)
                            widened = np.array(output.astype(mx.float32), copy=True).reshape(end - start, VOCAB).astype('<f4', copy=False)
                            chunk_rows, chunk_losses = consume_rows(bits, widened, start, end, mode, label,
                                                                   masked, ids, None if expected is None else expected[0], emit, retain)
                            rows.update(chunk_rows)
                            losses.extend(chunk_losses)
                            del bits, widened
                        output = inputs = None
                        if mode == 'equivalence' and end in ENDPOINTS:
                            record = cache_record(cache, end)
                            emit(dict(event='cache_endpoint', label=label, position=end, layers=record))
                            if expected is not None and expected[1][end] != record:
                                for layer, (actual, previous) in enumerate(zip(record, expected[1][end], strict=True)):
                                    if actual != previous:
                                        for name in ('keys', 'values'):
                                            value = getattr(cache[layer], name)[:, :, :end, :]
                                            raw = np.array(value.view(mx.uint16), copy=True).astype('<u2', copy=False).tobytes()
                                            wide = np.array(value.astype(mx.float32), copy=True).astype('<f4', copy=False).tobytes()
                                            retain(f'{label}-cache-{end}-{layer}-{name}', raw, wide)
                                        emit(dict(event='cache_mismatch', label=label, position=end, layer=layer,
                                                  expected=previous, actual=actual))
                                        break
                                raise ValueError('Exact valid-prefix BF16 cache mismatch')
                            caches[end] = record
                        if end in ENDPOINTS or end == CAPACITY:
                            usage(label + '-' + str(end))
                    expected_positions = ({position for position in range(len(ids)) if scored(position, ranges)}
                                          if masked else set(range(len(ids))))
                    if set(rows) != expected_positions:
                        raise ValueError('Complete prescribed source row coverage required')
                    emit(dict(event='pass_done', label=label, rows=len(rows), committed=len(ids), pending=0))
                    return rows, caches, losses
                except BaseException as error:
                    pass_error = error
                    raise
                finally:
                    output = cache = None
                    release_pass(mx, usage, label, report['cleanup_errors'], pass_error)

            if mode == 'equivalence':
                stock = run_pass('stock-full', False, False)
                adapted = run_pass('adapted-full', True, False, stock)
                run_pass('adapted-masked', True, True, adapted)
                report['equivalence_passed'] = True
                emit(dict(event='done', mode=mode, full_rows=1025, masked_rows=6, cache_endpoints=list(ENDPOINTS)))
            else:
                _, _, losses = run_pass('quality', True, True)
                if len(losses) != 1024:
                    raise ValueError('Exactly 1024 frozen deep targets required')
                report['quality'] = dict(targets=1024, score_rows=[7167, 8191],
                                         total_negative_log_likelihood=math.fsum(losses),
                                         mean_negative_log_likelihood=math.fsum(losses) / 1024)
                emit(dict(event='done', mode=mode, **report['quality']))
        report['complete'] = True
    except BaseException as error:
        primary = error
        report['primary_error'] = type(error).__name__ + ': ' + str(error)
    finally:
        model = None
        try:
            gc.collect()
            mx.synchronize()
            mx.clear_cache()
        except BaseException as error:
            report['cleanup_errors'].append(type(error).__name__ + ': ' + str(error))
        report['journal']['sha256'] = journal_digest.hexdigest()
        try:
            source.model_inventory(baseline.BASELINE, baseline.PINS)
            load_inputs(folder, mode, manifest_sha256)
            endpoint()
            source.isolation_profile()
            if dependencies() != report['dependencies']:
                raise ValueError('Source/helper endpoint identity differs')
            usage('cleanup')
        except BaseException as error:
            report['cleanup_errors'].append(type(error).__name__ + ': ' + str(error))
        if report['cleanup_errors']:
            report['complete'] = False
    if primary is not None:
        raise primary
    if report['cleanup_errors']:
        raise ValueError('BF16 context source cleanup failed')
    return report
