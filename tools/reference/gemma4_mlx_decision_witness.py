"""Stock strict-loaded same-checkpoint MLX decision witness; no task execution."""
from pathlib import Path
import ctypes
import hashlib
from importlib.metadata import version
import json
import os
import platform
import resource
import sys
import sysconfig
import time
from unittest.mock import patch
import gemma_mlx_decisions as decisions
import gemma4_affine_reference as source

ROOT = source.ROOT
LIMIT = 6*1024**3


def sha(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def runtime_paths():
    """Cold code/native closure, enumerated by the owner before launch."""
    site = Path(sysconfig.get_path('purelib')).resolve()
    stdlib = Path(sysconfig.get_path('stdlib')).resolve()
    base = Path(sys.base_prefix).resolve()
    paths = {Path(sys.executable).resolve(), base/'Python', base/'Resources/Python.app/Contents/MacOS/Python'}
    for folder in [site, stdlib]:
        paths.update(p.resolve() for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts
            and (p.suffix in ['.py', '.so', '.dylib', '.metallib', '.pth'] or any(x.endswith('.dist-info') for x in p.parts)))
    paths.update(p.resolve() for p in base.rglob('*.dylib'))
    for folder in [ROOT/'tools/reference', ROOT/'tools/bench']:
        paths.update(p.resolve() for p in folder.glob('*.py'))
    paths.add(ROOT/'tools/experiments/native_usage.py')
    return sorted(paths)


def native_images():
    library = ctypes.CDLL(None)
    count, name = library._dyld_image_count, library._dyld_get_image_name
    count.argtypes, count.restype = [], ctypes.c_uint32
    name.argtypes, name.restype = [ctypes.c_uint32], ctypes.c_char_p
    return {str(Path(name(i).decode()).resolve()) for i in range(count())
            if not name(i).decode().startswith(('/System/', '/usr/lib/'))}


def run(plan_path, plan_sha, output):
    output = Path(output).resolve()
    output.mkdir()
    result = dict(passed=False, decisions=[], source='stock MLX same-checkpoint',
                  task_grading_executed=False, task_gate_rejection_waived=False)
    try:
        assert sys.flags.no_user_site and sys.dont_write_bytecode and sys.pycache_prefix
        assert not any(name == 'mlx' or name.startswith('mlx.') or name == 'numpy' or name.startswith('numpy.') for name in sys.modules)
        assert sha(plan_path) == plan_sha
        plan = json.loads(Path(plan_path).read_text())
        assert plan['format'] == 'toklet-gemma-mlx-decisions-plan-v1'
        assert [c['index'] for c in plan['cases']] == [1, 24, 40, 43]
        assert {name: version(name) for name in plan['versions']} == plan['versions']
        assert set(plan['runtime']) == {str(p) for p in runtime_paths()}
        pins = {**plan['runtime'], **plan['inputs']}
        def guard():
            assert all(sha(Path(name)) == item['sha256'] and Path(name).stat().st_size == item['bytes']
                       for name, item in pins.items())
        guard()
        import mlx.core as mx
        import mlx.nn as nn
        import numpy as np
        from mlx.utils import tree_flatten
        from mlx_lm.utils import load_model
        from mlx_lm.models.gemma4 import Model
        sys.path.insert(0, str(ROOT/'tools/experiments'))
        import native_usage
        mx.set_default_device(mx.gpu)
        mx.set_memory_limit(LIMIT)  # MLX documents this as a guideline, not a hard cap.
        mx.set_cache_limit(0)
        def memory():
            mx.synchronize()
            values = dict(rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                physical_peak_bytes=native_usage.usage()['lifetime_max_phys_footprint'],
                mlx_peak_bytes=mx.get_peak_memory(), mlx_active_bytes=mx.get_active_memory())
            assert all(0 <= value <= LIMIT for value in values.values())
            return values
        expected = decisions.inventory()
        origins = {}
        loads = []
        strict_calls = []
        old_load, old_weights = mx.load, nn.Module.load_weights
        def load(path, *args, **kwargs):
            assert Path(path).resolve() == source.MODEL/'model.safetensors'
            values = old_load(path, *args, **kwargs)
            loads.append(str(Path(path).resolve()))
            for name, value in values.items():
                origins[id(value)] = name
            return values
        def weights(model, values, strict=True):
            if type(model) is Model:
                assert strict is True
                supplied = {name: dict(shape=list(value.shape), dtype=str(value.dtype).split('.')[-1])
                            for name, value in values}
                assert set(supplied) == set(expected)
                names = dict(bfloat16='BF16', uint32='U32')
                for name, value in values:
                    assert origins[id(value)] == name
                    assert list(value.shape) == expected[name]['shape']
                    assert names[str(value.dtype).split('.')[-1]] == expected[name]['dtype']
                strict_calls.append(supplied)
            return old_weights(model, values, strict=strict)
        clock = time.perf_counter()
        with patch.object(mx, 'load', load), patch.object(nn.Module, 'load_weights', weights):
            model, config = load_model(source.MODEL, lazy=True, strict=True, trust_remote_code=False)
        assert loads == [str(source.MODEL/'model.safetensors')] and len(strict_calls) == 1
        assert config['quantization'] == dict(group_size=64, bits=4, mode='affine')
        parameters = dict(tree_flatten(model.parameters()))
        assert set(parameters) == set(expected)
        assert all(origins[id(value)] == name for name, value in parameters.items())
        quantized = [(name, module) for name, module in tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module)
                     if isinstance(module, (nn.QuantizedLinear, nn.QuantizedEmbedding))]
        assert len(quantized) == 278
        assert all(module.bits == 4 and module.group_size == 64 and module.mode == 'affine' for _, module in quantized)
        assert model.language_model.tie_word_embeddings is True
        mx.eval(model.parameters())
        result.update(load_seconds=time.perf_counter()-clock, load_memory=memory(),
            consumed_tensors=len(parameters), consumed_storage_bytes=sum(value.nbytes for value in parameters.values()),
            strict_loader=True, quantized_modules=len(quantized), tied_head=True,
            inventory=strict_calls[0], previous_kvs=model.language_model.model.previous_kvs)
        for case in plan['cases']:
            prefix = case['prompt_ids']+case['agreed_ids']
            import struct
            assert len(prefix) == case['prefix_tokens'] and hashlib.sha256(struct.pack('<'+'I'*len(prefix), *prefix)).hexdigest() == case['prefix_sha256']
            cache = model.make_cache()
            assert len(cache) == 15
            steps = []
            start = time.perf_counter()
            for index, tokens in enumerate([case['prompt_ids']]+[[t] for t in case['agreed_ids']]):
                logits = model(mx.array([tokens], dtype=mx.int32), cache=cache)
                row = logits[0, -1, :]
                mx.eval(row, [c.state for c in cache])
                assert row.shape == (262144,)
                assert all(c.offset == len(case['prompt_ids'])+index for c in cache)
                greedy = int(mx.argmax(row).item())
                raw = np.asarray(row.astype(mx.float32)).astype('<f4', copy=False).tobytes()
                steps.append(dict(index=index, greedy_id=greedy, dtype=str(row.dtype), row_sha256=hashlib.sha256(raw).hexdigest()))
            stats = decisions.statistics(raw, case['original_id'], case['native_id'], greedy)
            path = output/('row-'+str(case['index'])+'.f32le')
            with path.open('xb') as file:
                file.write(raw)
            item = dict(id=case['id'], index=case['index'], prefix_tokens=len(prefix),
                original_id=case['original_id'], native_id=case['native_id'], stock_dtype=str(row.dtype),
                steps=steps, statistics=stats, seconds=time.perf_counter()-start,
                cache_classes=[type(c).__name__ for c in cache], row_file=str(path.relative_to(ROOT)), memory=memory())
            result['decisions'].append(item)
            print(json.dumps(dict(event='case_complete', id=case['id'], agrees_with=stats['agrees_with'])), flush=True)
            del cache, logits, row
        assert set(dict(tree_flatten(model.parameters()))) == set(expected)
        guard()
        images = native_images()
        assert images <= set(plan['runtime']), sorted(images-set(plan['runtime']))
        modules = {}
        generated = {}
        for name, module in list(sys.modules.items()):
            path = getattr(module, '__file__', None)
            if not isinstance(path, str):
                continue
            if name in ['torch.ops', 'torch.classes']:
                path = str(Path(sys.modules['torch'].__file__).resolve().parent/path)
            path = str(Path(path).resolve())
            if name == '_remote_module_non_scriptable':
                # Torch creates this deterministic helper from its pinned template.
                instantiator = sys.modules['torch.distributed.nn.jit.instantiator']
                template = sys.modules['torch.distributed.nn.jit.templates.remote_module_template']
                assert path == str((Path(instantiator.INSTANTIATED_TEMPLATE_DIR_PATH)/(name+'.py')).resolve())
                expected_text = template.get_remote_module_template(True).format(
                    assign_module_interface_cls='module_interface_cls = None', args='*args', kwargs='**kwargs',
                    arg_types='*args, **kwargs', arrow_and_return_type='', arrow_and_future_return_type='',
                    jit_script_decorator='')
                assert Path(path).read_text() == expected_text
                saved = output/(name+'.py')
                saved.write_text(expected_text)
                generated[name] = dict(path=path, sha256=sha(saved),
                    template=str(Path(template.__file__).resolve()),
                    instantiator=str(Path(instantiator.__file__).resolve()),
                    saved=str(saved.relative_to(ROOT)))
                continue
            assert path in pins, (name, path)
            modules[name] = path
        result.update(passed=True, actual_native_images=sorted(images), actual_modules=modules,
            generated_modules=generated,
            final_memory=memory(), runtime_manifest_sha256=plan_sha, hardware=platform.platform())
    except BaseException as error:
        result['failure'] = dict(type=type(error).__name__, message=str(error)[:4096])
    with (output/'result.json').open('x') as file:
        json.dump(result, file, indent=2, allow_nan=False)
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path)
    parser.add_argument('plan_sha256')
    parser.add_argument('fresh_output', type=Path)
    args = parser.parse_args()
    result = run(args.plan, args.plan_sha256, args.fresh_output)
    print(json.dumps({k: result.get(k) for k in ['passed', 'failure']}, indent=2))
    raise SystemExit(0 if result['passed'] else 1)
