"""Pinned stock MLX Qwen1.5 decision rows; no remote code or task grading."""
from pathlib import Path
import hashlib
import json
import sys


def sha(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def main(plan_path, expected, output):
    assert sys.dont_write_bytecode and sys.flags.no_user_site
    assert sha(plan_path) == expected
    plan = json.loads(Path(plan_path).read_text())
    output = Path(output)
    output.mkdir()
    def guard():
        for name, digest in plan['pins'].items():
            assert sha(name) == digest, name
    guard()
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np
    from mlx.utils import tree_flatten
    from mlx_lm.utils import load_model
    from mlx_lm.models.cache import make_prompt_cache
    mx.set_default_device(mx.gpu)
    mx.set_memory_limit(6*1024**3)  # Guideline only; external owner measures peaks.
    mx.set_cache_limit(0)
    model, config = load_model(Path(plan['model']), lazy=True, strict=True, trust_remote_code=False)
    assert config['quantization'] == dict(group_size=64, bits=8)
    assert config['tie_word_embeddings'] and len(model.layers) == 28
    parameters = tree_flatten(model.parameters())
    assert all(str(value.dtype) in ['mlx.core.float16', 'mlx.core.uint32'] for _, value in parameters)
    quantized = [m for _, m in tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module)
                 if isinstance(m, (nn.QuantizedLinear, nn.QuantizedEmbedding))]
    assert len(quantized) == 197 and all(m.bits == 8 and m.group_size == 64 for m in quantized)
    mx.eval(model.parameters())
    cache = make_prompt_cache(model)
    assert len(cache) == 28
    inputs = list(plan['prompt'])
    consumed = 0
    ids, rows = [], []
    for index in range(8):
        logits = model(mx.array([inputs], dtype=mx.int32), cache=cache)
        row = logits[0, -1, :]
        mx.eval(row, [c.state for c in cache])
        consumed += len(inputs)
        assert row.shape == (151936,) and all(c.offset == consumed for c in cache)
        values = np.asarray(row.astype(mx.float32)).astype('<f4', copy=False)
        assert np.isfinite(values).all()
        id = int(mx.argmax(row).item())
        assert id == int(np.argmax(values))  # First/lowest ID wins a tie.
        raw = values.tobytes()
        (output/('row-'+str(index)+'.f32le')).write_bytes(raw)
        top = np.lexsort((np.arange(len(values)), -values))[:2]
        rows.append(dict(index=index, id=id, dtype=str(row.dtype), prefix_tokens=consumed,
                         sha256=hashlib.sha256(raw).hexdigest(), top_two_ids=top.tolist(),
                         margin=float(values[top[0]])-float(values[top[1]])))
        ids.append(id)
        if id in plan['stops']:
            break
        inputs = [id]
    guard()
    assert mx.get_peak_memory() <= 6*1024**3
    result = dict(passed=True, ids=ids, rows=rows, consumed_parameters=len(parameters),
                  parameter_bytes=sum(v.nbytes for _, v in parameters), mlx_peak_bytes=mx.get_peak_memory(),
                  task_quality_qualified=False, app_adopted=False)
    (output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main(Path(sys.argv[1]), sys.argv[2], sys.argv[3])
