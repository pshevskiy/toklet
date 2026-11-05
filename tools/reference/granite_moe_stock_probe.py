"""Bounded Granite MoE stock decisions and final-layer router witnesses."""
import hashlib
import json
from pathlib import Path
import sys


def sha(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main(plan_path, expected, output):
    assert sys.dont_write_bytecode and sys.flags.no_user_site
    assert sha(plan_path) == expected
    plan = json.loads(Path(plan_path).read_text())
    output = Path(output)
    output.mkdir()

    def guard():
        for path, digest in plan["pins"].items():
            assert sha(path) == digest, path

    guard()
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.utils import load_model
    import numpy as np

    mx.set_default_device(mx.gpu)
    limit = 6 * 1024**3
    mx.set_memory_limit(limit)
    mx.set_cache_limit(0)
    model, config = load_model(Path(plan["model"]), lazy=True, strict=True, trust_remote_code=False)
    assert config["model_type"] == "granitemoe" and config["tie_word_embeddings"]
    assert "quantization" not in config and len(model.layers) == 24
    assert (config["num_local_experts"], config["num_experts_per_tok"]) == (32, 8)
    parameters = tree_flatten(model.parameters())
    assert len(parameters) == 242 and all(value.dtype == mx.bfloat16 for _, value in parameters)
    mx.eval(model.parameters())
    cache = make_prompt_cache(model)
    router = model.layers[-1].block_sparse_moe.router
    router_type = type(router)
    original = router_type.__call__
    routes = []

    def capture(instance, hidden):
        ids, gates = original(instance, hidden)
        if instance is router:
            scores = instance.layer(hidden)[0, -1, :].astype(mx.float32)
            routes.append((scores, ids[0, -1, :], gates[0, -1, :]))
        return ids, gates

    rows, ids = [], []
    inputs = list(plan["prompt"])
    consumed = 0
    router_type.__call__ = capture
    try:
        for index in range(plan["sample_cap"]):
            row = model(mx.array([inputs], dtype=mx.int32), cache=cache)[0, -1, :]
            assert len(routes) == 1
            scores, route_ids, gates = routes.pop()
            mx.eval(row, [c.state for c in cache], scores, route_ids, gates)
            consumed += len(inputs)
            assert row.shape == (49155,) and all(c.offset == consumed for c in cache)
            values = np.asarray(row.astype(mx.float32)).astype("<f4", copy=False)
            assert np.isfinite(values).all()
            decision = int(mx.argmax(row).item())
            assert decision == int(np.argmax(values))
            raw = values.tobytes()
            (output / f"row-{index}.f32le").write_bytes(raw)
            scores, route_ids, gates = [np.asarray(value) for value in (scores, route_ids, gates)]
            assert scores.shape == (32,) and gates.shape == route_ids.shape == (8,)
            assert len(set(route_ids)) == 8 and min(route_ids) >= 0 and max(route_ids) < 32
            cutoff = np.sort(scores)[-8]
            assert all(scores[route_ids] >= cutoff)
            top = np.lexsort((np.arange(len(values)), -values))[:2]
            rows.append(dict(index=index, id=decision, prefix_tokens=consumed,
                             sha256=hashlib.sha256(raw).hexdigest(), top_two_ids=top.tolist(),
                             margin=float(values[top[0]])-float(values[top[1]]),
                             router=dict(scores=scores.tolist(), ids=route_ids.tolist(), gates=gates.tolist())))
            ids.append(decision)
            if decision in plan["stops"]:
                break
            inputs = [decision]
    finally:
        router_type.__call__ = original
    guard()
    assert mx.get_peak_memory() <= limit
    result = dict(passed=True, ids=ids, rows=rows, consumed_parameters=len(parameters),
                  parameter_bytes=sum(v.nbytes for _, v in parameters), mlx_peak_bytes=mx.get_peak_memory())
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(ids=ids, mlx_peak_bytes=result["mlx_peak_bytes"])))


if __name__ == "__main__":
    main(*sys.argv[1:])
