# Qwen2.5-1.5B MLX8: bounded Mac engine execution

5 November 2026. The existing dense Rust/Metal path executes a second Qwen2.5
geometry without a numerical implementation change. This is **Mac512 diagnostic
engine coverage**, not managed Chat discovery, full numerical fidelity, held-out
quality, full declared32768 context or iPhone qualification.

The immutable artifact is `mlx-community/Qwen2.5-1.5B-Instruct-8bit` at
`c7e1ff25efda4a2c8f7ee312380557fb42e145a2`, payload1,640,414,038bytes, SHA256
`8b76d08a5a70261b9ba4da5c06ac28bf09ee848e2322bae901e84764b7bba8e8`.
Bounded HTTP headers/index agree on all732tensors:197U32 and535F16. Affine8/
group64, tied embedding/head, H1536/I8960,28layers,12Q/2KV/head128 already fit
native structural admission. The complete payload authenticates after a
conservative pre-acquisition screen; exact native plans then precede loading.

The frozen33-token chat-formatted prompt asks “What is 2 + 2? Reply with a single
digit.” No BOS is prepended. The8-slot greedy cap stops at151645 or151643;
lowest vocabulary ID wins ties. Stock strict-loaded MLX and native execution
both produce19 (“4”) followed by hidden stop151645. Compare each complete
151936-value row at its identical prefix before either sampled token is fed.

| Decision | Stock margin | Native margin | Maximum absolute row difference |
| --- | ---: | ---: | ---: |
| “4” |10.796875 |10.763645 |0.0823650 |
| Stop |4.625 |4.636282 |0.0786624 |

Exact greedy agreement passes. Full-row differences are descriptive; these two
rows do not establish a numerical fidelity envelope or original-weight quality.
Stock uses the checkpoint's F16 parameters directly; native uses Compensated
normalization, Reference attention, FullLogits and F16 KV.

Native chunk1 and32 produce bit-identical initial/final-generation rows and
matching pending/committed history. Both actually reach512 with the frozen
corpus prefix; token513 returns the exact capacity error while preserving
published logits and committed history. Boundary distributions agree across
partitions. Reset reproduces the original prompt row. Each session drops before
its replacement, retaining one immutable model.

| Memory | Bytes |
| --- | ---: |
| Resident native weights |1,640,334,262 |
| Native chunk32 admitted model/session |1,747,603,994 |
| Native process-lifetime RSS peak |1,715,470,336 |
| Greatest observed native physical high-water |1,705,772,832 |
| Stock process-lifetime RSS peak |1,301,856,256 |
| Greatest observed stock physical high-water |1,980,451,096 |

Numerical admission is4GiB and the separate process trial policy is6GiB. Physical
values are the greatest external lifetime high-water observations before exit;
RSS peaks come from Darwin `wait4`. Both owned processes exit0 with absent groups.
Stock source/model pins guard both endpoints; native release build and saved
worker inputs/results identify its code. No prospective native binary seal or
performance benchmark is claimed.

Formatting, strict all-target Clippy,550 default/658 Metal tests, release library
and probe builds pass. Independent saved-row/margin/hash/state-result replay
passes. The initial prompt-preparation assertion caught Transformers returning
`BatchEncoding`; explicit `return_dict=False` fixes it before any model execution.
The failed attempt remains. Astra approves the code and scoped execution evidence.

Next qualify SmolLM3's per-layer rotary bypass using the same dense blocks.
[Machine evidence](../benchmarks/results/2026-11-05-qwen15-engine-execution-v1.json).
