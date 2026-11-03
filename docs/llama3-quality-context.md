# Llama quality diagnostic context

The explicit Rust `Model::load_mlx_llama3_quality_diagnostic` route admits the
pinned MLX Llama 3.2 1B 8-bit artifact through 1,025 positions. It authenticates
config/index/container delivery and requires complete delivery after assembly,
then discards that identity. It cannot supply the audited app/codec identity.
Ordinary inspection/loading and the managed Mac app retain their 512 ceiling;
there is no new C export, catalog entry or package adoption.

This changes cold admission and dispatch bounds, not numerical kernels. The
existing scaled-RoPE frequency, tensor extent and packed-index checks apply.
Each scaled dispatch receives the validated model ceiling and checks the entire
absolute position range. The resident rotary owner and its accounting retain
the same layout. Immutable packed weights stay model-owned; separate per-layer
Half K/V buffers remain session-owned `[kv_heads, capacity, head_dim]` arrays.
Reference attention, compensated normalization and affine projections apply.

The explicit `llama3_quality_context` example consumes the frozen 1,025 raw IDs
from the existing loss screen. It compares complete FP32 numerical rows at
prefixes 17, 512, 513, 1,024 and 1,025 under chunk sizes 32 and 128 and independently
reset prefixes. Signed zeros compare equally; every other bit must agree. The
17/512 rows also match the ordinary route using the same chunk32 profile. A full
1,025-token prefix must survive append/set-token overflow unchanged, including
history, committed/pending counts and final logits. Ordinary/audited513 and
diagnostic1026 loads reject. Diagnostic snapshot/readback memory is included in
admission after enabling diagnostics. The two full model owners run sequentially.

These are consistency and resource checks. Original-reference long-row alignment
and the frozen 10,000-target/100-task adoption gates remain separate requirements.
The ordinary app window is not increased by this experiment.

Mac Apple M1 Pro results: all 15 comparisons and overflow preservation pass.
Peak RSS is 1,271,480,320 bytes; maximum admission including diagnostic buffers
and the 64 MiB app allowance is 1,727,887,552 bytes. These are distinct quantities,
both below 6 GiB. All 549 default and 657 Metal tests, strict Clippy in both configurations,
formatting and release build pass. Initial cold-check failures remain recorded.
[Evidence](../benchmarks/results/2026-11-03-llama3-quality-context-v1.json).
