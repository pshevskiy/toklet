# Resident tiny Qwen3 Metal execution

24 October 2026. This follows [scalar Qwen3](mlx-qwen3-scalar.md) and uses the
unchanged [independent oracle](mlx-qwen3-oracle.md). Admission is limited to its
two-layer hidden32/query64/KV32 geometry, head width16, vocabulary64, capacity32,
ordinary RoPE and frozen epsilon/theta. Real Qwen3 checkpoints remain unsupported
by this GPU admission rule until independently qualified.

## Shared mechanisms and ownership

The [Qwen3 module](../src/mlx/qwen3.rs) continues to own family config validation.
The shared [Metal dense graph](../src/metal/mod.rs) now supports independent
hidden/query layouts and an immutable pair of Q/K norm tensors per applicable
layer. Ordinary legacy graphs have no pair. No new shader, trait hierarchy or
per-element architecture dispatch is introduced.

Q and attention context are `[tokens,query_heads*head_dim]`; hidden activations
remain `[tokens,hidden_size]`. Projection and MPS views explicitly distinguish
hidden→query and query→hidden products. Equal-width models reuse the previous
immutable result/multiply descriptors through planning-time references. Float
scratch and zero bias cover every admitted projection width. Checked admission
counts three hidden and three query buffers, KV, all other scratch and metadata.

The existing RMS kernel treats each Q/K head as a contiguous width16 row. One
learned Q norm vector is shared across all query heads; a separate learned K
norm vector is shared across all KV heads. RMS writes to rotary scratch; RoPE writes back to
Q/K. Snapshots capture raw projections, normalized heads and rotary outputs in
order. The chosen rotary K feeds persistent head-major KV, and the chosen rotary
Q feeds causal GQA. Tracked dependencies and retained buffers order the whole
chunk in one command, with completion checked before state publication. No extra
host wait, model expansion or transient head buffer is needed.

Both tied and independently owned untied outputs use the original packed tables.
Sessions use Reference attention. Explicit Online requests reject before session
resource allocation. Native bridge defaults now select Reference for Qwen3;
Qwen2 retains its existing default. The ABI and installed app are unchanged.

The [profiler](../src/metal/profile.rs) accounts for both additional RMS encoders
per normalized layer. This fixes the source-review finding that the old event
capacity would be too small. Profiling remains diagnostic and does not establish
performance qualification.

## Completed functional probes

On Apple M1 Pro, the explicit [GPU suite](../tests/mlx_qwen3_metal.rs) passes
1,170 snapshots and 204,288 scalar comparisons under the original frozen Half
gates. Both output variants cover five cases, rotary offsets and chunk sizes
1, 3 and full prompt. Four-step generation matches tied `[3,3,3,3]` and untied
`[24,36,11,51]`, with eight full distributions and sixteen exact token selections.
Resident weight buffers total 23,884 and 26,444 bytes respectively.

State checks cover pending tokens, exact-prefix reuse, edited replay, reset,
partial prefill, capacity/overflow recovery and both readback modes. Exact session
and diagnostic budgets pass; limit-minus-one rejects without changing admitted
state. Four profiled executions complete with 42/43 events, frozen logits and
one command per chunk.

The shared native bridge test passes all six existing/new tiny variants,
including both Qwen3 outputs, with scoring, model/session lifetime, cancellation
and pending-prefix checks. The CLI passes both Qwen3 variants with chunk1/3,
hidden stop semantics and explicit Online rejection.

The final [receipt](../benchmarks/results/2026-10-24-qwen3-metal-v1.json) records
Rust1.90 formatting, strict default/Metal Clippy, 161 CPU passes (13 ignored),
179 Metal-feature passes (95 ignored), three Qwen/two Llama explicit GPU passes,
and the new Qwen3 GPU/native/CLI probes. All three Qwen3 GPU tests also pass in
an optimized release build. Legacy Half and Q4 release checks pass their unchanged
zero-allocation gates. Protected release binaries, Swift and installed apps are
unchanged.

The extra legacy Half debug test fails its allocation assertion at 6,320
allocations over 20 steps. The exact parent `f2bb800` produces the same count;
the current optimized release test passes zero. This establishes no introduced
allocation regression for that probe. Objective-C debug signature verification
is a plausible contributor, but no allocation backtrace was collected to
attribute every allocation. The failed debug record stays retained. An initial
baseline snapshot omitted Cargo's declared examples and failed manifest parsing;
the complete second snapshot supplies the actual parent comparison. Neither
failure is counted as a passing zero-allocation debug result. No assertion,
numerical gate or tolerance was loosened.

E1 still requires an immutable real checkpoint, independent numerical/quality
references, tokenizer and thinking/chat semantics, app integration and physical
iPhone trials. These functional probes provide no throughput, speedup, energy,
real-model quality or physical-device qualification. Wider Hugging Face admission
and downloads remain H1–H4 in the [active plan](engine-expansion-plan.md).
