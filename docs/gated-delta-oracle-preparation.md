# Independent gated-delta reference

This freezes the first unmasked recurrent-attention witnesses for E4/L4 in the
[active plan](engine-expansion-plan.md). It precedes any native recurrence target.
It does not implement Qwen3.5, a recurrent session or a long-context model.
See the [mechanism investigation](mlx-qwen35-design.md) for the remaining graph.

## Mathematical contract

[The producer](../tools/reference/gated_delta_oracle.py) imports no Toklet or MLX
implementation. FP32 inputs feed independent NumPy FP64 normalization, gates,
recurrence, readout and state. The upstream source is authenticated as inert
bytes during export; upstream execution is a separate comparison.

Q/K are `[tokens,key_heads,key_dim]`, V is
`[tokens,value_heads,value_dim]`, scalar gates are `[tokens,value_heads]` and
state is `[value_heads,value_dim,key_dim]` with contiguous key dimension.
A value head `h` reads key head `h // (value_heads // key_heads)`; divisibility
is mandatory. Normalization adds epsilon to the sum of squares and folds one
`key_dim**-0.5` readout scale into Q. Decay precedes state prediction, then a
rank-one correction, then readout through Q. Initial state remains immutable.

Five geometries cover an empty prefix, one scalar head, grouped heads with
unequal key/value dimensions, a 32-wide key and a 128×128 state. Nonzero initial
state, zero/epsilon-dominated normalization and saturated gates are explicit.
Scalar decay and update gates lie in `[0,1]`. Vector gates, negative eigenvalues,
masks, reduced-precision state and graph composition remain unsupported here.
The pinned upstream operation/kernel paths disagree on inactive masked readout;
a future mask witness must preserve that distinction rather than assume equality.

## Frozen format and admission

[The manifest](../tests/fixtures/gated-delta-v1/manifest.json) describes
little-endian FP32 inputs and FP64 expected arrays in one binary. Names, shapes,
dtypes, eight-byte-aligned offsets, byte counts and per-array identities are
explicit. The entire binary is bounded at 32 MiB.

[The stdlib reader](../tools/reference/gated_delta_fixture.py) requires both
external manifest/binary identities before interpreting values. It checks exact
case/array inventory, numerical contract, dimensions, ordered accounted spans,
zero padding, finite values and hashes. Duplicate/nonfinite JSON, Boolean shapes,
symlink roots/files and oversized inputs reject. Mathematical fixture scope flags
stay false for upstream execution/parity and native/device qualification; the
separate [acceptance receipt](../benchmarks/results/2026-10-24-gated-delta-oracle-v1.json)
binds actual upstream observations without rewriting that producer's scope.

## Separate upstream observations

[The comparison driver](../tools/reference/gated_delta_upstream.py) executes only
the captured MLX-LM source at revision `0d4d3d04` with recorded SHA256. Before any
numerical import it admits an external exact runtime manifest covering the
interpreter, producer/reader, NumPy/MLX noncache package files, native libraries,
Metal library and distribution metadata. Selected and imported package paths
must match those files. Already imported numerical modules and preexisting
numerical bytecode in the owned fresh cache reject.

CPU and GPU are separate requests. Each has 36 fixed comparisons: normalized
Q/K, decay/beta, full output, every prefix state, final state, and separate
chunked output/final state for four nonempty cases. Prefix calls restart from
the immutable initial state; chunked calls carry the returned FP32 state.
The existing source tolerance remains absolute `1e-5` plus relative `1e-4`.
The empty prefix is a mathematical no-op; upstream empty-call support is not
asserted. CPU requests select operations. GPU requests select operations for
narrow keys, the generic kernel for width 32 and the packed policy for the
128-wide eligible case. This is pinned-source dispatch accounting, not hardware
tracing or native Toklet qualification.

[The capture owner](../tools/experiments/gated_delta_reference.py) reuses the
existing process-group helper, creates exclusive output/log/cache directories,
guards source/runtime/input identities and retains raw arrays/logs. Returned
ownership rows stay unchanged. Helper exceptions retain an explicit attempt
with unknown PID/exit/cleanup rather than inventing a successful result.
Primary, capture and endpoint errors remain separate. Cooperative TERM/INT
attempts bounded-data publication; hard SIGKILL or stalled native code can still
prevent it, in which case missing reports remain explicit failures.

## Checks and scope

Twenty-three portable tests and four orchestration tests pass. They cover
hand-computed arithmetic, head mapping, epsilon/scale, immutable state, chunk
agreement, malformed artifacts, external identities, shadow imports and
child/helper failure retention. Fresh owned export/verification passes. Separate
pinned CPU and GPU runs each pass all 36 comparisons without changing tolerances.
All checks were rerun on the final source identities after removing draft labels.

The initial sandbox attempt failed before MLX import completed because Metal
was unavailable. Its logs and missing-report failure remain retained; a fresh
run with actual device access passed. Earlier passing captures with draft labels
also remain separate. No failure is converted into a numerical pass.

Manual Astra review corrected normalization intermediate overflow, symlink
publication roots, import resolution, partial-observation retention, interruption
publication and temporary-path aliases in tests. Final source/evidence acceptance
precedes the reference commit and any native target. The producer's ideal FP64
math is independent of upstream arithmetic; source parity still shares MLX's
runtime and is not a second independently implemented framework.

Causal convolution, partial rotary, attention/output gating, hybrid execution,
rollback/replay, resident Metal state, exact memory planning, real checkpoints,
quality and app/device tests remain open. These tiny fixtures confer no 128K
numerical generation, performance, mobile fit or full-family compatibility.
