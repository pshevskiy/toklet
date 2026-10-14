# Explicit compensated resident Half projection

`ProjectionMode::HalfCompensated` passes the unchanged primitive and independent
graph gates. It replaces Original MPS weight projections only when selected on a
Half model, including the final FP32 vocabulary output. This is a numerical
mechanism qualification, with no held-out quality or speed acceptance. Original
projection, Original RMS, attention, FullLogits and chunk128 defaults are unchanged.

## Arithmetic, layout and ownership

The [first-divergence investigation](2026-10-14-half-consistency.md) motivates this
candidate: identical inputs to layer0 Q gave shape-dependent MPS rounding errors.
The custom operation uses resident row-major Half weights `[out,in]` and Half
activations `[tokens,in]`, with explicit input/weight/output strides and the input
view's byte offset. One queried SIMD group owns one token/output dot product. It
reads contiguous weight/input columns across adjacent lanes, independently of the
number of active rows. This mechanism repeats weight reads across prompt tokens;
there is no prompt tile or performance claim.

Products of two finite Half operands are exactly representable in FP32, including
subnormal Half products. Two-component Neumaier accumulation and the existing
queried-width tree preserve small terms; Half bias is added last. Round-to-odd
prepares the inexact FP32 intermediate only before Half storage. A separately
named final-output kernel uses ordinary nearest FP32 rounding. Compensation does
not establish universally correctly rounded dot products.

The two pipelines are cached and their reduction widths must agree at session
admission; both queried widths are32 on this M1 Pro. Selection occurs at the
operation boundary. Checked descriptors/extents include the ordinary last-row
vocabulary view's nonzero offset. Stack parameters replace no buffers and allocate
no Rust heap memory. Existing immutable weights, scratch, guarded status and
completion/publication ownership remain in use; no extra model copy is retained.
Q4 models reject this Half-only mode. Diagnostic profiling admits17 events/layer
plus4 final/embedding events, plus one token-selection spare when required.

## Native gates and session checks

All native work is sequential and separate from builds, CPU tests and large hashes.
Each bounded owned-process runner executes exactly one passing ignored test,
checks the immutable binary endpoint, kills/reaps/closes owned resources and retains
stdout/stderr. Streams are bounded to1MiB and each worker to900 seconds; sampled
conservative worker-plus-driver lifetime charge must fit6GiB.

| Check | Result |
| --- | --- |
| Half/FP32 primitive outputs | 1,668 exact comparisons;22 exact Half double-rounding witnesses. Signed midpoint, subnormal, overflow, analytical large/bias cancellation, FP32 midpoint-even, odd strides, offsets, tails, guards and NaN status pass. |
| Frozen tiny source/storage gates | 2,520 complete captured stage arrays pass the unchanged FP32-source envelope and per-row greedy IDs. Five cases, chunks1/3/8/full, offsets and capacity tails are covered. The diagnostic inventory is exactly35 stages; raw cache/scores/probabilities are outside this capture API. |
| Ordinary/diagnostic publication | Full final-row FP32 distributions match bit-for-bit at every completed tiny chunk, including nonzero input-view offsets. |
| Session and selection | Exact budget boundaries, append/edit/reset replay, token-only/full-logit stage refinement, offsets/tails, cancellation/stop/sink errors, late diagnostics/profile and transactional nonfinite/completion failure pass. |
| Warmed allocation |20 full-logit and8 token-only numerical decode steps allocate zero Rust heap memory. Native framework allocations are separately accounted. |
| Preselected larger independent gate |969 selected full arrays pass the frozen Half/source envelope and greedy IDs; maximum gate fraction0.5792791429143427. Original normalization is selected. |
| Larger profile and memory |480 completed profile events; maximum admission3,226,794,990 bytes, own native lifetime peak3,194,374,976 bytes and sampled joint charge3,205,254,408 bytes, below6GiB. |

The first native tiny attempt fails before arithmetic comparison: its test
mistakenly treated all44 fixture entries as35 captured stages. A reviewed explicit
inventory fixes the harness; fixture bytes, shapes, gates and production arithmetic
are unchanged. Failed v1 evidence remains retained and excluded. The fresh v2
qualification runs all seven native targets successfully. Initial compile failures
are also retained; final formatting, strict Clippy in both configurations,
87 ordinary CPU tests,97 ordinary Metal-feature tests and178 Python tests pass.
An owned executable that deletes itself after successful stub output also verifies
that an endpoint hash failure preserves failed JSON and cleanup evidence. This
adversarial transport check executes no GPU work and is excluded from native gates.

## Evidence, review and next task

Source parent is `6729057`. Frozen complete snapshots and binaries are retained at
`artifacts/m5-half-projection-source-v{1,2}/` and
`artifacts/m5-half-projection-native-binaries-v{1,2}/`. All Cargo work uses
`CARGO_TARGET_DIR=artifacts/m5-checks`; protected CPU binaries are unchanged.
[Larger parity result](../../benchmarks/results/2026-10-14-half-compensated-parity-v1.json)
and the [qualification receipt](../../benchmarks/results/2026-10-14-half-compensated-qualification-v1.json)
bind all sources, independent fixtures/oracles, checked binaries and success/failure
logs. Astra reviews source corrections and saved evidence before the task commit.

The next task is explicit scorer integration and the unchanged short NLL guard.
Only a passing consistency guard permits full held-out loss collection. This
report does not establish larger batch/single bit equality, acceptable full loss,
task quality or latency. Both fixed Q4 quality failures remain; M5.2/M5.6 and
overall M5 remain open.
