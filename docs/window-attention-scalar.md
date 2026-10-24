# Scalar causal/sliding-window GQA

`cpu::window_attention` and `WindowAttentionShape` implement the standalone G2
target after the [independent attention witness](gemma-local-attention-reference.md).
They do not own or mutate a cache/session and no model graph uses them yet.

Queries/context are contiguous FP32 `[queries,query_heads,D]`; separate K/V
buffers are `[kv_heads,capacity,D]`. Only the first `keys` rows of each KV head
are valid. Capacity tails are neither read nor validated; callers still provide
initialized Rust slices. Tests fill the unused tails with NaN sentinels. The
query-head group maps
contiguously to a KV head; no expanded KV heads or intermediate copies exist.
Physical rows must be in logical temporal order, not wrapped ring order.

Absolute key/query starts define the positions. Included keys satisfy key≤query
and, for window W, query-key<W. No window means full causal attention. Scale is
explicit positive FP32, with no implicit `1/sqrt(D)`. Checked nonzero geometry,
head mapping, capacity, position/size arithmetic, exact lengths, finite query
and valid KV rows, and every query's nonempty visible range precede all writes.
Even a fully masked later query rejects before any scratch/context changes.

Dot products/scaling use FP64 and store FP32 scores. Scores retain all valid-key
dot products, including masked keys, for diagnostics; their overflow can reject
even when excluded from the context. Stable softmax uses FP64 and stores FP32
probabilities, with masked positions exactly zero. Context accumulates those
rounded probabilities times FP32 values in FP64, then rounds to FP32. Arithmetic
overflow leaves discardable partial outputs. Inputs remain immutable for replay.
The operation allocates no buffers and requires no extra alignment beyond FP32.
This is a correctness reference, with no optimization claim.

Six offline tests authenticate the prior receipt's literal SHA before fixture
identities and compare every nonempty score/probability/context array under
unchanged `1e-5 + 1e-4*abs(expected)` policy. Fixture layout conversion happens
only in the test adapter; production queries stay token-major and KV head-major.
Tests exercise NaN capacity padding, exact masked zeros, hand grouped-head and
stride values, all six slice lengths and invalid geometry/positions/inputs,
late fully masked queries, score and context overflow, replay, and exact
multiple-head prefill/chunk agreement with absolute offsets.

A seven-way uniform softmax also exposes the documented rounding policy: FP32
rounded1/7 probabilities sum slightly above one in FP64. With maximal FP32
values this context cast overflows and safely rejects, rather than returning
infinity. A finite replay succeeds. No tolerance or failure gate is loosened.

The [receipt](../benchmarks/results/2026-10-24-window-attention-scalar-v1.json)
binds exact source, fixture, raw checks, prior numerical sources and protected
binaries. Initial formatting caught missing leading zeros in test literals;
the unformatted proposal is retained and corrected before the checked freeze.
Astra reviews source and final evidence before commit. Builds use a separate
copy-on-write compiler cache; no runtime/build performance is qualified.
Formatting, strict default/Metal Clippy, 223 CPU/245 Metal-feature tests
(15/107 ignored respectively), and both iOS device/simulator cross-checks pass.
Cross-compilation does not establish physical device behavior.

Physical ring mapping, shared-KV producer/consumer state, transaction commit,
cancellation/rollback and prefix reuse still need independent witnesses and
implementation. Other G2 and full graph/session/Metal/real-model/quality/device
gates remain open in the [Gemma4 plan](gemma4-implementation-plan.md). The upstream
rotary precision gap remains unwaived. No native ABI, output/context/byte cap or
phone package changes.
