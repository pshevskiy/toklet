# Resident Metal Half-score router

5 November 2026. Standalone GPU qualification for the first routing leaf of the
[Granite sparse-MoE plan](granite-moe-next.md). The leaf is test-gated until the
resident graph has a consumer; it does not enable MoE inference or a catalog route.

A single thread owns each token row. Half scores `[T,E]` widen to FP32; a finite
check precedes writes for that row. Repeated scans select descending score and
ascending expert ID on ties, using a 32-bit selected-expert mask. The selected
slots alone enter FP32 softmax, with explicit ranked-order summation. IDs and
FP32 gates occupy separate `[T,K]` buffers with a shared physical row stride.
No sorting or normalization runs over unselected probabilities.

The allocation plan bounds capacity and experts to 32, validates `1≤K≤E`, checks
both strides and UInt32 offsets, and counts guarded input/ID/gate/status buffers
plus fixed Rust owner bytes. The qualification owner allocates from those exact
lengths and rejects short budgets and device buffer limits before allocating.
A 32-row/32-expert/top 8 owner with strides 37/13 uses 5,798 operand bytes, plus the
fixed owner header. Pipeline reservation is separate from numerical buffers.
This is an exact consumed leaf layout; full session/KV/expert scratch is open.

Encode borrows resident inputs and outputs, validates sizes, device and alias
identity, and retains resources until completion. No upload, numerical buffer
allocation, readback, host wait or status reset occurs in the encode path.
Status is OR-only. Other rows can complete when one row is invalid; the caller
must discard the candidate on any nonzero status. Tests keep a late invalid row
untouched and prove that a subsequent valid encode does not clear shared status.

The independent Half-input fixture was frozen before target execution. It
rounds source words to Half first and uses the established scalar sorting oracle.
IDs are exact; gates use `2e-7 + 2e-6*abs(reference)`. Tests also compare to the
reviewed Rust CPU router. Cases include signed zero, Half rounding ties, mixed
cutoff ties, large finite Half scores, batches 7/8/9, full capacity and an exact
ranked FP32 mass witness. Padding, inactive rows, guards and input immutability
are checked on repeated launches.

Apple M1 Pro release GPU checks pass: 15 cases, 45 fixture launches, 1,527 slots,
maximum gate absolute error 7.45e-9 versus the CPU/independent references. The
invalid-row NaN/Inf and same-command status tests pass. Unprofiled encode shows
zero Rust allocations; driver encoders and compilation are separate. No speed
benchmark, expert projection/reduction, model/session or phone qualification.

[Fixture](../tests/fixtures/moe-router-half-v1/fixture.json),
[checks](../artifacts/moe-router-half-checks-2026-11-05/).

Astra approves the standalone leaf with no actionable findings. All six required
check jobs pass, including 574 CPU/684 Metal-feature tests, strict Clippy and
release build. Both explicit release GPU tests pass. Source replay is byte-identical.
[Result](../benchmarks/results/2026-11-05-moe-metal-router-v1.json).
