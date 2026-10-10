# In-place Half Metal residual, 10 October 2026

The standalone [Rust primitive](../../tools/experiments/rust_metal_residual.rs)
and [shader](../../tools/experiments/rust_metal_residual.metal) add an immutable
Half delta to mutable Half state. Native qualification passes on M1 Pro. The
[driver](../../tools/experiments/rust_metal_residual.py) checks 20 frozen tiny
residual stages and one independent manual rounding case. Numerical policies,
engine/session execution and milestone status remain unchanged.

## Shape, arithmetic and ownership

Both buffers are contiguous `[rows,hidden]` Half arrays. One lane owns each
flattened element, converts its two operands to FP32, adds once and stores Half
once with round-to-nearest-even. Lanes do not read other state elements; the
delta allocation is distinct and immutable. This makes the in-place update
race-free without an extra output allocation. Adjacent lanes access adjacent
state/delta elements. Logical traffic is six bytes per scalar, or 192 bytes
per width-32 row, before cache effects. No speed or fusion acceptance is claimed.

Slots 0/1 carry mutable state and immutable delta at offset zero. Slot 2 copies
a UInt32 count with size/alignment 4/4. Checked shape and rounded dispatch
indices fit UInt32; the shader checks count before reading. Seven sentinels
follow each buffer. The loader bounds input to one MiB, 1–32 unique bounded
ASCII names, 1–32 rows, width 32/896 and 8,192 cumulative elements. All operand
bits must be finite unsigned Half values. Parsing and allocation admission
complete before device discovery; JSON/file temporaries are released.

Two owned shared allocations persist across each case's three warmups and ten
samples. The full state baseline is restored before every repetition outside
GPU access, so repetitions do not accumulate deltas. Queue and pipeline are
cached. Distinct retained command resources and owners remain alive through
one wait; host reads follow successful completion/status/error checks. Every
repetition verifies the delta and its guards and all state guards. No raw buffer
handle or borrowed slice escapes.

Finite operands can produce a result outside Half range. The native call checks
all active result bits after completion and rejects nonfinite Half. That
in-place result is invalid and discarded as the call exits. Detection does not
restore the previous state or qualify transactional rollback, cache commits,
session recovery or cancellation. Any future reuse would require restoration
before another dispatch. The standalone overflow case checks this exact failure
stage/message and exit status, rather than accepting a successful result.

Admission counts shared storage, host baseline and readback for each of the two
buffers, plus the fixed 64 MiB runtime/fixture/diagnostic allowance. The largest
tiny case uses 4,124 shared and 8,248 host bytes, admitting 67,121,236 bytes.
Unified physical allocations are counted once. The engine's 6 GiB ceiling is
unchanged; native child footprints do not establish driver/system physical memory.

## Independent qualification

Five existing tiny cases each supply two layers' attention residual and final
layer output: 20 stages and 5,632 active elements. Layer-zero attention uses
embedding state, later attention uses previous-layer output, and the second
residual uses attention-residual state plus down projection. Every stored Half
oracle is independently rechecked against FP32 addition and one Half RNE before
GPU execution. Complete original FP32 references and their unchanged pre-GPU
source gates are also mandatory; exact Half agreement does not replace them.

The manual 17-row case supplies 544 elements cycling through 16 hardcoded
operand/result triples: positive and negative nearest-even ties, negative-zero
addition versus opposite-sign cancellation, minimum Half subnormals, the
subnormal/normal boundary, exact cancellation and finite maximum range.
Independent stdlib IEEE conversions recheck all fixed expected bits before
target work. These exact result bits need no new numerical tolerance. At a
queried width of 32, 17 rows fill complete threadgroups; this case makes no
inactive-tail claim. The existing add/tail regression covers the shared rounded
launch helper separately.

The total positive residual inventory is 21 cases, 6,176 values and 273 completed
dispatches: 80,288 active outputs and 1,911 state guards over all repetitions.
A separate maximum-Half-plus-maximum-Half input must fail after completed work.
The custom validator returns explicit `False` for this negative fixture, and
the base driver additionally requires `passed=false` and exit 1. Ordinary
primitive validators still require success and exit 0.

Pinned manifests and complete input/reference/oracle payload hashes are checked
before device work. Full native diagnostics are validated before compaction to
counts/hashes/gate proofs, while raw JSONL remains local and hash-bound. Final
checks revalidate manifests, drivers and all generated input identities.
The native build remains an isolated pinned utility with no engine dependency.
All 39 add/tail, 364 RMS and 78 embedding regression dispatches and structured
shader/admission failures repeat with the mode/build/validation changes.

Three native unit tests cover finite operand admission including deliberate
overflow, shape/name/inventory limits, guarded storage and checked budget math.
Five offline Python tests cover exact/source oracles, corruption, restoration/
in-place/completion/guard/accounting assertions, overflow failure semantics and
evidence aliases. A base-driver test checks custom negative-fixture exit policy.
The isolated utility passes formatting/Clippy with warnings denied and all
11 standalone Rust tests. Engine formatting/Clippy and all 75 ordinary Rust
tests pass, along with 77 offline Python checks. Astra accepted the completed task
without findings after independently checking source and artifact identities,
raw validators, exact Half outputs, unchanged FP32 gates and overflow rejection.

The [native record](../../benchmarks/results/2026-10-10-metal-residual-v1.json)
passes all 273 positive dispatches, 80,288 active outputs and 1,911 state guards.
Every output matches the independent Half oracle exactly, including all manual
edges. Maximum unchanged FP32-source gate fraction is `0.4957460689362005`;
maximum absolute source error is `0.0016481876373291016`. The intentional
maximum-Half sum completes, then returns the expected validation error and
exit 1; its in-place result is invalid/discarded without a rollback claim.
All add/RMS/embedding regressions and structured shader/admission failures pass.
No watchdog fires. Positive residual child physical peak is 14,254,656 bytes,
below its conservative 67,121,236-byte admission and 6 GiB; the overflow child's
peak is 4,964,928 bytes. The largest regression peak is 16,810,560 bytes (RMS).

The host is M1 Pro, macOS 26.6.2, SDK 26.5 and Rust 1.90. Metal compile policy
remains 2.3, safe math, precise functions and deployment target macOS 15. Power
is AC with AC low-power mode disabled and no recorded thermal/performance
warning. These are functional checks with no latency or energy acceptance.
The result and [stderr](../../benchmarks/results/2026-10-10-metal-residual-v1.json.stderr.txt)
bind sources/build copies, generated inputs, frozen manifests, qualification
helpers/drivers, executable and full raw JSONL. Key SHA-256 identities are:

| Artifact | SHA-256 |
|---|---|
| Residual driver | `51bae4e5e61324118af622647ae64394af1952aad023a5f426520ee5ff1a92c0` |
| Executable | `a83e6bc465e9cb7b274669d82d4a3eadd36f842541e426ce6a5667bb30795966` |
| Full native JSONL | `ba6f5ef4d53ab754df92eda9275b612f737d0bc0972dfbe001324477e9dec0fa` |

The [hold](../../benchmarks/results/2026-10-10-cpu-residual-check-window.json),
[idle](../../benchmarks/results/2026-10-10-cpu-residual-check-window-idle.json)
and [check completion](../../benchmarks/results/2026-10-10-cpu-residual-check-window-completed.json)
records preserve isolation. Only orchestration parent 34871 is held; numerical
worker 44485 finishes naturally before native/check work. Idle state is verified
at 03:39:45.193255 UTC. The original scalar numerical worker remains idle with
unchanged CPU time. Frozen scalar executables and qualified candidate source/
receipt identities remain exact. The coordinator stays held for the subsequent
selected Qwen residual task, which runs only after this task's acceptance/commit.

Qwen residuals, composed forward/cache/session behavior, quality, performance
and GPU traces remain pending. M4 remains open.

```sh
.venv/bin/python tools/experiments/rust_metal_residual.py --output benchmarks/raw/metal-residual.json --build-dir artifacts/metal-residual --input-dir artifacts/metal-residual-inputs
```
