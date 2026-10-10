# Stored-Half gating preparation, 10 October 2026

The bounded stored-Half product passes all 143 completed gating commands,
87,360 exact independent Half outputs and 1,001 output guards. Original FP32
source gates also pass, with maximum tolerance fraction 0.491684814232 and
maximum absolute error 0.003214120865. This qualifies a tiny primitive;
performance, Qwen/full-model composition and M4 acceptance remain pending.

The [compact native record](../../benchmarks/results/2026-10-10-metal-gating-v1.json),
[stderr](../../benchmarks/results/2026-10-10-metal-gating-v1.json.stderr.txt),
[independent saved-evidence analysis](../../benchmarks/results/2026-10-10-metal-gating-analysis-v1.json)
and [executed producer](../../benchmarks/results/2026-10-10-metal-gating-analysis-v1.py.txt)
retain checks, exact identities, inputs, source gates, completed-work timings,
memory and raw-output hashes. Complete native outputs remain locally retained
at `benchmarks/raw/2026-10-10-metal-gating-v1.json.native.jsonl` before compaction.

The [Rust wrapper](../../tools/experiments/rust_metal_gating.rs) and
[shader](../../tools/experiments/rust_metal_gating.metal) implement one
elementwise product over two immutable contiguous Half arrays `[rows, 64]`.
Output is a third distinct owned array with the same shape. Each lane converts
both operands to FP32, multiplies them, and stores one Half RNE result. The
stored-Half SiLU stage boundary is preserved before multiplication.

Finite Half products have at most 22 significant bits and nonzero magnitudes
from `2^-48` to less than `2^32`. They are exactly representable in FP32 and
FP64, so an independent FP64 product followed by an IEEE Half conversion
provides an exact reconstruction oracle, including zero signs. A finite product
can still exceed Half range; the completed result must then be discarded.

Input admission accepts width 64, rows 1–32, at most 32 uniquely named cases
and 8,192 total active values. One-MiB JSON, unsigned finite Half patterns,
checked shape/storage arithmetic and the supplied working-set limit are
validated before device work. Each buffer has seven tail guards. Dispatch
width comes from the queried device and pipeline; rounded tails check the
active count before accessing any buffer.

The queried execution width is 32 on this Apple M1 Pro. These width-64 cases
have no inactive dispatch lanes. Output guards are still checked. Same-binary add regressions
exercise rounded tails at 1, 17 and 257 elements; this does not establish
gating-specific rounded-tail device coverage.

Three shared buffers, two immutable host baselines and three diagnostic
readbacks are accounted conservatively as eight times the guarded byte count,
plus the existing 64-MiB runtime allowance. The largest admitted case has
2,048 active values, 4,110 bytes per guarded allocation and 67,141,744 bytes
of total admission. Shared physical allocations are counted once. Retained
fixture/parser/JSON diagnostics fit the bounded runtime allowance.

The wrapper caches its pipeline and reuses all three native buffers across
three warmups and ten repetitions per case. It writes the inputs once and
initializes output sentinels before each command. Retained references and
lexical owners preserve resources through completion. Host reads occur only
after Completed/no-error checks, then validate both complete immutable inputs,
all guards and finite active output. Overflow returns a structured validation
error and discards the separate output. No session or KV state is advanced.

The [qualification driver](../../tools/experiments/rust_metal_gating.py) binds
all ten tiny `gate_silu`, `up_projection`, `gated` triples, their shapes,
complete payload hashes and original FP32 source gates. Its independent Half
product must reproduce every frozen oracle bit before target execution. A
17-row manual case covers signed zeros, normal ties in both directions,
positive/negative subnormal ties, the normal/subnormal boundary and finite
maximum. The 18 manual predictions were fixed before implementation and
rechecked with stdlib IEEE conversions without device execution.

The positive gating inventory is 6,720 values across eleven cases and 143
completed commands. Every output passes exact independent Half bits as well
as unchanged original FP32 source gates. The separate max-Half-times-two case
completes and rejects nonfinite Half output with the required structured error
and exit1. Same-binary add/tail, RMSNorm, embedding, residual (including overflow)
and SiLU regressions all pass. There are 1,040 completed positive commands across
all primitives and four expected negative runs: invalid shader, insufficient
budget, completed residual overflow and completed gating overflow.

The positive gating process's physical peak is 13,402,712 bytes. The largest
own-process peak across all native jobs is 17,121,856 bytes, below the 6-GiB
limit; gating admission is 67,141,744 bytes. These are own-child kernel charge
counters after completion, with logical shared storage counted once. No
end-to-end inference memory or performance advantage is inferred.

Four Rust parser/admission tests and five
[offline Python tests](../../tools/experiments/test_rust_metal_gating.py)
cover malformed/bounded inputs, memory accounting, independent products,
signed zeros/ties, data/guard corruption, immutable inputs, completion,
distinct output ownership, source gates, overflow and evidence-path refusal.
The base driver test additionally binds copied gating sources on build failure.
All four new Rust tests pass as part of eighteen isolated utility tests.
Isolated formatting, Clippy with warnings denied and locked/offline release
compilation pass. All 63 experiment Python tests, 35 benchmark Python tests and
76 ordinary engine Rust tests pass, together with root formatting and Clippy.
The [task-check transcript](../../benchmarks/results/2026-10-10-metal-gating-task-checks-v1.json)
retains commands, exits and output.

Checks/native work begin only after the uninterrupted long CPU worker exits
naturally. The [hold](../../benchmarks/results/2026-10-10-cpu-gating-check-window-hold.json)
and [idle proof](../../benchmarks/results/2026-10-10-cpu-gating-check-window-idle.json)
show only orchestration held, worker absent and transport defunct at
07:35:25 UTC. The [executed guard](../../benchmarks/results/2026-10-10-cpu-gating-check-window-idle.py.txt)
checks all ten measured states before accepting this window. Root numerical
and CLI sources remain unchanged. The isolated package protects frozen release
workers; hashes bind original/copied utility sources, inputs, helpers, manifests
and the binary before and after qualification.

Executed commands, requiring fresh evidence/build/input paths:

```sh
python3 tools/experiments/rust_metal_gating.py \
  --output benchmarks/raw/2026-10-10-metal-gating-v1.json \
  --build-dir artifacts/rust-metal-gating-v1 \
  --input-dir artifacts/metal-gating-inputs-v1
python3 -m unittest discover -s tools/experiments -p 'test_*.py'
cargo fmt --check
cargo clippy --locked --offline --all-targets -- -D warnings
cargo test --locked --offline
```

This prepares a tiny primitive only. Qwen width-4,864 rows, SiLU/MLP composition,
decoder/KV integration, asynchronous scheduling, end-to-end speed, full quality
and M4 acceptance remain separate tasks. There is no fusion or adoption decision
from source inspection.

The device is Apple M1 Pro, MacBookPro18,3, 16 GiB, macOS 26.6.2/25G83,
Rust 1.90.0. The native math policy is Metal 2.3, safe math, precise functions,
and macOS 15 or newer. Full GPU traces and energy measurement are unavailable.

Compact record SHA:
`01d93a2a9645a626baaec4927165f4e098161fe5e571c05f8a3db06ec96495ec`.
Full local native output SHA:
`43b8a2bfa23ed1f14242db4a00dbb3d149107906e2483e3985ef2b43277a6bb7`.
Isolated binary SHA:
`3b7c4e079bf34a7c00cd0ab190bb2a4756b2e129101edb68558f7e4bab524ac8`.

Status: Astra accepted the completed task without findings after independently
replaying all outputs, source gates, raw-to-compact events and identities, and
checking memory, transcripts, idle-window evidence and reported limits.
