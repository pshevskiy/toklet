# Tiny split-half Metal rotary, 12 October 2026

All twenty frozen tiny Q/K rotary stages pass their unchanged source gates and
match every frozen Half bit. An exact position-zero identity and completed
finite-input overflow rejection also pass. Astra accepted the completed task without actionable findings after byte-identical independent replay.

The [shader](../../tools/experiments/rust_metal_rotary.metal) reads contiguous
`[tokens,heads,8]` Half input and writes a distinct Half output. A lane owns
feature `f` and `f+4` of one token/head; inverse frequency is
`theta^(-2*f/8)`, angle is `(offset+token)*inverse`, and theta is the frozen
1,000,000. The two results are `a*cos-b*sin` and `b*cos+a*sin`. FP32 angle,
power, precise sine/cosine and uncontracted products/sums precede one Half
store per component. It implements split-half pairing, not adjacent pairing.
No dynamic dispatch, head expansion, host coefficients or in-place aliasing.

The [native host](../../tools/experiments/rust_metal_rotary.rs) admits only
rows1–32, heads2/4, dimension8 and theta1e6, with checked `offset+rows<=2048`.
Input JSON remains at one MiB,32cases and8,192 aggregate elements; malformed
unsigned values, duplicate IDs, nonfinite Half input, wrong shapes/theta and
out-of-range positions fail before native allocations. Numerical qualification
covers the frozen positions0–31 and offset5–7; the parser's2048 bound does not
establish numerical or model/context qualification at that depth.

Shared input/output storage covers twice the rounded pair grid plus seven
sentinels. Full input baselines and all output inactive/tail guard values are
checked after completion. Allocation-size/grid arithmetic is checked; queried
pipeline/device widths govern dispatch. The copied four-UInt32 shape argument
has explicit16-byte size/4-byte host alignment. Resource owners and tracked
command dependencies remain live through completion; host access begins only
then. Inputs/output and compiled pipeline remain resident over three warmups
and ten samples per case, with one command commit/wait per repetition.

The [driver](../../tools/experiments/rust_metal_rotary.py) reads independent
saved Q/K projection inputs for both layers of all five cases. Offset5 is an
absolute rotary-position fixture; it does not imply five cached prior tokens.
The manual position-zero case uses sixteen nonzero dyadic values with exact
identity Half bits, independently exposing component ordering. The overflow
case rotates maximum finite positive/negative Half at position1; completed
nonfinite output is rejected and discarded.

The [successful native record](../../benchmarks/results/2026-10-12-rust-metal-rotary-v2.json)
contains **273 rotary commands,55,120 active Half values and5,863 guard values**.
Every active Half bit equals its frozen/manual reference. Maximum original
source gate fraction is0.49812881633031697 and absolute error0.0021777302026748657.
There are no rotary discrepancies and no tolerance changes. GPU internal
coefficients and pre-store FP32 rotary values are not read back; the evidence
qualifies final Half outputs on this device and these inputs.

Same-binary composed MLP, all isolated attention/MLP projections, query/manual,
add, gating, tails and required failures total **1,651 positive commands and
six negative jobs**. Previously investigated query/V projection differences
retain their scopes. Twelve build checks include formatting, Clippy with
warnings denied,27 standalone Rust tests and locked offline arm64 release
build. Final binary SHA-256 is
`9264117662913dae2354000aba9f03ab8137c5a775498c5fd5d15598e98d242e`.
The [excluded first attempt](../../benchmarks/results/2026-10-12-rust-metal-rotary-v1.json)
stopped before GPU execution on Clippy's `is_multiple_of` suggestion. Its
stderr and changed copied Rust source are retained; no numerical results were
accepted from that attempt.

Maximum rotary admission is **67,119,174 bytes**: two shared buffers, one host
baseline and two readbacks at their full mapped sizes, plus64MiB runtime
allowance for bounded fixtures/JSON/repetition records. Shared physical memory
is counted once. Peak own-child physical charge across all jobs is
**13,140,592 bytes**. These are primitive-process checks, not full-model or
system-memory evidence. Native raw rotary guard values are independently
replayed; existing projection/MLP guard checks retain Boolean evidence.

Hardware remains M1 Pro / MacBookPro18,3 /16GiB / macOS26.6.2(25G83), Rust1.90.0
and SDK26.5; minimum target macOS15, shader language2.3, safe math/precise
functions. Start/end observations retain AC power, configured AC low-power
mode off and no thermal/performance warning. Continuous temperature/clocks,
energy, total device/system memory and GPU traces remain unavailable. No
latency or speed qualification is claimed.

The [idle proof](../../benchmarks/results/2026-10-12-rust-metal-rotary-idle-v1.json)
and [archive producer](../../benchmarks/results/2026-10-12-rust-metal-rotary-archive-v1.py.txt)
retain natural tool completion, prior numerical-chain absence, stopped VM,
unchanged protected endpoints and the excluded pre-GPU record/source. All
native children were reaped before offline checks and endpoint hashing.
The [analysis](../../benchmarks/results/2026-10-12-rust-metal-rotary-analysis-v1.json)
and [exact producer](../../benchmarks/results/2026-10-12-rust-metal-rotary-analysis-v1.py.txt)
verify raw events, all validators/source gates/guards, memory/layouts, source
and copied-source/driver/input/manifests/binary identities. Full offline checks
pass:88 experiment Python tests,193 benchmark Python tests, root
formatting/Clippy and76 ordinary Rust tests in an isolated debug target.
Real-model suites were not rerun. Root release binaries remain unchanged.

Retain this bounded rotary primitive. Cache writes, causal GQA, propagated
layer/decoder/Qwen, backend quality, scheduling and M4 completion remain open.
