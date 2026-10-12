# Tiny lossless Metal K/V layout writes, 12 October 2026

All ten frozen cache cases and four manual writes match exact Half storage bits
for both new chunks and untouched prefix/tail capacity. Last-slot and full
capacity writes pass; one token beyond capacity is rejected before native
work. Astra accepted the completed task without actionable findings after byte-identical independent replay.

The [shader](../../tools/experiments/rust_metal_cache.metal) copies raw UInt16
Half storage from contiguous token-major `[rows,2,8]` K/V to separate
head-major `[2,32,8]` buffers. Source `(token*2+head)*8+feature` maps to
`(head*32+offset+token)*8+feature`. One lane owns a unique target in each cache;
inactive lanes return before access. Raw integer copying preserves signed
zero, subnormal and maximum-finite Half bits without arithmetic conversion.
K input is already rotated and V is unchanged. Prior tokens never move.

The [host](../../tools/experiments/rust_metal_cache.rs) admits only rows1–32,
heads2/dimension8/capacity32 and checked `offset+rows<=32`. Input files remain
at one MiB,32 cases and32,768 aggregate input/cache values. Shared chunk
buffers cover the rounded queried launch plus seven guards; each full cache
has seven guards after its allocated capacity. Four distinct tracked buffers
and their initialized owners remain live through completion. Copied shape
arguments are three UInt32 values (12-byte size/4-byte host alignment).
All shape/index/allocation arithmetic is bounded before dispatch.

The [driver](../../tools/experiments/rust_metal_cache.py) checks ten saved
layer/case K/V cache arrays against an independent scalar layout reconstruction.
All frozen caches place their local tokens at cache offset0. In particular,
the fixture using rotary absolute offset5 still writes at cache offset0;
it does not invent five committed preceding tokens. Four manual cases cover
last-slot offset31/one row, append offset7/three rows, overwrite offset0/two
rows and full capacity32 rows. Distinct stale initial K/V patterns expose
prefix/tail corruption and head/feature swaps; new chunks include exact
signed-zero, subnormal and maximum-finite bits.

The [native record](../../benchmarks/results/2026-10-12-rust-metal-cache-v1.json)
retains **182 cache commands** over three warmups and ten samples per case.
All **186,368 cache payload values** match their expected bits:52,416 newly
written values and133,952 untouched values. All **2,548 output guard values**
are retained and independently replayed. Native code additionally verifies
5,044 input guard values, but complete input readbacks are not archived.
Full cache output bits are retained in local hash-bound raw JSONL; the compact
record stores their hashes. No floating-point tolerance is needed for layout
copies, and all existing numerical gates remain unchanged.

Shared allocations and pipelines are reused. Qualification restores each saved
initial cache outside command timing on every repetition; this is a diagnostic
write primitive, not allocation-free steady-state decode. No authoritative
session-valid-prefix state exists here or advances. Future decoder integration
must commit only after the complete forward step succeeds and replay staged
writes after failure. These records do not qualify cancellation or failure
recovery of an integrated session.

Same-binary rotary, attention/MLP projections, composed MLP, query/manual,
add, gating, tails and failures total **1,833 positive commands and seven
negative jobs**. All pass, preserving the previously investigated query/V
rounding differences. Thirteen build checks include formatting, Clippy with
warnings denied,29 standalone Rust tests and locked offline arm64 release
build. Binary SHA-256 is
`f0c61fc7f53feff4cfb67daeb83b1ba84ec3bac2c493c35c37f7828fbd4a26af`.

Maximum cache admission is **67,121,320 bytes**. Shared physical allocations
are counted once, with full host baselines and four full readbacks separately;
the total is three times shared bytes plus64MiB for bounded fixture/JSON/runtime
state. Peak own-child physical charge across all jobs is **22,758,024 bytes**.
These are bounded primitive-process measurements, not full-model or system
memory evidence. Root release binaries and frozen model endpoints are unchanged.

Hardware remains M1 Pro / MacBookPro18,3 /16GiB / macOS26.6.2(25G83), Rust1.90.0,
SDK26.5 and minimum macOS15. Shader language2.3 uses safe math and precise
functions. Start/end observations retain AC power, configured AC low-power
mode off and no thermal/performance warning. Continuous temperature/clocks,
energy, total device/system memory and GPU traces remain unavailable. No
latency or speed acceptance is claimed.

The [idle proof](../../benchmarks/results/2026-10-12-rust-metal-cache-idle-v1.json)
and [archive producer](../../benchmarks/results/2026-10-12-rust-metal-cache-archive-v1.py.txt)
retain natural tool completion, prior numerical-chain absence, stopped VM and
unchanged protected endpoints. Native children are reaped before offline
checks and endpoint hashing. The [analysis](../../benchmarks/results/2026-10-12-rust-metal-cache-analysis-v1.json)
and [exact producer](../../benchmarks/results/2026-10-12-rust-metal-cache-analysis-v1.py.txt)
replay raw events, exact cache targets/untouched values/guards, all original
numerical validators, layouts/admission and source/copied-source/driver/input/
manifest/binary identities. Cache analysis reconstructs each head/position/
feature independently of the driver's slice-based oracle.

Three targeted tests and full offline suites pass:91 experiment Python,
193 benchmark Python, root formatting/Clippy and76 ordinary Rust tests using
an isolated debug target. Real-model suites were not rerun. Retain this
bounded layout-write mechanism; causal GQA, propagated decoder, session
commit/cancellation, Qwen/backend quality, scheduling and M4 completion remain
open.
