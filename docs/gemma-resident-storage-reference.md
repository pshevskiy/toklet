# Complete Gemma resident-storage reference before native loading

The [frozen inventory](../tests/fixtures/gemma-resident-storage-reference-v1/inventory.json)
and its [manifest](../tests/fixtures/gemma-resident-storage-reference-v1/manifest.json)
define expected original source bytes and guarded resident buffers for a later
complete E2B text-weight loader. This task executes source-only reference tools;
it does not load an engine/GPU model, admit complete-model storage or grant
generation, app, download or phone capability. The
[evidence record](../benchmarks/results/2026-10-26-gemma-resident-storage-reference-v1.json)
and [active plan](gemma4-implementation-plan.md) preserve those boundaries.

The pinned original container is 3550670554 bytes, SHA256
`038e39a37a7667373d2c3991375446b10c96ae1d717a68674870343db376b76e`,
from `mlx-community/gemma-4-e2b-it-4bit` revision
`238767527555cb75a05732a84dff5d6ba0dd6809`. Existing config/index/header fixtures
are authenticated independently. The capture builds the named 35-layer logical
inventory; replay separately derives roles from the pinned header. There are
278 packed matrices and 262 BF16 vectors, or 1096 physical text components.
The two large tables use sixteen 16384-row chunks each, all other matrices one:
308 packed chunks, 924 packed buffers and 262 converted vector buffers, giving
1186 expected guarded allocations. These are expected counts, not observed GPU
allocations. Scalar vectors are included; excluded modalities are not resident.

Pass 1 hashes the exact captured length/header bytes used for interpretation,
then streams all contiguous physical spans, including excluded modalities,
while retaining per-text-component digests from those same bytes. Exact size,
EOF and the pinned whole-container digest precede any selected-byte expectation.
Pass 2 reads bounded row chunks and vectors, requiring increasing complete
component coverage and matching the pass-1 component digests. Original packed
bytes, alignment padding, seven-u32 guards and exact BF16-to-FP32 vector bits
have separate expected digests. Negative-zero bits remain intact. No full CPU
packed model or expanded matrix is retained by these source tools.
The real checkpoint's vectors contain zero negative-zero values; preservation
is demonstrated by the explicit synthetic signed-zero/short-read control.

The identity combines a historical whole-container anchor with matching selected
text consumption. It does not establish a second complete-container hash,
current path/regular-file stability or excluded bytes after pass 1. No remote
model code or chat template executes. These storage expectations neither resolve
the original source-BF16 numerical differences nor qualify codebook/activation
domains or graph behavior afresh.

Two independently controlled captures match byte for byte: 806478 bytes,
SHA256 `a3dff3b9d825fcf4fa6f391262732c5f429345682d683d6a39bc551ddacd68e2`.
Both independent complete replays pass without importing the capture algorithm.
Fifteen portable checks cover short/partial reads, exact header binding, changed
bytes between passes, signed zeros including one-byte reads, odd metadata
padding, nonfinite vectors, duplicate JSON, existing/dangling output refusal
and partial publication. Failed records preserve consumed prefixes, original
errors and offending vector bits/index; secondary diagnostic failure cannot
replace the original error. Early component observations are explicitly
unauthenticated candidates. Nineteen actual reader cases (baseline plus eighteen
mutations) reject changed source/guard/conversion/range/accounting views,
duplicate/missing coverage, coordinated hash replacement and widened claims.

Known matrix accounting is 2603500912 requested bytes: 2603435280 guarded
buffer bytes, 31136 inline owner bytes and 34496 chunk descriptor bytes, under
the preceding qualified 112-byte owner/112-byte chunk ABI. Expected guarded
FP32 vector buffers add 1151284 bytes. Maximum simultaneous three-array source
payload is 82575360 bytes, or 82575432 with its descriptors; maximum raw vector
conversion overlap is 3072 bytes. Context and cold metadata policies are
67108864 bytes each. New vector/bundle ABI, actual owner Vec spare capacities,
aggregate model-load policy and admission remain explicitly unknown/unqualified.
The later loader must charge each inline owner once and readmit all actual
capacities before upload; these expected buffer sums are not a full-model budget.

On MacBookPro18,3, Apple M1 Pro, 16 GiB, macOS 26.6.2 (25G83), capture RSS/physical
peaks are 29425664/21791488 and 29163520/24101560 bytes. Replays peak at
28753920/21938800 and 28721152/21283440; regressions at 37978112/29295216.
All fit the separate 512 MiB per-process allowances. These are source-tool
process peaks, not resident-model memory or inference measurements.

All eight owned groups close; exact source copies, interpreter, input/output
identities and protected binaries are checked. Rust/Metal/iOS evidence is the
prior committed `607bbb1` task (356 CPU, 400 Metal-feature tests, required lint,
both iOS targets and optimized checks), explicitly carried forward under exact
unchanged-source/include/build-file bindings. Those tests are not newly run by
this reference-only task.

Retain this pre-target inventory. Next is the separately reviewed concrete
immutable E2B storage bundle and two-pass loader, cumulative/actual-capacity
admission, transactional failure/replacement and Context teardown, followed by
full graph/session integration. All source/precision/rotary, G2–G7, quality,
performance, app/phone and 131072-output-plus-prompt gates remain open. No
capability, eligibility, tolerance or context/output limit changes.
