# Complete immutable Gemma E2B GPU storage

The storage-only `gemma_affine::resident::E2bWeights` loader owns all 278 original
packed text matrices and 262 converted learned vectors for the pinned
`mlx-community/gemma-4-e2b-it-4bit` checkpoint at
`238767527555cb75a05732a84dff5d6ba0dd6809`. It does not implement a forward graph,
KV session, tokenizer, app profile or larger context capability. See the
[public evidence](../benchmarks/results/2026-10-26-gemma-resident-weights-v1.json)
and [independent pre-target storage inventory](gemma-resident-storage-reference.md).

## Layout and ownership

Matrices retain the original row-major `[out_features,in_features]` packed4,
group64 representation with separate BF16 scales/biases, existing bit order and
codebook validation. At most 16384 rows belong to one immutable chunk. There are
308 chunks and 924 packed device buffers; each includes seven sentinel words,
with zero padding for an odd two-byte metadata tail. All 262 vector buffers hold
FP32 bits obtained directly from BF16 bits, preserving signed zero. The original
real vectors contain no negative-zero values; the small transaction witness
checks that case explicitly.

Small named bindings resolve every matrix/vector role and layer geometry at
loading time. Immutable buffers and their retained Metal resources outlive the
construction Context. The bundle owns no activation, KV, tokenizer or numerical
session scratch and retains no second full CPU packed model. Original input
bytes are staged one three-component chunk at a time; vector conversion follows
after that source scratch has dropped. Layout and bindings remain specific to
the admitted E2B graph rather than a universal model framework.

## Authentication, admission and failures

The loader validates bounded config/index and the captured safetensors header,
complete tensor coverage and explicit text-only modality policy. It checks the
whole-model minimum and device buffer limits before payload reads/uploads.
Pass one hashes the exact interpreted header and all physical source bytes.
Pass two hashes the exact selected bytes supplied to each upload, including all
1096 text components. Only complete consumption publishes an E2bWeights identity.
The identity is a historical full-container anchor plus matching selected text
consumption; excluded modalities are neither executed nor asserted unchanged in
a second pass. An arbitrary Read+Seek adapter is not a current-file identity API.

Actual outer Vec capacities, child descriptor capacities, original guarded GPU
buffers and maximum reusable source capacity replace their corresponding minima
once. Checked arithmetic includes all remaining minimum payload during loading.
The separate policies reserve 64 MiB for Context, 64 MiB for cold metadata and
128 KiB for hash stack. These are policies, not observed allocator totals.
The caller must first subtract other models/sessions, application/codec state,
caller-owned config/index and reader backing allocations from its available
budget. A retained old model must also be charged during a replacement attempt.

Errors drop the unpublished candidate. Typed original I/O errors survive; the
loader does not mutate an earlier model. The small GPU witness covers admission
before source IO, changed selected bytes between passes, late typed IO failure,
a late nonfinite vector, late invalid metadata and a valid replacement. It
captures the complete previous owner's buffers/identities after every attempt.
Failed candidate buffers themselves are explicitly uncaptured; no driver-cache
release or in-flight command recovery claim follows from their RAII cleanup.

## Actual Mac storage qualification

The optimized Metal-feature test ran on MacBookPro18,3 / Apple M1 Pro, 16 GiB,
macOS 26.6.2 (25G83). Every original guarded buffer matches its independently
frozen digest after construction Context teardown. All 1186 live buffer
identities are distinct. Budget-minus-one through the public loader reads only
the 318228-byte prefix/header and rejects before any model payload IO.

| Accounting / observation | Bytes |
| --- | ---: |
| Requested retained model ownership, including actual slots and guards | 2604665212 |
| Conservative complete load admission policy | 2821589444 |
| Maximum three-component source capacity with descriptors | 82575432 |
| Separate diagnostic allowance | 33554432 |
| Whole native process peak resident set | 2644393984 |
| Whole native process peak physical footprint | 2637728120 |
| Mac process policy | 6442450944 |

The root owner is 6728 bytes, each vector descriptor 24, each immutable matrix
owner 112 and each chunk descriptor 112 on this build. These Rust layout
observations are not a stable C ABI. Actual outer slots are 278/262; the
independent replay verifies all buffer hashes, chunk boundaries and accounting.
The small transaction run checks 90 original projection values after Context
teardown, five rejected candidates and one replacement; its observed RSS/physical
peaks are 19218432/8012352 bytes within its separate 512 MiB allowance.

Formatting, strict default/Metal all-target Clippy, 356 CPU tests and 408
Metal-feature tests pass. Ordinary ignored counts are 16/127; those are not a
claim that every hardware/model test ran. Both iOS targets compile, ten portable
reader controls pass, and the two explicit optimized GPU tests and independent
replays pass. All 15 owned command groups close; 3762 input bindings, 30 retained
outputs and protected release binaries match. Actual source identity is the
frozen file set over base `aec8a4f`, not that base revision alone.

An early strict-lint failure was corrected; another development reader attempt
selected system Python 3.9, which lacks the reference helper's strict zip API.
The final isolated reader uses the pinned absolute Python 3.12.8 executable.
Astra caught incorrect expected error text and Cargo test-target selection before
actual runs. No failed actual GPU attempt or numerical tolerance change occurred.
The first publication reused an older artifact-directory name; its overwritten
collection script was recovered byte-for-byte against its original SHA, and the
new publication was moved to a fresh directory with the first result preserved.
No model or native test capture changed and no hardware rerun was required.

## Remaining scope

This closes G4.1 storage only. Next is G4.2: compose existing GPU projections/
gathers with CPU graph ordering into the complete short Mac diagnostic. Whole
resident graph/KV/scheduling, required precision/quality decisions, model-defined
full-window support, app integration and independent phone feasibility remain
open. The current requested weights alone exceed the tested Qwen3 phone profile's
2304 MiB process policy; that policy is not a universal iOS limit. Existing
mathematical/BF16/GPU/rotary failures remain unwaived. Timings here are functional
load/hash observations, not representative prefill/decode performance.
