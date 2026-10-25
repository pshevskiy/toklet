# Audited source evidence carried by a resident Smol model

`metal::Model::load_mlx_smol_audited` now carries the
[historical source evidence](smol-native-content.md) through successful resident
model construction. This explicit one-checkpoint path reuses the existing MLX
upload and assembly implementation. The
[receipt](../benchmarks/results/2026-10-25-smol-model-content-v1.json) binds source,
checks and six separate completed Mac processes.

## Publication and ownership

Every model assembled through the common constructor starts with absent identity.
The audited import copies the source's private-field immutable identity only
after all tensor uploads, finite/affine validation and final assembly succeed.
An incomplete source proof returns an error and discards the unpublished model.
There is no public setter or way to retrofit an ordinary imported model. Legacy
MLX, Half and Q4 constructors retain absent identity.

The read-only getter describes the authenticated metadata and raw bytes delivered
to this successful load. It is not a new resident-buffer digest, current-disk
snapshot, tokenizer pairing or application capability. Source ownership ends
inside loading; the copied evidence remains with the immutable model.

The import choice occurs once at the coarse loading boundary. No numerical
kernel, graph, per-token hash, dynamic inner-loop dispatch or full-model copy is
added. Padded resident-weight planning/admission now precedes device/context
compilation. Queried maximum buffer length is still checked before weight-buffer
allocation. The inline optional identity occupies 168 bytes on this arm64 build,
inside the existing 64 MiB metadata policy; the complete model structure reports
832 bytes. Physical GPU weight accounting remains 142,955,990 bytes.

## Validation

An ordinary Metal-feature method checks the specific metadata-budget rejection
and missing-path I/O error before device access. Formatting, both strict checks,
all 329 CPU/353 Metal-feature tests, both iOS targets and the release admission
method pass. Initial targeted filters selected zero methods and receive no
coverage credit. The full Metal suite already passed the exact new method;
corrected debug/release invocations each run it once against the same checked
binaries. No source change, rebuild or GPU repeat was needed. The model field and getter add no default CPU numerical changes.

The existing three real Smol tests accept only absent/explicit legacy or explicit
audited test modes. Unknown mode values reject. Each actual case runs in a
separate owned process with a controlled environment; no parallel test mutates
its environment. Factory observations retain mode, context, identity and sizes
before numerical assertions. Nineteen controller corruption cases and twelve
valid mode/case/prefix combinations pass. The successful owned regression wrapper
does not claim invocation-failure retention qualification.

Both modes pass the unchanged selected checkpoint's independent per-array gates:
816 stage snapshots / 4,988,928 values per graph run, maximum observed absolute
stage error 16. The generation/state test selects eight IDs across its full-logit
and token-only schedules, checks four reference distributions and pending/reset/
changed-prefix behavior; maximum logit error is 0.0589752197265625. The separate
512-capacity/chunk-128 test preserves committed partial prefill, rejects overflow
and exactly matches its fresh full distribution. It remains a capacity/replay
witness, not an independent long-context numerical oracle.

Every audited factory reports the exact known config/index/header/reference
container hashes and 694 successful raw reads over 694 physical ranges. Every
legacy factory reports absent identity. Current logical session admissions are
225,474,820 bytes for the selected short cases and 254,862,386 for the capacity
case in both modes. Historical receipts retain their original accounting scope;
these current totals are not attributed solely to the new identity field.

On MacBookPro18,3 / Apple M1 Pro / 16 GiB RAM / macOS 26.6.2, completed resources
are separately recorded below, each within the existing 1.5 GiB diagnostic policy:

| Mode and case | Maximum RSS, bytes | Peak physical footprint, bytes |
| --- | ---: | ---: |
| Legacy graph | 206,159,872 | 203,489,952 |
| Legacy state | 186,482,688 | 175,669,896 |
| Legacy capacity | 260,030,464 | 249,168,568 |
| Audited graph | 207,929,344 | 204,997,304 |
| Audited state | 186,859,520 | 175,768,200 |
| Audited capacity | 260,472,832 | 249,332,408 |

These are regression process observations, not controlled inference latency or
throughput measurements. The current task reuses the existing numerical test
assertions and frozen arrays; it exports no new raw numerical arrays and claims
no new pairwise bitwise comparison between import modes or quality qualification.
The controller's aggregate summary sanity checks do not replace per-array gates.

## Next boundary

The successfully loaded model can now supply engine-owned historical content
evidence. Native capabilities still need separately consumed codec/profile
binding, device/session admission, response ownership and ABI/app integration.
This task adds no native ABI, app eligibility or phone grant and changes no
context/output capability. Gemma4 graph/source precision, representative
performance and 131072-output requirements remain open in the
[active plan](gemma4-implementation-plan.md).
