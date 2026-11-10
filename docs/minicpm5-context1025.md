# MiniCPM5 1,025-position numerical diagnostic

10 November 2026. **Diagnostic and numerical rejection retained with Astra
approval.** The selected Heretic affine8 route fails 82 of 1,773,020,160
original-gate comparisons. Quality scoring and managed adoption remain open.
Ordinary admission stays at 512 positions; this explicit diagnostic API grants
no broader model, context, device or performance qualification.

## Contract and implementation

The [prospective contract](../artifacts/minicpm5-context1025-2026-11-10/prospective-contract-1.json)
was frozen before deeper CPU/native outcomes. Its SHA256 is
`b5464b7d67cee7fa8d0df28afc5fb0e68f0c5d550232e572f93858dc4f61bad3`.
It retains all 51 original six-case gates byte for byte, both independent CPU
profiles, and prefixes of 128, 512 and 1,025 IDs from the unchanged complete
[quality input payload](minicpm5-quality-inputs.md). No gate is recalibrated from
these results. Both original 10k-target/100-task comparisons and output caps stay
binding; no quality score is produced here.

A separate CPU wrapper copies the original NumPy graph with only its admission
literal changed from 512 to 1,025. An AST test restores that literal and requires
exact graph identity. The original implementation stays unchanged. FP32/FP64
reduction profiles retain staged Half activations and final FP32 logits. Capture
covers embedding, final normalization, logits and 16 stages at layers 0, 1 and 41;
306 arrays occupy 2,640,983,040 bytes. Native scores/probabilities are outside this
frozen comparison inventory.

`Model::load_mlx_minicpm5_context_diagnostic` validates the fixed Llama geometry
(H=2048, I=6144, Q=16, KV=2, D=128, 42 layers, vocabulary 130560), ordinary split
RoPE, untied output, 296 affine8/group64/BF16 matrices and 85 BF16 norms. Context
admission is 1..=1025 and the model-load budget is at most 6 GiB. Session creation
requires Compensated normalization and Reference attention; MLX affine projection
is preserved. Callers authenticate the artifact separately; the CLI checks the
pinned config, index and weight files before GPU creation and again after execution.
No audited content or managed-app identity is granted by this geometry loader.

The `minicpm5_probe context` CLI uses native chunks 1 and 32, the serial RMS-store
baseline and full diagnostic readback. It streams 87,624 arrays/3,546,040,320 bytes
and 1,718 completed chunk records. Immutable model weights are shared across
session-owned KV/scratch; the existing generic session planner remains in use.
Partial discard/exact-prefix reuse, fully cached prompts, pending/boundary append,
invalid/oversized rejection, changed-prefix discard/restore and reset/replay pass
for all six case/chunk combinations. Corrected-last logits are separately compared
against the original CPU gates; same-schedule reset/replay is bitwise equal.

## Numerical result

| Quality prefix | Values compared against both profiles, both chunks | Failures |
| --- | ---: | ---: |
| 128 | 136,706,048 | 14 |
| 512 | 545,257,472 | 20 |
| 1,025 | 1,091,056,640 | 48 |
| Total | 1,773,020,160 | 82 |

Failures occur in layer 1 gate projection for all prefixes, plus layer 0 down
projection and layer 41 attention projection at 1,025. Both chunk schedules have
the same failure counts and saved witness values. Final logits and corrected-last
logits pass their original gates in these captures. The intermediate failures
still reject numerical qualification.

The largest error/limit ratio is 1.76090785 at layer 0 down projection, row 785,
column 1108: native 0.134521484375 versus FP32-profile 0.1356201171875, with limit
0.0006239013671875. All 442,863,360 paired CPU coordinates have overlapping
original permitted intervals. Unlike the earlier Qwen deeper-context results,
there is no demonstrated incompatibility between the two CPU bounds. The cause
of the native/reference differences needs investigation; this report does not
attribute them to a particular kernel or reject the model artifact itself.

[Complete comparison](../artifacts/minicpm5-context1025-2026-11-10/comparison-2.json),
[independent cold replay](../artifacts/minicpm5-context1025-2026-11-10/comparison-replay-checks-2.json)
and [48 raw failure-witness replays](../artifacts/minicpm5-context1025-2026-11-10/failure-witness-replay-1.json)
retain the exact outcomes. The witness replay uses stdlib `struct` and scalar
binary64 math, plus 96 inside/outside predicate controls. Four witnesses per
failing group are retained, so the 48 records do not enumerate all 82 failures.

## Resource observations and verification

| Owner | Final `wait4` RSS peak | Maximum sampled physical footprint | Samples |
| --- | ---: | ---: | ---: |
| Independent CPU producer | 2,029,486,080 B | 1,321,388,192 B | 388 |
| Native producer | 2,291,482,624 B | 2,812,414,472 B | 4,316 |
| Offline comparison | 133,152,768 B | 67,699,384 B | 75 |
| Fresh offline replay | 176,504,832 B | 65,241,856 B | 76 |

All owned process groups close, source pins remain unchanged, sample errors are
empty, and the 6 GiB guard and 2 GiB disk floor pass. Native sampled lifetime
physical peak is 2,827,110,920 B; final physical peaks are unavailable. The largest
native session ledger plus the explicit 32 MiB application allowance is
2,903,028,972 B. Codec state is not loaded by this numerical diagnostic; its
separate 512 MiB policy is unchanged. Raw capture limits remain 3 GiB CPU/4 GiB
native. Declared 131,072-position source metadata does not qualify that capacity.

The consumer checks an externally hashed evidence seal; exact CPU/native
commands, PID ownership, process lifetime, complete sample counts and receipt
peaks reconcile. It verifies finite values, original gates, every array shape,
row/span/file coverage, consumed regular descriptors and path identity, frame
hashes, completed timings and balanced session ledgers. CPU leaf-array digests
are checked while comparing the two profiles. Native journals are consumed from
pinned descriptors and checked again at EOF. A completed numerical rejection
returns a completed report with `numerical_gates_passed=false`; it is not adoption.

Both cold comparisons use a frozen 2,822-file NumPy 2.3.2 runtime, with fresh
absolute cache prefixes, no MLX/Torch/GPU reference execution and actual import/
native-image origin checks. Complete reports are identical after removing only
the intentionally different `admitted_runtime.pycache_prefix` field.

- Fresh fmt, strict CPU/Metal Clippy, 681 CPU tests, 827 Metal tests and nine
  example tests pass. The 49 CPU/337 Metal ignored tests are not claimed run.
  [Required checks](../artifacts/minicpm5-context1025-2026-11-10/required-checks-2/checks.json).
- Native68 builds all three arm64 slices with 46 C exports each; CPU/Metal tiny C
  smokes pass. This does not qualify real-model execution on iPhone.
  [Package](../artifacts/minicpm5-context-native68-2026-11-10/build.json),
  [C smokes](../artifacts/minicpm5-context1025-2026-11-10/native-smoke-68/checks.json).
- Seventeen portable tests, 15 actual release-CLI rejection controls and 51 altered
  saved-metadata/sample rejection controls pass. Structural controls do not
  authenticate coherently fabricated measurements.
  [Portable](../artifacts/minicpm5-context1025-2026-11-10/portable-all-2.json),
  [CLI](../artifacts/minicpm5-context1025-2026-11-10/cli-negatives-2/checks.json),
  [metadata/samples](../artifacts/minicpm5-context1025-2026-11-10/report-controls-3/checks.json).

Capture timing includes diagnostic readback, journaling and functional replay;
it does not qualify ordinary latency, telemetry overhead or a performance gain.
The default modes, ordinary 512/chunk32 policy, full quality gates and deferred
UI/iPhone scope remain unchanged.

## Astra findings and targeted correction

Astra found two P2 consumer gaps: impossible load/serial GPU timelines could be
accepted, and zeroed fixed memory components could pass after balancing totals.
The reader now requires positive integer load time preceding the first chunk,
ordered GPU windows and disjoint load/host/GPU spans within the run. It derives
every ledger component from fixed shapes, guards, snapshots/readback and the
frozen arm64 metadata sizes (Buffer16/Cache32/Views120/Snapshot72 bytes).

All 1,718 formerly accepted zeroed ledgers and 1,717 repeated GPU windows after
the first now reject. Thirteen complete resealed datasets cover altered load,
individual/all balanced memory omissions, GPU order and excessive serial span.
The actual original series passes. Seventeen portable tests and a fresh set of
51 metadata/sample negatives pass. Both corrected cold full comparisons reproduce
the original numerical payload exactly; CPU/native producers and gates remain
unchanged. Astra approves scoped diagnostic retention and commit, with both
findings resolved and no remaining findings.
[Final review](../artifacts/minicpm5-context1025-2026-11-10/astra-review-result-2.json),
[review/commit](../artifacts/minicpm5-context1025-2026-11-10/publication-checks.json),
[Initial review](../artifacts/minicpm5-context1025-2026-11-10/astra-review-result-1.json),
[counterexamples](../artifacts/minicpm5-context1025-2026-11-10/astra-validation-gaps-1.json),
[corrected controls](../artifacts/minicpm5-context1025-2026-11-10/astra-controls-1/checks.json).

The original consumer/test sources are preserved by reviewed patch and hash-
checked snapshots. A snapshot-preparation script initially indexed an empty
leading diff section; it exited before launching a comparison and is corrected.
[Snapshots](../artifacts/minicpm5-context1025-2026-11-10/comparator-before-astra-1/snapshots.json),
[preparation failure](../artifacts/minicpm5-context1025-2026-11-10/comparison-2-preparation-failure-1.json).

## Retained attempts and next work

The [first native attempt](../artifacts/minicpm5-context1025-2026-11-10/native-attempt-1-interpretation.json)
rejected at the ordinary 512 geometry guard after artifact hashing, before GPU
creation/upload/execution. Its incorrect constant capture-completed flag remains
in the raw failed receipt; `passed=false` is authoritative. The separate loader
and fresh required/native checks precede the successful second capture.

The [first CLI-negative runner](../artifacts/minicpm5-context1025-2026-11-10/cli-negatives-1/interpretation.json)
misidentified the last remaining input as an output during the arity control;
the CLI correctly rejected. The corrected second runner passes. The first
metadata-control report counted five positive groups instead of four; its
[interpretation](../artifacts/minicpm5-context1025-2026-11-10/report-controls-1/interpretation.json)
and corrected second report retain that reporting error. Producer inputs and
numerical data are unchanged.

A pre-review source check initially resolved relative native source pins against
the protected root instead of the isolated checkout. Its
[failure interpretation](../artifacts/minicpm5-context1025-2026-11-10/pre-review-attempt-1.json)
is retained; the corrected check uses the build checkout for source and the
published package for output pins. A subsequent runner schema assumption about
the test-count document is also corrected; that document reports per-suite
counts rather than a top-level `passed` flag. No required-test or source mutation
occurred. [Second runner log](../artifacts/minicpm5-context1025-2026-11-10/pre-review-console-2.log).

User-requested [disk recovery](../artifacts/minicpm5-context1025-2026-11-10/disk-cleanup-2/result.json)
recovers about 51 GiB net while preserving the live native process, all source
inputs, models, evidence and protected user work. Its cleanup-controller
interrupt/resume and compressed manifests are retained separately.

Next: isolate one failing projection from its captured input, reconstructed
weights and original profile bounds; test one numerical correction without
relaxing gates. Then repeat the original complete numerical and quality gates
before deciding adoption. The whole improvement goal remains active.
