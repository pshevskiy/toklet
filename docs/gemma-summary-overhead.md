# Gemma summary overhead with bounded larger cohorts

The two fixed 100-pair headless cohorts meet the original 1% acceptance gates.
Astra approves scoped code and evidence retention with no findings. Application
summary mode stays opt-in: controller publication, visible text/render and other
workloads have separate qualification. This changes the explicit benchmark only;
worker/native numerical code, model weights and the production ring are unchanged.

## Workload and result

Official `google/gemma-4-E2B-it` at
`70af34e20bd4b7a91f0de6b22675850c43922a03`, BF16 Mac G2, Apple M1 Pro.
The pinned codec/asset manifest is
`654dc6dfd66fd893a24f0faa39355b71d5e7ffc3f53c2dac394dfcfd9ed95298`.
This is the original short Chat: 15 prompt tokens, reply cap 12, eight visible
IDs plus one hidden stop, capacity 512 and chunk 32. It does not qualify summary
overhead on a 512-token prompt or the full numerical workload matrix.

Each fresh process alternates disabled/summary order for three warmup pairs
and 100 measured pairs. Both processes finish: 412 requests, 400 measured,
206 complete summary records. All outputs match the original eight IDs;
history/committed/pending stay 24/23/1. Nothing is excluded or retried.

| Metric | Cohort 1 median ratio overhead | Cohort 2 median ratio overhead | Cohort 1 paired-median 95% upper | Cohort 2 paired-median 95% upper |
| --- | ---: | ---: | ---: | ---: |
| Callback TTFT | 0.219% | 0.237% | 0.208% | 0.457% |
| Complete caller request | -0.027% | -0.064% | 0.458% | 0.477% |

The median ratio and paired-median bootstrap are different statistics. Each
must satisfy its unchanged <=1% gate, for both metrics in both cohorts. Bootstrap
seed 1 and 10,000 resamples are unchanged. Small negative point estimates are
noise-scale observations, with no speedup claim or tail-percentile inference.

The earlier fresh ten-pair cohorts remain retained and unqualified: TTFT upper
bounds were 2.271%/2.123%, despite 0.069%/0.246% median overhead. The larger
sample count and stopping rule were declared before collecting its outcomes.
[Original result](../artifacts/lfm3-tokens2-2026-11-13/gemma-observer-original-closure-1.json),
[prospective campaign](../artifacts/lfm3-tokens2-2026-11-13/gemma-stronger-overhead-prospective-1.json).

## Bounded collection and clocks

The runner accepts `--measured-pairs 10..100`; omission preserves ten pairs,
three warmups, two processes and the original native-ring export. Existing real
ten-pair statistics replay exactly. All 40 portable controls across the three
existing CI modules pass in normal and optimized Python modes.

For larger runs, one snapshot follows each complete pair, outside BOTH timed
submit-to-idle intervals. An idle disabled mode is temporarily enabled for export
and restored. Each selected native row is copied byte for byte; extraction has
synthetic numeric-spelling, missing-ID and duplicate-ID controls before loading.
The production ring stays at 32 records. The version-1 benchmark capture is
limited to 512 KiB, with at most 103 native rows of at most 4 KiB each. Its header
nests the final native manifest semantically unchanged; 71 ring overwrites are
disclosed per cohort while the external capture retains every enabled request.

All 412 request IDs are unique and all 206 native rows join exactly. Independent
replay verifies status, owners/state, counts, nested monotonic timestamps,
caller TTFT/request, worker queue/TTFT/request clocks, complete phase counts,
footprint bounds and the statistics. Native phase durations overlap caller
clocks and are not added. The last host callback timestamp remains unavailable;
its aggregate decode rate cannot be directly reconstructed from this schema.

## Checks and provenance

Strict Swift 6 compilation with complete concurrency and warnings-as-errors,
pre-model controls and ten real conformance records pass. Both cohorts pass
AC/Low-Power-off/nominal-thermal gates and the 6 GiB engine ceiling. The maximum
observed owned-tree charge including helper reserve is 5,009,003,440 bytes;
compiler charge is 1,434,468,352 bytes under its 3 GiB cap. These are sampled,
conservative process observations, not a continuous allocator cap.

The current clean-source native archive binds 462 Rust/Cargo input hashes;
489 total input endpoints and the full three-file source diff match before/after.
Rust/feature/dependency inputs are unchanged, so their required fmt, strict
CPU/Metal Clippy and 786 CPU/946 Metal tests are reused. No kernel or production
observer change is included. Controller/UI/render, other profiles and iPhone
remain separate, and the default application mode is unchanged.

[Portable/dependent checks](../artifacts/lfm3-tokens2-2026-11-13/o1-stronger-dependent-1/checks.json),
[default real-data replay](../artifacts/lfm3-tokens2-2026-11-13/gemma-stronger-default-replay-1.json),
[independent full audit and raw seals](../artifacts/lfm3-tokens2-2026-11-13/gemma-stronger-independent-audit-1.json),
[native archive binding](../artifacts/lfm3-tokens2-2026-11-13/gemma-current-staticlib-1.json),
[checks/review/publication](../artifacts/lfm3-tokens2-2026-11-13/gemma-stronger-publication-checks.json).
