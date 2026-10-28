# Bounded real Phi3 Mac adapter

The pinned `mlx-community/Phi-3-mini-4k-instruct-4bit` checkpoint at revision
`5b3819ed6317784fb20eddeae9bed984f778d0d0` passes Toklet's selected native Mac512
engine-adapter checks. Original fused QKV/gate-up rows compose the existing
projection, normalization, ordinary rotary, attention and MLP blocks. There are
no new numerical kernels in this task. The [independent source/text reference](phi3-real-reference.md)
and [tiny CPU/Metal graph](phi3-dense-composition.md) remain separate evidence.

## Numerical and state results

On Apple M1 Pro, macOS26.6.2/25G83, the frozen
`MlxAffine/Compensated/Reference/Half` profile passes23,946,432 main comparisons
across1,282 complete source-comparable records. There are zero failed values;
maximum gate fraction is0.9399477806788512 and maximum absolute error1.5.
Acceptance uses each exact case/stage's already-frozen main/wide envelope:
`abs(native-main) <= max(1e-4,2*maxabs(main-wide)) + 1e-3*abs(main)`.
Wide comparisons are diagnostic only; their exceedance count is0. These
Half-storage independent profiles do not establish canonical Torch/MLX runtime
parity or original unquantized quality.

Six fixed probes cover1/3/4/17 tokens, offsets0/5/509, and a changed future tail.
Layers0,1,31 expose51 native stages per chunk; source scores/probabilities are
oracle-only. Requested chunks32/128/3 use `min(requested,512-offset)`.
Raw arrays and descriptor journals retain all1,288 captured records,
96,555,264 bytes, including the six state-only logit rows. Passing intermediate
values can therefore be independently rescored from preserved bytes.

Complete logit schedules and the causal first-three-row prefix agree exactly
as finite numbers, allowing equal signed zeros. Cached prompts preserve their
distributions. Teacher-forced histories, separate native four-token generation,
FullLogits/GreedyToken selection, partial-prefill discard/reappend, reset,
pre-cancellation and callback-failure recovery pass. Native generated raw IDs
are `[3,3,3,3]`; a forced stop checks the generic generation API. Natural source
EOS/text generation is not qualified by that control.

Two full512-token native state journeys, chunks32/128, have equal final logits.
Appending or setting513 tokens returns the exact capacity error and preserves
history, evaluated/pending positions and logits. This is a native state/schedule
check, not an independent512-row source accuracy or held-out quality test.
Absolute offset509 uses three local keys; it does not imply509 cached keys.

## Actual ownership and process fit

The original payload is2,149,644,288 bytes. Device-queried Half norm row stride
is6,144 bytes;65 norms add910 guard bytes to the packed resident allocation,
for2,149,645,198 bytes of weights and zero extra positional storage. The exact
source and resident loading admissions are2,216,818,688 and2,216,819,598 bytes.
One actual model load succeeds at the latter budget; one byte less rejects
before numerical allocation. No second live model is required.

Public plans agree with all four constructed session memories:

| Chunk | Readback | Exact admitted bytes |
| --- | --- | ---: |
|32|FullLogits|2,430,574,690|
|32|GreedyToken|2,430,318,218|
|128|FullLogits|2,467,081,954|
|128|GreedyToken|2,466,825,482|

Each exact-budget constructor succeeds; each one-byte-under request returns
the precise required/limit budget error. Diagnostic snapshots and F32 CPU
readbacks are already counted in session scratch. Extra source metadata,
descriptors, logit witnesses and I/O reserve18,941,952 bytes within32MiB; raw
serialization streams words rather than copying an entire tensor. Maximum
combined diagnostic admission is2,698,625,144 bytes. That allowance is not a
production UI/tokenizer budget.

The owned native process's observed maxima are resident1,787,068,416 bytes,
physical footprint2,516,764,952 bytes and OS lifetime physical maximum
2,517,715,224 bytes, below6GiB. RSS is sampled, not an OS lifetime RSS maximum.
Source and native models never coexist. These results demonstrate bounded
process fit; they are not throughput or latency qualification. One diagnostic
load took1.74049s, without a performance acceptance comparison.

## Review, checks and limits

Formatting, strict default/Metal all-target Clippy,410 default tests,
288 Metal-library tests, three runner tests and four portable checker controls
pass; suites overlap. Ordinary controls need no downloaded model or GPU.
Astra reviewed the fixed protocol, source, runner, checker, runtime/binary
identities and owned controller before execution. Prospective corrections
removed copied probe IDs and retained exact error/nonfinite failure witnesses;
no target outcomes determined inputs or tolerances. Astra independently rescores all23,946,432 native comparisons, recomputes every
source envelope and authenticates6,857 consumed identities and closed groups.
The final native evidence is approved for bounded retention and commit.

This closes selected B2 Mac engine-adapter acceptance. Declared4096 positions
and sliding window2047 remain distinct from implemented512. Wider requests,
LongRoPE and Online attention remain rejected. Full scalar FP32 loading still
exceeds6GiB. Production Phi Chat/codec integration, natural EOS, native text-error
rollback, iPhone14Pro, full declared context, held-out/original adoption quality
and comparative performance remain unqualified. This is one pinned artifact,
not arbitrary Phi-family support.

See [compact evidence](../benchmarks/results/2026-10-28-phi3-real-short-vertical-v1.json),
[TODO](../TODO.md) and [improvement record](improvements.md). Continue B3's
compact Gemma local/global composition; preserve the separate B5/M8 requirements.
