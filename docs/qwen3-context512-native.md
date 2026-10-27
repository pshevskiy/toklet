# Native Qwen3 C512 canonical diagnostics

This trial uses the frozen [complete CPU captures](qwen3-context512-reference.md)
and [precision diagnosis](qwen3-context512-precision.md). The existing saved FP32
contract remains the canonical target oracle with unchanged stage/logit gates.
FP64 comparisons are reported separately. The retained7282 CPU-envelope failures
and five distinct disjoint value pairs do not create a dual-oracle acceptance rule.

No production operator, precision, backend eligibility or position cap changes.
The selected Qwen3-0.6B eight-bit model uses explicit MlxAffine projections,
Compensated normalization, FullLogits readback and Reference attention. The model
has512 loaded positions for these trials, with absolute offset zero. Only one
numerical session is live at a time; immutable model weights are shared once.
Per-session numerical admission is bounded by1536MiB; completed process peaks
must remain within the separate6GiB Mac limit.

The frozen coverage matrix contains both original and future-tail512-token cases
for four schedules: chunk32, chunk128, uneven completed appends of
17/33/79/1/127/128/127 tokens, and chunk32 followed by the final single-token
append. Every chunk is captured. All512 rows of embedding, final norm, logits
and18 exposed states in layers0/1/27 are checked:57 states per chunk and5016
snapshots overall. Attention scores/probabilities are not exposed by this target
interface. Each actual native value is retained as little-endian FP32; Half stage
values are converted exactly by the diagnostic readback. The ordinary engine
continues to use its established storage representation.

Raw arrays and per-array journals precede numerical assertions. Independent cold
readback verifies exact journal keysets, shapes, byte extents, all per-state row
coverage, actual native-frame hashes and full consumed-oracle-array hashes. It
reconstructs both sets of numerical metrics and all512 raw argmax IDs for each
case/schedule. Original/future native prefixes through position510 must be
byte-identical per schedule. Model/reference/recipe/admission and generation
metadata are bound separately to the frozen plan, including strict integer and
boolean types; coherent relabeling is rejected.

The original-only generation witness admits510 prompt tokens plus two requested
outputs, cancels after emitting1879, resumes with0 and flushes that final pending
sample to512 committed positions. The frozen original sequence contains those
same final two IDs. It also checks pre-cancelled reset, overflow rejection with
exact error variants and unchanged state, injected output failure/retry, fresh
sampled-history replay and partial-prefill discard/reappend. A separately frozen
source-row hash binds final-row numerical comparison. It makes no claim about
cancelling an in-flight GPU command, and it does not reproduce the altered tail.

## Actual result, 27 October 2026

**The canonical numerical gate rejects this trial.** There are18,400 failed
values,2300 in every case/schedule, including2038 logits per case/schedule.
The separate FP64 diagnostic rejects37,040 values. Source CPU-envelope failures
remain7282. No gate or production capability is upgraded. Layer27 is the first
failing captured layer; layers2–26 are not observed. Passing token choices alone
does not waive numerical or quality requirements.

All4096 native argmax rows match the frozen canonical IDs. Each of57 complete
512-row captured state arrays is byte-identical across all four schedules for
each case; original/future prefixes are also byte-identical. Thus these captured
schedule comparisons show no state difference. The pending/cancel/output-failure/
retry/overflow/fresh-replay/discard contracts pass. The generation final row passes
both reported numerical comparisons, with151,936 values per contract. The overall
trial compares1,008,357,760 values against each oracle including that row.

Formatting, strict default/Metal Clippy and368 ordinary CPU tests pass, with19
expected ignores. Thirty-one cold Python controls pass. The ignored Rust
qualification assertion remains strict and fails for canonical numerical
rejection. Independent readback successfully reconstructs all5016 snapshots,
complete row coverage, typed numerical metrics/examples, all argmax IDs, source
byte hashes, metadata and state-prefix evidence. Complete actual binary output
is4,033,431,040 bytes including the final generation row.

The first reader compared shortest-decimal FP32 examples as FP64, causing a
representation-only replay failure. The corrected reader compares only those
declared FP32 fields by exact bits, preserving signed zero and rejecting one-ULP
changes. Counters, positions, binary64 deltas/bounds, metrics and numerical gates
remain exact. The repaired trial reuses all unchanged GPU/compiler/raw evidence;
it reruns only cold units and readback. Both the original numerical rejection
and the initial reader failure remain archived.

Apple M1 Pro/MacBookPro18,3,16GiB/macOS26.6.2(25G83), Rust1.90 release:
maximum logical model/session/diagnostic admission is1,081,145,016 bytes with one
live session. Native process peaks are1,190,167,944 physical/636,125,184 RSS bytes,
below6GiB. Readback peaks are272,909,224 physical/279,019,520 RSS, below512MiB.
These are functional diagnostic measurements, without performance or whole-app
fit qualification. All12 build/initial-trial/recovery groups are absent.

See the [public receipt](../benchmarks/results/2026-10-27-qwen3-context512-native-v1.json).
Next perform one bounded same-input Q/K/V projection diagnosis at selected layer27
coordinates, using saved native/reference input-norm rows and authenticated packed
weight rows. Separate inherited input differences from local reduction and Half
rounding, including passing-coordinate controls. This requires no new all-layer
CPU trace or upstream-framework prerequisite, and does not justify wider gates.

Source parity, unquantized/held-out quality, useful performance, the full40960
model window, larger app codec/text/event limits and physical iPhone qualification
remain separate in the [active plan](long-reply-engine-plan.md).
