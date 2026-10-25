# Native Gemma4 basic-hello generation diagnostic

The closed `basic-hello` diagnostic composes the
[native text codec](gemma4-native-codec.md),
[named short E2B model](gemma4-short-scalar.md) and
[generation-state owner](gemma-generation-state.md) against the independently
frozen [formatted mathematical witness](gemma4-basic-chat-reference.md).
The original raw profile remains the default. This example introduces no app
profile, native ABI or context/output capability.

## Workload and ownership

`gemma4_generate --profile basic-hello CHECKPOINT NEW_OUTPUT_DIRECTORY` admits
only the frozen user message `Hi`, exact ten-token rendering, stops1/106/50 and
16 sampled tokens per schedule. One model/codec/session serves uninterrupted and
split2+remaining schedules, with reset between them. A split resumes only after
an output limit, uses the same stop set, and subtracts sampled tokens from the
16-token total. A natural stop in either call remains hidden and pending.

Before asset reads, subtract1GiB for the codec and16MiB for diagnostic/text
ownership from the6GiB process admission ceiling. Pass the remaining numerical
budget to payload loading and session creation; model/store ownership is counted
once in the session report. The diagnostic reserve covers bounded input/render
and ID buffers, both keep/skip decoded Strings, JSON/ledger serialization and a
64KiB file writer. Up to32 full logits rows occupy32MiB on disk, streamed without
retaining a full-vocabulary copy per step. Disk bytes are separate from the
resident diagnostic reserve. Numerical execution still admits only32 positions.

The concrete sampling wrapper captures the full distribution before appending
the sampled ID. Failure preserves partial descriptors, completed reports and
valid/pending history. An underlying capture I/O failure remains the primary
error. Decode only emitted IDs after generation; no append-only text stream or
quality claim follows from complete-string decoding.

## Actual Mac result

On Apple M1 Pro/MacBookPro18,3 with16GiB RAM and macOS26.6.2/25G83, both schedules
select the same11 IDs as the frozen witness and emit exactly
`Hi! How can I help you today? 😊`. Hidden stop106 remains pending; each final
history has21 IDs,20 evaluated and one pending. The split's second call reuses11,
evaluates its one pending input, samples nine and emits eight. All22 complete
logit rows are bitwise identical between schedules.

All44 complete-logit comparisons pass unchanged `atol=1e-5, rtol=1e-4` over
11,534,336 paired values. Maximum error-to-bound fractions are
0.46041798867955425 against FP64 and0.0013559174399768907 against staged FP32.
The22 raw rows occupy23,068,672 bytes in ignored artifacts, authenticated by the
[receipt](../benchmarks/results/2026-10-25-gemma4-basic-chat-generation-v1.json).
These selected rows do not waive earlier full-graph precision failures.

The numerical report admits2,765,262,700 bytes, including model/store ownership
of2,671,706,564 bytes. Adding codec1,073,741,824 and diagnostic16,777,216 gives
3,855,781,740 combined admitted bytes. Measured whole-process resident/physical
peaks are2,273,132,544/2,900,626,216 bytes, below6GiB. The188.52-second combined
load/two-schedule/output observation is not a controlled latency benchmark.
Raw timings are retained without representative performance qualification.

## Verification

Seven focused offline/release methods cover original raw capture behavior,
formatted prefix/row boundaries, malformed profile selection, reserve-minus-one
rejection before asset access, first/second/final-slot stops, no-stop output
limit and a second-call callback failure preserving the first completed report.
Formatting, both strict checks,307 CPU tests,329 Metal-feature tests, both iOS
build targets, release compilation and all seven release methods pass.
The earlier six-method preliminary attempt is retained before the direct
formatted capture-bound regression was added.

The controller's actual complete observation-verification block and final
classification run over a synthetic22-row packet, including all44 numeric
comparisons. Wrong decoded text, leaked stop, wrong reused/prefill counts,
combined reserve and truncated data reject. A finite changed value that preserves
argmax and identical schedules completes capture while recording numerical
failure. This synthetic regression cannot qualify native model execution.

## Limits

This is a bounded Mac diagnostic. Earlier full-graph precision/source, GPU
intermediate and long-rotary failures remain unwaived. Broader basic-text quality,
append-only streaming, resident Metal, representative performance, mobile memory,
phone execution, app integration and131072 output tokens remain open. The official
checkpoint declares131072 total positions, which cannot hold131072 output tokens
plus a nonempty prompt. No whole G2/G3/G4 gate is marked complete.
