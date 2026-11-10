# MiniCPM5 bounded text codec and console metrics

Status: Astra approves the scoped implementation and retained diagnostic route.
[Review/commit receipt](../artifacts/minicpm5-text-codec-2026-11-10/publication-checks.json).
This is an explicit diagnostic Mac route. The full model/quality/adoption goal
remains open. The user deferred screen capture and physical iPhone work.

The [prospective contract](../artifacts/minicpm5-text-codec-2026-11-10/prospective-contract.json)
binds the accessible Heretic MLX8 artifact at `50ab1b22`, five exact text assets,
512 MiB codec allowance, separate 4 MiB stream and 32 MiB application allowances,
6 GiB total engine ceiling, and session capacity512/chunk32/offset0. Kernels,
precision, numerical tolerances, the ordinary normalization implementation and
all model-adoption gates are unchanged.

## Text semantics and independent evidence

The Rust [codec](../src/app_chat/minicpm5.rs) parses the exact authenticated
`tokenizer.json` bytes captured through a bounded regular-file handle; all five
asset hashes and sizes must match. It preserves 130,560 IDs, the 510 added-token
inventory and all flags, source ByteLevel processing, and stop union `[1,130073]`.
Thinking IDs8/9 are nonspecial and remain in decoded text. Raw special-token
encoding preserves the source BOS postprocessor; chat rendering already includes
BOS and therefore disables additional special tokens.

The admitted no-tools domain has typed string messages, optional initial system,
alternating user/assistant, at least one user and explicit boolean thinking and
generation flags. A generation prompt requires a final user. Empty strings and
whitespace are admitted. Unknown/duplicate fields, tools/media/reasoning payloads,
unsupported C0 controls and the exact510reserved literals reject. Wire/canonical/
rendered/decoded text is bounded at64KiB, codec IDs at65,536 and messages at63.
Full-decode admission uses a conservative per-ID byte upper bound; some long
Unicode sequences can reject before allocation despite a smaller exact decode.
System/user strings are verbatim. Every plain assistant turn uses the original
empty-think wrapper and removes leading LF characters only. The source tokenizer
has BatchLongest-left padding with `</s>`; single-sequence and one-item batches
retain their original unpadded IDs. No source normalization or policy is cleared.

The independent [producer](../tools/reference/minicpm5_text_codec_reference.py)
uses admitted Jinja2/Tokenizers packages and the unchanged original template.
It never imports MLX or executes the model. Its exact upstream stop-union source
slice is a separate witness, not qualification of the entire MLX runtime.
Two initial91-record captures preceded native execution. A failing renderer test
exposed the mistaken earlier-assistant assumption; six source leading-LF cases
were appended while all91old records stayed unchanged. The real tokenizer then
rejected the initial no-padding assumption. An exact source-padding record and
six single-sequence/batch controls were appended; all97old records remain byte
identical. Both revised captures agree on the final98-record
[fixture](../tests/fixtures/minicpm5-text-codec-v1/manifest.json).
Original contract wording, failed binaries, source snapshots and logs are retained:
[renderer correction](../artifacts/minicpm5-text-codec-2026-11-10/source-semantics-correction-1.json),
[padding correction](../artifacts/minicpm5-text-codec-2026-11-10/source-padding-correction-1.json).
The appended records are accurately identified as collected after native code
existed; original source bytes and previous observations were never rewritten.

## Console route and timing boundaries

Build the `minicpm5_probe` example in release mode with the Metal feature. Its
explicit text command is:

```text
minicpm5_probe text MODEL MESSAGES.json OPTIONS.json NEW-OUTPUT serial|parallel-store|parity [MAX-NEW=16]
```

Messages and options must be regular nonsymlink files within64KiB; the output
directory must be new. `MAX-NEW` is1..256 and prompt plus output must fit512tokens.
The serial mode preserves ordinary full-logit execution; parallel-store selects
the previously reviewed RMS optimization. No managed route or native C ABI is
added. Input validation/tokenization precedes numerical loading. The codec reserve
is deducted from the numerical load/session limit throughout construction,
including upload temporaries. Shared weights are counted once for the two-session
parity diagnostic; its combined ledger must also fit6GiB.

`text.json` retains all13requests: three warmups and ten measured samples, one
loaded model/codec/session, reset-empty history for each request. Reports separate
model load/conversion/compilation, codec construction, preflight tokenization,
session setup, rendering, tokenization, stream construction/finish, completed
prefill/decode chunks, selection, internal decision TTFT, first decoded text,
throughput, sampled/committed/pending state, stop reason and admitted owners.
The request clock starts before rendering and ends after stream finish. Reset,
telemetry capacity reservation, JSON and separate full-decode validation are
outside it. The initial request can have multiple32-token prefill chunks.
First decoded text means the first nonempty decoder chunk; console publication,
paint, visible TTFT and allocation count remain unavailable. GPU timestamps
represent completed command work; host fields are disjoint within completed wall
clocks. The [offline reader](../tools/bench/minicpm5_text_performance.py) checks
per-event, per-request and whole-owned-lifetime enclosure, exact work/owner
ledgers and complete native-parity token/text binding. It rejects changed report
bytes even when a plausible summary could otherwise be calculated.

`parity.json` compares all130,560 FP32 logit bits at every sampled step between
serial and parallel-store, including multi-chunk prefill and final pending token.
It is native implementation equivalence, not an extension of the independent
CPU numerical oracle or a model-quality gate.

## Checks and measurements

Fresh fmt and both strict all-target Clippy checks pass, with679CPU/821Metal tests
and36portable reader/reference tests after the Astra causal-window fix. The first Metal test attempt hit the disk
guard; verified inactive compiler-cache cleanup recovered27.1GiB and the unchanged
phase passes on rerun. The failed attempt remains in
[first combined check evidence](../artifacts/minicpm5-text-codec-2026-11-10/required-checks-combined-1.json);
[final corrected-source checks](../artifacts/minicpm5-text-codec-2026-11-10/required-checks-3/checks.json).
The release codec passes98source records and66stream schedules; final RSS153,698,304B
and physical peak133,693,992B remain below512MiB under the explicit
`MallocLargeCache=0` diagnostic profile.
[Fidelity receipt](../artifacts/minicpm5-text-codec-2026-11-10/codec-fidelity-3/checks.json).
Nine real CLI rejection controls pass before numerical admission, including
oversize/symlink/typed inputs and an over-capacity tokenized prompt.
[CLI controls](../artifacts/minicpm5-text-codec-2026-11-10/cli-controls-2/checks.json).

The first native66 package and CPU/Metal C smokes pass; two initial codec
component campaigns pass. The unchanged660-phase cache stress then rejects a
25,099-byte ASCII batch because the initial per-token decode bound charges every
ASCII character three bytes. The bound now charges ASCII one byte, retaining
three for other mapped bytes and the original UTF-8 fallback length; limits and
stress work are unchanged. A new independent ByteLevel decoder regression covers
all256alphabet characters, pairs, raw fallback and ASCII admission. Failed source,
binary and stress evidence are retained in the
[bound correction](../artifacts/minicpm5-text-codec-2026-11-10/decode-bound-correction-1.json).
Final corrected-source checks and native67all-three-slice/C smokes pass.
The unchanged cache stress completes660phases/22,000distinct250-byte words across
three rounds in2.648s, with215,121,920B peak RSS and133,874,240B physical peak.
Two codec campaigns each retain3warmups/10measured fresh constructions. Median
construction is194.97/193.30ms, rendering12.0/12.9µs, tokenization200.0/172.2µs,
fixed Unicode full decode5.19/5.08µs and incremental decode including construction
and finish27.7/27.3µs. These are descriptive component clocks, not engine speedups.
[Final codec samples/resource summary](../artifacts/minicpm5-text-codec-2026-11-10/codec-summary-3.json).
All eight matched request cohorts pass (80measured requests plus24warmups),
with exact token/text/state binding to the two prior parity captures. Across both
parity workloads4,177,920complete FP32 logit values compare bitwise. Three actual
request reports and both parity reports also reject83resealed altered copies;
all primary bytes stay unchanged. Astra found that first-text validation allowed
later GPU work inside its upper window. The reader now subtracts the next
decision's known completed work and selection, or final stream finish for a
last-decision push; finish-only text remains inside finish. Six old-accepted
actual counterexamples now reject, and portable controls cover all three cases.
The consumer and its test are the only two amended frozen inputs; Rust/native,
original producer/contracts, binary, raw samples and derived metrics are exact.
[Causal controls](../artifacts/minicpm5-text-codec-2026-11-10/first-text-controls-1/checks.json),
[consumer amendment](../artifacts/minicpm5-text-codec-2026-11-10/consumer-causal-amendment-1.json).
[Altered-report controls](../artifacts/minicpm5-text-codec-2026-11-10/text-controls-2/checks.json),
[offline revalidation](../artifacts/minicpm5-text-codec-2026-11-10/revalidation-2.json).

On Apple M1 Pro16GiB/macOS26.6.2, under `MallocLargeCache=0`, AC power, nominal
thermal and Low Power Mode off cohort boundaries:

| Workload | Serial request seconds, runs1/2 | Parallel-store seconds, runs1/2 | Request reduction | Decode-rate increase |
| --- | --- | --- | --- | --- |
|14-token, thinking off|2.3683 /2.3706|1.9218 /1.9364|18.85% /18.32%|29.05% /28.16%|
|38-token, thinking on/multi-chunk|3.7259 /3.8409|3.2681 /3.3881|12.29% /11.79%|21.05% /20.82%|

Both pairs of both workloads clear the original5%request screen, exceeding the
largest request IQR1.84% and between-run variation3.67%. Internal decision TTFT
falls5.62–6.09% and3.14–4.09% respectively; first decoded text and prefill throughput
also improve. These are completed loaded-model text requests including CPU text
phases, not a new default/model-quality/summary-overhead qualification.
Model load remains separately7.844–9.273s, codec construction243–253ms and session
setup9.59–13.18ms in these cohorts. Exact aggregate admission is3,360,498,514B
serial and3,427,607,378B parallel, including the separate codec/stream/application
allowances. Maximum request-cohort wait4RSS is2,609,004,544B and maximum sampled
lifetime physical footprint2,763,180,528B, both below6GiB; final physical peak is
unavailable. Reset, serialization, validation and warm filesystem behavior stay
explicit. No sustained or tail/continuous-stability claim comes from ten samples.
[Raw samples and fixed evaluation](../artifacts/minicpm5-text-codec-2026-11-10/text-performance-evaluation-1.json).

Astra approves after the sole causal-enclosure finding is resolved. The
[frozen text workload contract](../artifacts/minicpm5-text-codec-2026-11-10/text-workload-contract.json)
selects a14-token thinking-off prompt and38-token thinking-on history with leading
LF controls,16maximum new tokens, two serial/parallel comparisons per workload,
AC/nominal thermal/Low Power Mode off boundaries and the unchanged≥5%retain
policy. This contract predates all native text request outcomes and is unchanged.

Default allocator fit, original10k-token/100-task quality, deeper independent
numerical qualification, reproduction of296quantized matrices, default/managed
adoption, summary overhead≤1%, visible-frame timing and device qualification
remain open. The existing source-authentication and short numerical/RMS evidence
retain only their documented scopes.

A read-only root protection probe initially compared Git `%s` with the saved
`%gs` reflog format; using the original format confirms all12user files, both
stashes, root/main revisions and app link are exact. No mutation followed the
failed assertion. [Protection proof and failed inspection](../artifacts/minicpm5-text-codec-2026-11-10/root-protection-midrun-1.json).
