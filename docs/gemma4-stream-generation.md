# Short native Gemma4 generation with text streaming

The explicit `basic-hello-stream` profile in
[`gemma4_generate`](../examples/gemma4_generate.rs) composes the
[checked short scalar model/reply](gemma4-basic-chat-generation.md) with the
[bounded no-Strip text stream](gemma4-native-stream.md). The existing `raw` and
`basic-hello` profiles retain their request policies and memory reserves.
This is a fixed diagnostic, with 32 numerical positions and a 16-sample ceiling.
It grants no app, native ABI, phone or long-output capability.

## Ownership and scheduling

The pinned codec and numerical model load once. A session is reset between
three schedules, each with a fresh stream and a diagnostic ledger bounded to
16 token chunks plus one terminal chunk. Only callback-delivered visible IDs
enter the text stream. Each borrowed chunk is copied into that ledger before
another mutation or comparison. Whole reply decoding is a bounded diagnostic
comparison, rather than the delivery implementation.

| Schedule | Request policy | Final numerical state | Visible text |
| --- | --- | --- | --- |
| Uninterrupted | At most 16 samples, selected model stops | 20 committed, one pending hidden stop 106 | Hi! How can I help you today? 😊 |
| Split | Two samples, continue only after OutputLimit with the remaining budget | Same exact final state | Same exact reply |
| Cancel | Request cancellation after the second successful visible delivery; no continuation | 11 committed, one pending sampled ID; 12 history IDs including prompt | Hi! |

One stream spans both split requests. The internal OutputLimit does not flush
it: doing so could incorrectly end a byte-fallback run. Each successful
terminal model stop, cancellation or final output limit flushes once. Hidden
stop 106 is sampled into numerical history but never delivered to text.
A failed callback does not roll numerical state back. Retain its completed
reports, visible IDs, copied chunks, counters/status and original error; a
poisoned stream needs a fresh owner and replay. The terminal sink error is also
preserved after successful numerical reports.

Before any assets are read, subtract separate codec 1 GiB, stream 4 MiB and
diagnostic 16 MiB reserves from the existing 6 GiB process policy. Pass the same
remaining numerical budget through payload loading and session construction.
Count the borrowed codec once. The diagnostic reserve covers bounded ledgers,
serialization, rendering and complete-decode comparisons; streamed full-logit
files are separately bounded disk output. Per-token temporary text allocations
remain, and a fallback run may delay visible output until finish.

## Checks and actual Mac observation

Formatting, default and Metal strict checks, all 310 CPU/332 Metal-feature tests,
both iOS targets and 12 release diagnostic methods pass. The five new methods
exercise the actual scheduling helper: an independent ASCII/ASCII/invalid-byte
sink across the split; cancellation after two deliveries; callback failure in
the second request; terminal sealed-sink failure; and insufficient combined
reserve rejection before asset access. The sink test checks scheduling, while
the separately accepted decoder fixtures check actual fallback decoding.

The actual MacBookPro18,3 / Apple M1 Pro, 16 GiB run uses the same immutable
`mlx-community/gemma-4-e2b-it-4bit` checkpoint at
`238767527555cb75a05732a84dff5d6ba0dd6809` and frozen ten-token prompt.
Both full schedules match all eleven sampled IDs and ten visible IDs.
Cancellation matches the first two samples without another request. Every
cumulative visible chunk matches the authenticated independent scalar vocabulary
decode; each schedule has exactly one terminal chunk. Stream ownership remains
65,816 bytes before/after every schedule, inside its separate 4 MiB allowance.

All 24 full logit rows / 25,165,824 bytes are retained. All 48 unchanged FP64 and
staged-FP32 gates pass over 12,582,912 values; maximum gate fractions are
0.46041798867955425 and 0.0013559174399768907. Corresponding rows are bitwise
identical across both full schedules and the cancelled prefix. The controller's
actual verifier and classification reject 12 corrupted scheduling/text/state/
array packets. A finite numerical mismatch retains successful process capture
while rejecting numerical acceptance. Synthetic packets and owner sizes are
verifier evidence only. The repeated owned regression binds source, logs and
cleanup for a successful run; it does not establish invocation-failure retention.
The initial identical regression result has no standalone invocation/log credit.

Combined logical admission is 3,859,976,044 bytes. Observed resident/physical
process peaks are 2,054,062,080 / 2,776,222,504 bytes, below 6 GiB. The 236.63-second
combined loading/generation/diagnostic observation is not a controlled latency
benchmark. See the [complete receipt](../benchmarks/results/2026-10-25-gemma4-stream-generation-v1.json).

## Remaining gates

This checks one short scalar formatted reply with integrated text delivery.
Broader quality/full G3, earlier precision/source/GPU intermediate and long-rotary
gaps remain unwaived. Resident Metal, representative performance, qualified
load-time model/codec identity, native ABI/app integration and independent
mobile admission/execution remain open. Official E2B's 131072 total positions
cannot supply 131072 output tokens plus a nonempty prompt. No capability,
catalog, profile or numerical limit changes.
