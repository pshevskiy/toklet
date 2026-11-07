# O2.B2.A: bounded Gemma UI publication metrics

The managed official Gemma 4 E2B Mac route now correlates Send, worker completion
and controller publication in the existing opt-in request-summary ring. This is
the publication portion of O2.B2. Visible-frame measurement remains separate.

## Boundaries and ownership

`FriendlyChatStore.sendOrStop` requests an optional monotonic stamp before chat
preparation. The worker admits it only for enabled Gemma BF16 summaries with the
existing 1 MiB telemetry headroom. Ordinary disabled workers return without a
clock read or condition acquisition. A weak worker reference and the request
UUID plus Send stamp associate the eventual controller receipt with its record.

The optional schema2 `ui` object records Send uptime, preparation to worker
submission, Send-to-first-callback TTFT, terminal handling, worker-to-handling
delay and publication time. Root worker timings retain their previous meaning.
Publication is stamped after workspace mutation and before existing synchronous
history persistence. The current UI publishes response text at completion;
callback TTFT therefore precedes text publication.

`firstNonemptyPublicationMilliseconds` and `firstAnswerPublicationMilliseconds`
are available only for a selected conversation with nonempty published text.
The latter is scoped to this pilot's audited non-thinking template. Neither
field measures rendering, window visibility or completed-prose quality. Empty,
background, rejected and missing-message cases retain explicit outcomes/nulls.
The UI disposition is separate from the authoritative native request outcome.

The same fixed 32 slots, <16 KiB record, 4 KiB serialized row and 1 MiB admission
allowance apply. No text, token arrays or view references enter summaries.
Stale, ambiguous, duplicate or invalid-clock receipts increment the saturating
`uiUpdateErrors` counter without changing another record or inference state.
Late valid receipts can finish an existing record after mode changes or close.
An evicted record cannot be reconstructed from its receipt.

## Verification scope

The strict Swift 6 no-model fixture checks formulas, clock ordering, root timing
semantics, empty/background/rejected outcomes, identity collisions, stale and
duplicate receipts, pressure and disabled collection. A separate real Gemma
fixture drives the actual `FriendlyChatStore` and 100 ms controller poll with
isolated history. Controls cover success, input rejection, output-decode failure,
Stop, queued cancellation and queued close. Successful requests must preserve
the frozen token IDs, full text hash and history24/committed23/pending1 state.

Two fresh-process paired cohorts each use three warmups and ten measured trials
per mode. Caller timing includes submission, polling and existing persistence;
UI fields expose their narrower publication boundary. This fixture has no
window. Process lifetime peaks and thermal observations cover this component,
not whole-app rendering. The workload uses cap512/chunk32/reply32; the existing
headless reply12 overhead track is checked separately.

On the M1 Pro, the two publication cohorts pass all 52 requests (40 measured).
Summary-mode median Send-to-publication is 2399.372 / 2363.645 ms;
worker-completion-to-handling is 78.713 / 63.691 ms. Preparation is
0.090 / 0.082 ms. Component process lifetime peaks are at most 4,919,365,880
bytes, below 6 GiB; every recorded thermal state is nominal. These values
describe the fixed public prompt and completion-only controller publication.

The paired 95% bootstrap upper bounds exceed 1% for TTFT in both cohorts and
for total time in the first. The component overhead bound is **unqualified**.
Negative point estimates are noise-compatible observations, not speedups.
The separate full-caller headless conformance and both paired overhead cohorts
also pass; their 1% bound remains unqualified. No numerical source or native
archive changed. Verified pins support reuse of the required Rust/native checks;
changed production Swift, standalone fixtures and app builds are checked afresh.
Six new UI report tests and 29 portable checks pass. Astra approved the combined
review after requiring typed summary/raw state equality and frozen native work
counts at the request-ID join. The strict offline analysis reproduces both
cohorts; the exact collecting runner and unchanged execution pins are retained.

Evidence, required checks and Astra disposition are linked from the
[delivery receipt](../benchmarks/results/2026-11-07-o2-ui-publication-v1.json).
Failed prototype assumptions and superseded controls remain in the
[iteration record](../artifacts/o2-ui-publication-2026-11-07/iteration-notes.json).
Summary collection remains default-disabled; this delivery does not qualify
visible-frame latency, tail percentiles, physical iPhone behavior or a speedup.
