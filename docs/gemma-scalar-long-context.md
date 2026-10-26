# Native scalar Gemma witness context through1025

This task adds an explicit synthetic scalar diagnostic constructor. The default
witness, real E2B constructor and hybrid/GPU paths retain their32-position
limits. It does not grant a new app profile or claim the selected real model's
full shared input/output window.

`Model::new_witness_with_context(weights,C)` constructs the exact private tiny
WITNESS geometry with1<=C<=1025. Immutable borrowed packed weights stay outside
session-owned state. Separate FP32 K/V use `[kv_heads,capacity,head_dim]`;
local physical capacity is min(W4,C), global capacity is C. Token histories and
resolved routes belong to each session. Existing staged producer proposals,
all-frame validation and final publication order provide failure rollback.

The per-call row bound is independent of C: at most32 rows, with query score and
probability storage proportional to heads*chunk*C. Default planning and
construction use min(C,32), and raw/generation admission reject larger chunks
before allocation or backend work. Both histories are counted in generation
admission. Executable diagnostic context, per-call rows and the real model's
declared window remain distinct.

The accepted [full-sequence oracle](gemma-long-witness.md) supplies frozen
mathematical-FP64 and staged-FP32 values for complete1025-position logits,
producer K/V and17 selected positions of other states. Native tests compare
logical KV positions rather than ring slot order. The tiny numerical graph
retains W4; real E2B W512 numerical qualification is separate.

A separate exact-integer structural witness uses two KV heads, local W512,
full-prefix global state,15 producers and35 consumer routes. Five capacities
511/512/513/1024/1025 use uneven chunks landing on boundary positions. An added
sixth schedule straddles both wraps in multirow calls510→517 and1018→1025.
First five schedules were frozen before production edits; Astra identified the
missing straddle after admission/test drafts. The extension was frozen twice
before native execution; original source/fixture pins remain preserved. No
numerical cache kernel was changed while preparing the extension.

The structural fixture records logical K/V hashes, absolute visible intervals,
segmented head strides/content and physical committed-slot hashes. Its integer
recipe encodes producer, head, position and feature exactly in FP32. Native
records are replayed by a separate standard-library reader with fixture hashes
pinned before interpretation. Changed values, metadata, incomplete coverage and
history semantics must reject through explicit reader controls.

Planned checks cover exact-budget admission and budget-minus-one rejection,
allocation-free numerical forwards/reset/failure, stable buffer addresses,
selected layer and final-head failure/retry, capacity+1 rejection, complete
logits/producer KV and selected late intermediates. Generation separates frozen
prompt-logit evidence from sampled-history replay consistency; the sampled final
position remains pending until evaluation succeeds. Numerical or resource
failures will be recorded without relaxing gates.

## Actual checks, 27 October 2026

All eight native schedules pass: C511/512/513/1024 with uneven chunks and
C1025 with chunk1/3/32 plus an uneven schedule. Independent replay checks
8,901,008 values across both frozen contracts and13,712 raw argmax rows,
including repeated state observations and both contracts. Maximum numerical
fraction is0.15695623024778313 of the unchanged gate. Complete logits and
producer K/V cover every input position; other intermediate values cover the
17 frozen selected positions. This is distinct from complete real graph parity.

The structural suite checks9,870 layer views and4,230 producer commits across
six cases, including both straddling calls. Exact range/head-stride/content,
committed slots and shared consumer routes agree. Generation checks68 frozen
prompt/restored-logit values and four argmax rows. One sampled final token is
pending at history length1025/committed1024; rejected append and injected final
failure preserve public history, inner KV/logits and pending state. Retry and
bitwise replay of that sampled history pass; restoring the fixed prompt returns
to independent frozen expectations. No generated-text quality is inferred.

Exact-budget and budget-minus-one, numerical allocation counts and pointer
stability pass. At C1025, admitted raw session bytes are1,086,540/1,145,724/
2,003,892 for chunks1/3/32. Actual K/V payload is1,051,648 bytes; history is4,100
bytes, and staging is1,536/4,608/49,152 bytes. Both generation histories are
counted separately. Native physical/RSS peaks are14,270,872/28,753,920 bytes;
these are tiny witness measurements, with no real-model fit claim.

Default CPU tests total361 passed/19 ignored. Formatting and strict all-target
Clippy pass with default and Metal features; the scoped Metal regression suite
passes19 with five explicit hardware/long tests ignored. The three long scalar/
state tests are executed explicitly in release mode, not counted as ordinary
hardware-independent test passes. No GPU kernel was changed or newly qualified.

Seventeen coherent reader controls reject changed numbers, negative-index route
aliases, exact-case chunk metadata, matching but false allocation totals,
segmented layout/visible intervals, committed-slot hashes, incomplete coverage
and pending/history semantics. The first run stopped at a test-only Clippy
warning; the corrected native run then passed but reader-control RSS577,503,232
exceeded512MiB. Both failures remain archived. Removing redundant full-transcript
copies lowers the passing reader-control peak to458,293,248 RSS/332,563,224
physical bytes. Line traversal is lazy; parsed records are still materialized.
No memory or numerical threshold is relaxed.

The corrected reader replay reuses the authenticated passing Rust builds and
native capture; all Rust, fixture, toolchain and binary identities match. Only
affected readers are rerun, then previously pending KV/generation and Metal
scope checks finish. All27 owned groups from the three attempts are absent.
Astra accepts the scoped mechanism and complete evidence. See the
[public receipt](../benchmarks/results/2026-10-27-gemma-scalar-long-context-v1.json).

Next is L3 engine-owned capability/shared-window admission for the existing
Qwen3 bridge and both app modes, retaining qualified limits. Real Gemma whole32,
bounded long BF16 reference, unchanged full1025 precision/quality/fit and
progressive real-context/device qualification remain required. This task does
not grant new app eligibility, phone support or the full selected-model window.
