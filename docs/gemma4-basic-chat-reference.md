# Gemma4 formatted basic-hello mathematical reference

The fixed `basic-hello` reference composes the authenticated
[pinned text witness](gemma4-text-codec-reference.md) with the
[complete mathematical graph](gemma4-text-math.md), before native formatted
model generation. The original four-step raw reference remains the default.
No Toklet numerical implementation or tokenizer runs in this reference.

## Fixed workload and state

The frozen user message `Hi` renders the exact ten-token prompt
`[2,105,2364,107,10979,106,107,105,4368,107]`. The text receipt and both fixture
files are authenticated before use and revalidated at the endpoint. Selected
artifact stops are `[1,106,50]`. Sample at most16 IDs, including a hidden final
stop. Maximum evaluated prefix length is25; this does not extend the native
32-position execution limit.

Staged FP32 chooses the first index of the maximum capped logit. FP64 evaluates
the same selected prefix as a separate diagnostic, without choosing history.
Each prefix traverses all713 finite states in each precision. Retain complete
last-row final normalization `[1536]`, head `[262144]` and capped logits
`[262144]` in both precisions. Each completed step occupies6,309,888 bytes;
the maximum100,958,208 bytes fits a separate128MiB capture bound.

Only publish a sampled ID after both precision traversals and all selected-row
comparisons complete. A sampled stop remains pending, absent from visible IDs;
an output-limit token remains pending and visible. Raw failure data and
incomplete descriptors are retained. Capture integrity, numerical gates,
selection disagreement and stop-membership disagreement are separate results.
No failed diagnostic changes the staged-FP32 rollout or numerical tolerances.

## Observed result

Both contracts select
`[10979,236888,2088,740,564,1601,611,3124,236881,103453,106]`.
The first ten visible IDs decode to `Hi! How can I help you today? 😊` in both
keep/skip-special modes. The final stop106 is hidden and pending; all20 earlier
IDs, including the ten-token prompt, are committed. Maximum evaluated prefix
length is20. All selected maxima are unique, and both selection and stop-set
membership disagreement counts are zero.

All33 unchanged selected-row gates pass over5,784,064 paired values with
`atol=1e-5, rtol=1e-4`; the maximum error-to-bound fraction is
0.7812509635026432. The66 retained arrays occupy69,408,768 bytes. Eight distributed
probes per vector form the3,168-byte
[portable fixture](../tests/fixtures/gemma4-basic-chat-math-v1/math.json).
The collector decodes selected visible IDs through the previously authenticated
independent scalar vocabulary oracle; it imports no tokenizer/template runtime.
Full arrays remain in ignored artifacts, authenticated by the
[receipt](../benchmarks/results/2026-10-25-gemma4-basic-chat-math-v1.json).

Measured reference resident/physical peaks are343,932,928/152,011,976 bytes,
below512MiB. The366.02-second combined load/22-graph/output observation is not a
latency benchmark. Neither this single prompt nor these selected rows close the
earlier full-graph precision or quality gates.

## Verification

Six offline greedy methods, eight graph methods and eight affine methods pass.
They include closed-profile bounds, first/final-slot stop, no-stop output limit
and a failure between precision traversals that retains three FP64 arrays
without publishing a sample. The controller independently replays every saved
value, maximum/margin/tie count, gate, prefix, stop, visibility and pending state.
Its actual terminal-check code accepts four valid synthetic cases and rejects
nine corrupt cases. An earlier controller draft had a sample-count variable
shadowed by a per-vector value count; it was corrected before model execution.
The draft and its narrower regression are retained as historical diagnostics.

The independent reference process has a512MiB allowance. The largest possible
FP64 head-plus-logits pair is104,857,600 bytes at prefix25; reservations remain
128MiB each for runtime/BLAS, tiled projection and head/output plus64MiB each
for capture overlap and graph/metadata. This allowance is separate from native
engine, tokenizer and application ownership.

## Scope

Native formatted generation, append-only text streaming, combined memory
admission, quality, original-BF16/source precision, resident Metal, representative
performance, phone fit, app integration and131072 output tokens remain open.
Earlier full-graph, GPU intermediate and long-rotary failures remain unwaived.
No model catalog, ABI, context/output limit or G2/G3/G4 completion changes.
