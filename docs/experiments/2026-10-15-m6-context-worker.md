# M6 paired context worker

15 October2026. Component qualification only; no long real-model behavior,
performance or KV precision decision is established here.

The new `metal_context` example owns one immutable selected model and separate
Reference/Online sessions. It subtracts shared weights once and reserves all
Reference-owned memory before the Online constructor may allocate. Separate
profiling workers admit both counter allowances before requests. Session memory
is reported throughout; the6GiB ceiling remains.

Requests support the original reset/append/cached/pending controls, fixed-output
completed engine samples, paired full-vocabulary NLL/error comparisons, and
separate decode profiles. Numerical execution is the already qualified engine.
RPC parsing/validation precedes ordinary engine timing; cache mutation, completed
prefill/vocabulary readback and each completed decode step are timed. This is
fixed-ID engine timing, without tokenization, sampling or visible text latency.
Comparison rows and individual profile operators use separate bounded events.
The1MiB request and512-request bounds are explicit.

Profiles prepare an exact prefix, then commit one final decode input at the
named depth. Subsequent profiles recover the same valid prefix and recompute
that final input, recording actual reused/prefill counts. Profile completion has
no pending token; ordinary fixed-output generation retains its last pending ID.
The future profiling decision uses the union of attention GPU intervals divided
by matching completed decode-command GPU duration, then a median across three
profiles. Host waits/calibration brackets are not the denominator.

The [frozen context contract](../../benchmarks/m6-context-v1/manifest.json)
predeclares4K/8K/16K/32K total counts, N−128 prompts plus128 outputs, byte hashes,
focused512-target long numerical gates, sampling scopes and conditional FP16 KV
retention. Inputs cycle the original held-out stream; they do not establish
natural long-document ability. Full10000-target and100-task gates remain separate.
Reduced long/secondary sample counts are descriptive/exploratory; unexplained
regressions require investigation. A speedup acceptance needs the declared
initial-short repetition and confirmation policy.

The [worker receipt](../../benchmarks/results/2026-10-15-m6-context-worker-qualification-v1.json)
binds source snapshots, immutable executable, contract/data, checks, raw native
stream and standalone functional replay. Five portable Rust protocol tests,
97 CPU tests,110 ordinary Metal-feature tests, formatting and strict CPU/Metal
all-target Clippy pass.

Two tiny native workers complete23 requests/262 events, including228 profile
operator events. Exact cache/pending/append state, output-reservation rejection,
unchanged admission, paired vocabulary/loss metrics, calibrated decode profiles
and repeated same-depth prefix recovery pass. All workers exit0 with clean
cleanup and unchanged pins. Maximum unique admission is335,608,084 bytes and
joint native lifetime charge27,378,888 bytes. These are tiny structural checks;
the artifact does not qualify Qwen long-context memory or latency.

Astra accepted the code, frozen contract and bound evidence without findings.
Retain this worker for the separately reviewed collectors and real-model runs.
Production defaults and M7 remain unchanged.
