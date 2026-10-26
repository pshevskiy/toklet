# Original Gemma E2B hybrid execution on Mac

The original resident packed weights now execute through the shared Gemma
scalar/hybrid graph and the public real norm-copy/adapter path. This record
covers short diagnostics for the pinned `mlx-community/gemma-4-e2b-it-4bit`
revision `238767527555cb75a05732a84dff5d6ba0dd6809`, not app eligibility, a
resident GPU graph, all32 implemented positions or the declared131072 window.

See [tiny composition](gemma-hybrid-graph.md), [complete immutable storage](gemma-resident-weights.md),
the [public evidence](../benchmarks/results/2026-10-26-gemma-hybrid-real-v1.json)
and [delivery plan](gemma4-implementation-plan.md). Original precision and
quality gates remain open; passing staged-FP32 results do not waive them.

## Mechanism and ownership

No numerical operator or production graph arithmetic changes in this task.
Original packed4/group64/BF16 products and gathers use the existing GPU
workspaces. CPU norms, rotary, attention, PLE/residual/scalar/softcap operations
and complete KV/history/logit transaction order remain shared with the scalar
reference. Immutable model ownership and session scratch stay separate.

All278 packed matrices and262 GPU learned vectors remain resident. The small
CPU norm copy contains1143948 payload bytes; actual copy plus descriptors is
1153484 bytes. Exact bit checks and independent original FP32 hashes verify all
262 copies. Construction Context is dropped; a fresh same-device Context runs
the graph. Hashing streams4KiB blocks rather than retaining another packed model.
All1186 original guarded buffers have distinct identities and match the frozen
storage inventory before copying, after copying and after repeated sessions.

The shared generation example adds explicit `--backend hybrid` on Apple Silicon
Mac with the Metal feature. Scalar remains default. A small cold module handles
real hybrid loading and admission; both paths share the same generic
`InferenceSession` observer, generation, codec and stream schedules. There is no
new architecture interpretation, Hugging Face eligibility or app profile.

The hybrid raw path reserves16MiB diagnostics before loading/session admission;
formatted and streamed paths also reserve the existing1GiB codec and4MiB stream
policies as applicable. Raw graph capture instead reserves128MiB for diagnostic
snapshots, hashes and capture state. These are policies, not measured residency.
Exact/minus-one raw and generation admission are separate witnesses; unexpected
operator/device/allocation errors cannot count as the expected budget rejection.

## Complete short graph evidence

The same four raw IDs `[2,100,257,100]` run schedules `[4]`, `[1,1,1,1]` and
`[1,3]`. Existing test-only layer fault points capture nine selected states at
all35 layers without publication. Each fault retains complete committed prefix
snapshots, storage pointers and allocated bytes. Successful forward then captures
full262144-wide logits, final norm, PLE, all15 producer staged K/V and each valid
committed cache position. Reset/replay is checked per schedule. The layer faults
are diagnostic boundaries, not injected in-flight GPU failures.

The capture has3016 full arrays /66600960 bytes. Independent replay performs
33300480 comparisons against the previously frozen mathematical and staged-FP32
full arrays. Staged FP32 has zero failures and maximum gate fraction
0.0013644315853741284, using unchanged `1e-5 + 1e-4 * abs(reference)` thresholds.
The mathematical contract still rejects27 near-zero logits per forward,
repeated after reset:54 failures per schedule, maximum gate fraction
1.2599434792233821. Their original bytes, coordinates, counts and gates remain
retained; this is not a universal numerical acceptance.

| Raw graph accounting / observation | Bytes |
| --- | ---: |
| Complete model, CPU vector copy and graph metadata | 2605832624 |
| Model plus raw session admission | 2862709336 |
| Separate diagnostic policy | 134217728 |
| Native process peak RSS | 2767110144 |
| Native process peak physical footprint | 2937375360 |
| Mac process policy | 6442450944 |

Physical observations include the whole native process, with completed GPU work.
They concern this short graph diagnostic and do not establish throughput,
maximum context or independent phone fit. Native assertions establish failure
transaction/pointer/accounting behavior; transcript relationships are not a
separate full KV-snapshot or allocation-layout oracle.

## Generation and remaining acceptance

Raw generation checks uninterrupted, split, cancel/resume and explicit caller
stop; formatted generation checks uninterrupted and split; streaming adds
cancellation with one terminal flush. All62 captured full-vocabulary rows pass
both original independent contracts:32505856 value comparisons, maximum
mathematical gate fraction0.5103206474158878 and staged-FP32 fraction
0.0013656471300556837. Same-prefix schedules are bitwise identical. These
sampled-prefix rows passing do not resolve the27 earlier full-prefill-logit
failures documented above.

Separately, exact normalized scenario/report/codec states match pinned previously
qualified scalar captures. This is state refinement, distinct from independent
numerical replay. Raw selected IDs are `[107,100,45518,107]`. The hello schedule
samples11 tokens including hidden stop106, emits10, and returns
`Hi! How can I help you today? 😊`, with20 committed and21 history tokens.
Stream chunks, split continuation, cancellation, hidden stop, pending state,
rendered input/template IDs and terminal flush all match their qualified states.

Generation ownership is2862709528 bytes for every profile. Specific expected
`MemoryBudget` errors reject one byte less; exact budget succeeds with identical
actual categories. Combined ownership/policies are2879486744 raw,3953228568
formatted and3957422872 streamed bytes. Policies are not physical-footprint
predictors: the raw process peak exceeds its requested ownership/policy sum,
while every native process is independently checked against the full6GiB policy.

| Functional profile | Full distributions | Peak physical footprint, bytes | First delivered token / complete first request, seconds |
| --- | ---: | ---: | --- |
| Raw, four sampled tokens | 16 across four schedules | 2899692064 | 0.6325935 / 1.1377035 |
| Formatted hello | 22 across two schedules | 2970880856 | 0.9384325 / 2.471050458 |
| Streamed hello | 24 across three schedules | 2981284696 | 1.007017375 / 2.54371425 |

Times are completed-work observations from a single functional trial on
MacBookPro18,3 / Apple M1 Pro,16GiB / macOS26.6.2(25G83), release Rust1.90
with Metal. Generation timing includes full-vocabulary diagnostic hashing/writes and
excludes source validation/loading, construction,
tokenizer loading and prompt rendering/encoding. There are no warmups/repetitions,
paired baseline or sustained/power qualification; these are not benchmark or
performance-release claims. Raw samples/reports remain in the evidence.

Eight real-graph and eight generation reader controls pass, including coherent
corruption, partial capture and unchanged-argmax numerical rejection. Strict
default/Metal lints, formatting,356 CPU/408 Metal tests (16/130 ordinarily ignored),
12 generation-example tests in each configuration and both iOS compile targets
pass. All31 owned process groups close. Protected release executables remain
unchanged. The pre-run controller-shadowing bug was caught by Astra and repaired
before execution; the two development compilation checks and all original
source/output captures remain retained. No actual numerical run was discarded.

The next work remains the G2.1/G5 precision/quality decision and G4.3 resident
scheduling plus progressive context. Qualify the32-position boundary before
claiming that whole implemented capacity, then useful512-position and larger
model-specific boundaries. Source BF16, earlier GPU/rotary discrepancies,
held-out quality and representative timing remain open. Mac app/native
integration and a separately admitted phone representation/lifecycle remain open.

The [model context requirement](long-reply-engine-plan.md) targets both Chat and
Expert with the selected model's full supported input/output window. Rendered
prompt, template, retained history and continuation share the model total;
separate model input/output caps also apply. These diagnostics grant no new
window or app capability. Smaller device/qualified bounds must be explicit
partial support, not silent substitutes for the full requirement.
