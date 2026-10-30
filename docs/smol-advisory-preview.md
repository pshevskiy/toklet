# Bound Smol window estimates and Apply

Friendly Chat can preview and apply context, chunk and whole-session memory
settings for its authenticated managed Smol model. The estimate remains based
on reopened model metadata. It does not authenticate those files as the loaded
weights, allocate a replacement or grant final memory admission.

The request keeps the audited Smol metadata profile and explicitly opts into a
resident advisory mode. Model ID, session ID and loaded upper are checked under
the worker condition at enqueue, inspection checkpoints and publication. The
effective search upper cannot exceed the opened model's limit. Requested,
effective and selected values are displayed separately; a request of 1,024 with
an opened limit of 512 can select at most 512. This enables no larger native
context range. The default source-profile API remains metadata-only for Smol;
existing ordinary preview semantics are unchanged.

Apply uses the existing shared transaction. Current session/configuration,
selected budget/buffer fit, reply limit, exact displayed request and one bridge
addition must match. Actual native D1 computes the loaded model's coexistence and
constructor requirements; fresh process headroom and actual D2 allocation must
also pass before acceptance. Loading-fit flags from reopened files are separate
from replacing a session on already resident weights. Accepted settings use the
existing healthy terminal preference save. Conversation and draft stay intact;
cached tokens reset, and the next Send admits the rendered conversation again.

The [verification record](../benchmarks/results/2026-10-30-smol-advisory-preview-v1.json)
passes 33 new synthetic controls, existing planning/Apply/worker/store
regressions and all three app builds. The maintained Smol preference controller
is updated and compile-checked for the explicit advisory mode. Unchanged wire,
native and successful tiny Metal evidence is reused through exact closures.

One cached Smol controller completes four previews, two deliberately stale. It
observes requested1,024/effective512 without numerical preview allocation. An
explicitly synthetic 1 MiB underestimate is rejected by actual D1 while the old
session and user state remain intact; that report is not attributed to the native
estimator. The actual 256/chunk16 Apply then records:

| Check | Actual result |
| --- | ---: |
| Model / old session / new session | 1 / 2 / 3 |
| Aggregate requirement / coexistence budget | 660,314,446 / 805,306,368 bytes |
| Planned constructor / actual candidate admission | 220,246,514 / 220,246,514 bytes |
| Incremental requirement / observed configured headroom | 94,199,068 / 6,230,342,936 bytes |
| Preference writes before / after UI reconciliation | 0 / 1 |

Headroom is the configured Mac process policy, not free RAM or a reservation.
The coherent terminal worker snapshot is captured before UI reconciliation;
submitted request, old plan, new candidate and durable configuration correlate.
After reconciliation clears the transient result, durable settings, committed UI
state and the single save are verified separately. Prefix, workspace, draft,
reply and Qwen record remain intact, and owning close succeeds. No text is generated.

The first compile failed on test-only JSON typing; a latent maintained-test member
typo was corrected before it was reached. The first Mac controller failed because
it checked transient evidence after UI reconciliation. Both failures remain
preserved. Test-only capture fixes leave production and numerical criteria
unchanged; the continuation recompiles and reruns only the changed controller.
The failed run's write alone is not promoted to full Apply success.

Independent Astra saved audit verifies eight raw frames plus summary, admission
arithmetic, exact inherited checks/products and all31current/historical process
closures. Final publication review precedes commit. Rendered
Mac UI is unavailable while locked; iPhone runtime, successful real generation
after resizing, broader context/quality and performance remain separate work.
The selected 36-export native package and Rust source are unchanged.
