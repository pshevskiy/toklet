# App worker runtime memory preflight

MLX loads now request the engine's actual Metal resource plan before allocating
model/session numerical buffers. The Swift worker uses native factory policy2;
it does not maintain an architecture-to-attention table. The exact numerical
budget is the requested whole native budget minus the shared20,480-byte owner
allowance. Actual factory options still receive the whole budget and perform
their existing checked debit once.

`NativeResourcePlan` copies and decodes the bounded immutable C report, releasing
its owner on the creating OS thread before returning a Swift value. It checks
schema, request geometry/budgets/modes, component sums with overflow detection,
and each fit flag against its own total. A consistent non-fitting report is
valid information. The worker retains it for the failed attempt and rejects the
load before creating numerical owners. Loading still reopens and validates the
artifact and device: an earlier fit report grants no payload validity.

The snapshot exposes the last attempt's runtime plan separately from metadata.
A new load, suspend, unload or close clears both; a queued cleanup cannot publish
a superseded load's report. A cold gate checks cleanup after inspection and
before the actual load. It does not make the existing C model loader cancellable
once that call has begun. Reports are not qualifications or free-memory checks.

Both metadata and runtime summaries have bounded copying/decoding ownership. The
conservative logical application allowance is1,311,232 bytes for both summaries,
separate from numerical admission and tokenizer/text/application state. This is
neither a reservation nor a measured process peak. Driver/pipeline setup still
allocates; inspection creates no numerical weight, KV, scratch or logits buffers.

## Verification

The [receipt](../benchmarks/results/2026-10-24-swift-runtime-preflight-v1.json)
binds sources, the unchanged three-slice native policy package, owned commands,
raw results and historical inputs. Swift6 strict default/Metal checks cover25
wire mutations, arithmetic boundaries, a session-only budget rejection, exact
plan-to-factory admission, stale-plan invalidation, three queued cleanup cases
and an artifact changed after successful inspection. Existing21-mutation and
legacy CPU/Half/MLX checks pass. All89 native-helper tests pass; ten helper source
lists include the new adapter, including the score helper's indexed subsets.

Real SmolLM2, Qwen2.5 and Mistral Chat results match the complete preceding JSON,
including multi-turn, context rejection, cancellation, suspend/replay and release.
The existing Qwen256-output test also matches its entire prior result and checks
runtime-plan admission against the actual factory. The comparison collector
separately authenticates the accepted long-output receipt and frozen65-ID input;
the profile runner originally guarded the selected prior output hash. Both iOS
SDK modules, full unsigned Release iOS/simulator apps and the Mac app build pass.
Full app builds cover their captured UI source versions, not later parallel UI
edits. The first strict build found a test-only mixed-integer array inference error;
the explicit UInt64 array correction passes, and the failure remains recorded. Its initial test source is an exact-SHA-verified
reconstruction from the corrected source, not an originally saved source copy.

Rust, shaders and the C header are unchanged from the accepted native-policy
snapshot, so its185 CPU/207 Metal-feature checks and explicit GPU evidence are
inherited only for those exact bytes. A fresh default archive was also built for
the unavailable-Metal API regression. Original release binaries are unchanged.

Retain this preflight. No output/context limit, numerical kernel, download
catalog, text profile, task quality, performance or physical-iPhone qualification
changes. The installed phone app retains its separately recorded package.
H1 still needs qualification-receipt evaluation; general Hugging Face acquisition
and downloaded chat remain separate active-plan tasks.
