# Native numerical capacity report

`toklet_mlx_capacity_plan_create` exposes the existing capacity solver through
the native C interface. It uses the same 56-byte `TokletRuntimePlanningOptions`
and creating-thread JSON handle/copy/release contract as
`toklet_mlx_plan_create`. The original entrypoint still returns its unchanged
flat resource report. The additive entrypoint returns `{resources, capacity}`.

The range covers `[requested chunk, requested capacity]`, keeping modes,
position offset and load context fixed. `selected` is the largest point that
fits both the numerical session budget and device buffer limit. If no point
fits, it is null. `first_excluded` is the adjacent rejected point, the minimum
when none fits, or null when the maximum fits. A zero session budget can produce
a valid no-fit report; a zero load budget remains invalid.

Source and resident loading fit remain separate from session-capacity fit.
A selected point cannot override a false loading flag. `resources` describes
the requested upper capacity, while the selected point has its own footprint;
callers must not substitute one for the other. Construction revalidates current
inputs, budgets and device state. A report reserves no memory.

Attention policy 2 follows the ordinary native factory; explicit policies retain
their existing restrictions. The new public Rust wrapper is
`metal::planning::inspect_mlx_capacity_with_native_attention`. Both C entrypoints
share cold validation, error mapping and handle capture through a small enum
branch. The existing pure solver and resolved concrete memory plan are reused;
no numerical kernel or model admission changes.

Planning initializes Metal/MPS and pipelines, while allocating no numerical
weights, KV, scratch or logits. Driver/framework bytes remain an allowance.
This API searches an already admitted caller upper bound. It does not discover
the model's declared maximum or the ordinary factory's enforced maximum;
the [ordinary metadata-only limits report](native-context-limits.md) now supplies
the enforced per-model bound. Typed Swift/app integration remains B5c. Neither
report establishes context, text, whole-app memory, phone or performance
qualification.

Formatting, strict default/Metal Clippy, default 457 tests (20 ignored), Metal
564 tests (184 ignored), release test compilation, static library build, exact
29-header-export matching, C compilation and strict Swift 6 compilation pass.
The first Swift build invoked resolved `swift-frontend` instead of the `swiftc`
alias. That failed job remains preserved. The corrected owner guards both the
invocation alias and resolved binary and revalidates the 13 successful jobs;
one Swift retry completes the explicit 13+1 aggregate. Retained-product copying
uses a checked known-size bound and exclusive destinations.

Five explicit Mac planning/ABI jobs pass on Apple M1 Pro. Four factory fixtures
check range 3–31 at offset 1, native versus explicit policy, exact/one-byte-under
session budgets, no-fit zero budget and a fitting session with failing loading
budgets. Existing factory regressions preserve frozen parent memory, and
20 direct-report checks cover immutable snapshots and wrong-thread rejection.
The C and Swift callers check the additive symbol, bounded handle reads and
range/schema behavior; Swift also checks equality of the old report and nested
`resources`. These are compiled assertion results. Full capacity JSON was not
retained for an additional independent decoder; the saved C summary and Swift
output alone do not prove every nested range field.

Astra independently authenticates current code, all actual products, runtime
and tool identities, aliases, successful-job provenance and closed process
groups. [Exact evidence](../benchmarks/results/2026-10-30-native-capacity-ffi-v1.json)
records the checked scope. Sparse process samples are not complete peaks; no
real-model forward, new factory eligibility, phone or enlarged-context
qualification is claimed. App controls and their safe application remain open.
