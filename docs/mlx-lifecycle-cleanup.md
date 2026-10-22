# Lifecycle trace cleanup after coordinator failure

The authenticated-job trial retained a real failure path: an injected READY sink
error closed native owners but left the trace's active command set, so detach
reported an additional cleanup error. The original
[raw result](../benchmarks/results/2026-10-22-native-mlx-apple-jobs-v1.json)
remains unchanged and is not counted as clean lifecycle completion.

The coordinator now completes a remaining trace label only after its existing
full-close predicate verifies worker completion, empty owner/pending state and
closed acknowledgement, then detaches scene signals. A healthy trace can end
its exact remaining label and detach; an already failed trace retains its original
failure and still rejects detachment. No error, failed observation or generation
result is rewritten. The recorded closed event keeps its historical active label;
the final empty label means cleanup completion, not successful generation.

The strengthened MLX negative test actually reaches READY and retains
`Forced READY sink failure`, with `passed=false`, closed/released owners, empty
cleanup/endpoint errors, no final trace error or active command, and verified
post-close detachment. The legacy Half Metal negative test reaches the same
injected failure. The CPU tiny test instead records `No demonstrably active
generation window`; that original error also remains while cleanup is complete.
These observations are deliberately distinct from actual OS lifecycle behavior.

Both legacy lifecycle-app unit runs pass 34 case labels each. Low-level CPU/Half
trace checks pass 13 labels each, including overlapping/wrong commands, premature
detach, bounds/clock/state failures, exhausted-trace background release and warning
unload after a trace failure. The full authenticated MLX tiny checker also passes
its existing 40 schema rejections, measurement/quality/task/state components and
now-strengthened failure closure. Strict iOS source checks and fresh linked
simulator/unsigned device app builds pass. No Rust, shader, numerical-gate or
historical CPU binary changes are made.

The [task receipt](../benchmarks/results/2026-10-22-native-mlx-lifecycle-cleanup-v1.json)
retains all 12 owned outer commands and nine linked-app child commands, current
source/archive/app identities and the preceding negative result's identity.
This closes the observed trace-cleanup gap. Real-model numerical references,
quality/memory/performance and physical iPhone MLX trials remain open in the
[M8 plan](mlx-implementation-plan.md).
