# M7 native history snapshots

`toklet_session_history` copies the exact sampled token history into caller-owned
storage after a request. This includes tokens still pending evaluation, hidden
stop tokens and samples retained after callback failure. A null output with zero
capacity queries the required count. Insufficient storage returns Invalid,
writes the required count and leaves the token output untouched. The existing
owner-thread, busy-callback and poisoned-session guards apply. There is no
allocation, borrowed storage exposure, numerical operation or ABI-record change.

The Swift harness needs exact history after cancellation. Counts and emitted
IDs alone cannot determine whether a precancelled replacement prompt changed
session state. The caller-owned snapshot removes that ambiguity while keeping
frontend/replay storage outside numerical kernels and explicit in application
memory accounting. History copying is a coarse request-boundary operation;
it does not copy KV or introduce work per decoded scalar/head/tile.

Replay with the same retained immutable model can create a new session, submit
the complete history with zero requested output, then continue. That eagerly
evaluates formerly pending IDs: identical history does not imply identical
committed/pending counts. The new test explicitly checks6/5/1 before recreation,
6/6/0 after replay, then the independent tiny oracle's next[47,47] output.
It also checks a precancelled different prompt retains the original exact IDs,
small-buffer no-partial-copy, query and pointer/capacity errors. Existing
callback and wrong-thread tests exercise the new entrypoint as well.

The C smoke client now reads and checks the exact eight-token history after
continuation. Final qualification requires rebuilding both the library and C
client so the symbol is actually linked and invoked. This task establishes
native history/replay integration only; mobile memory, lifecycle, quality,
performance and overall M7 remain unqualified.

Formatting, strict all-target Metal Clippy,108CPU/120Metal-feature tests
(8/59ignored) and the optimized library build pass. The rebuilt C client links
and invokes the new symbol on both Mac CPU and Metal, with exact history and
[47,47,47,47] output assertions. All before/after bindings remain unchanged;
workers exit successfully with no cleanup errors. An initial targeted compile
caught a test integer-literal type and an unused import; both were corrected
before the passing checks, without numerical or tolerance changes.

The [component receipt](../../benchmarks/results/2026-10-19-m7-native-history-qualification-v1.json)
binds four sources,19 evidence files and14 exact published copies, including
the rebuilt native library/client identities and frozen tiny artifact.

Astra accepted final code/data/publication after verifying4source/19evidencepins/
14copies, protected binaries, clean native workers and all check counts.
