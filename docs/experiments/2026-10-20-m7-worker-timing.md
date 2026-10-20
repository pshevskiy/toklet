# M7 worker timing capture

The Swift worker now preserves bounded native token timing events and offers a
condition-based completion wait for non-UI diagnostics. This prepares real-model
phone measurement without estimating native latency from the UI's100ms poll.
It does not qualify phone throughput, memory or M7.3.

The existing raw-ID snapshot retains at most128 IDs. A parallel reserved array
retains at most128 timing records: native event index, native elapsed seconds
and Swift callback uptime nanoseconds. Both arrays clear on ordinary command
submission and share the existing condition lock. The callback rejects invalid
index/order, nonfinite or negative native time, or overflow before appending;
output failure leaves the sampled native token pending under the existing ABI.
No per-token queue, native ABI change, engine arithmetic or default changes are
introduced. Snapshot copies retain Swift's normal copy-on-write behavior; this
is not an allocation-free application claim.

Native elapsed seconds come directly from the Rust generation event and its
completed synchronous CPU/GPU work. The callback uptime uses monotonic
DispatchTime, sampled **after acquiring the condition lock**, so it includes any
lock acquisition delay. A caller can bracket submission with that same process
clock. The completion wait returns a snapshot after the busy flag clears,
including queued lifecycle cleanup. Its caller's completion timestamp includes
wakeup/scheduling delay and is not the exact engine-return instant. NSCondition
uses a wall-clock timeout; event timing remains monotonic. The wait rejects
nonfinite, negative and greater-than3600s deadlines and must not run on the UI
thread. Normal UI polling is unchanged.

The shared tiny checks use this wait, require matching event/ID counts, sequential
indices, finite nonnegative monotonic native time, monotonic callback timestamps
and callback timestamps inside each completed request's bracket. Zero-deadline
idle queries, invalid deadlines and clearing on reset are checked. Existing
exact IDs, ten state transitions, cancellation, transactional replay, full
unload, close acknowledgement and weak-owner release still pass on Mac CPU and
Metal. Each result retains nine ordinary-command timing records with seven
emitted events. The separate precancelled replay and interrupt-unload assertions
are state checks and do not add timing records. These tiny traces validate the
measurement mechanism; no timing speedup is claimed.

The [qualification receipt](../../benchmarks/results/2026-10-20-m7-worker-timing-qualification-v1.json)
binds current code, the reviewed native archive, exact build/runtime inputs and
raw output. Strict optimized Mac Swift compilation, both backend checks and
sixteen physical-controller regressions pass. Both simulator and unsigned
device Release SDK builds pass strict Swift concurrency/warnings-as-errors.
The inherited Xcode iPad orientation and AppIntents metadata warnings remain.
No new physical phone execution, signing or supported limits are attributed to
this component. The previously accepted
[physical tiny integration](2026-10-20-m7-physical-tiny.md) remains historical
evidence for its exact earlier app.

Astra accepted the code task and publication packet with no remaining findings,
after verifying110 evidence pins,29 copies, both SDK builds and actual Mac
CPU/Metal state and timing evidence. Phone qualification remains open.

Two raw Xcode build stdout files contain tool-emitted whitespace-only lines and
final blank lines. Path-specific Git attributes preserve their exact bytes and
exclude only those two raw evidence files from whitespace checking. Source and
documentation whitespace checks remain active; all29 staged evidence blobs match
their original hashes.
