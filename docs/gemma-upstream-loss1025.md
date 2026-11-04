# Full upstream BF16 1,024-target block

The first frozen1,025-ID block completes all1,024 next-token targets through
the reviewed stock BF16 scorer. All268,435,456logit values are finite. The32
position batches are retained in a512MiB file; independent replay reads one
batch at a time rather than retaining all distributions in memory.

NumPy stable NLL replay and greedy selection pass every target, with maximum
NLL difference7.11e-15. A separate stdlib compensated-sum anchor in every batch
also passes. Mean NLL is7.374558536943502. All8,388,608words in the first32rows
match the earlier33-ID pilot exactly; this was a descriptive comparison across
shapes, not a fitted acceptance tolerance.

Exact target order/coverage, prospective logical requests and the independently
prepared10,258-read/4,565,018,182-byte chain pass. Runtime/input/source endpoint
checks and clean owned process exit pass. On M1 Pro16GiB, peak RSS is
1,490,796,544bytes and physical footprint1,401,342,064bytes, below6GiB and the
2,276,912,256-byte shape-specific admission estimate. Authentication takes6.22s;
scoring230.51s includes the complete stock text forward, full-width head loading,
per-batch scoring and capture. This is diagnostic cost, not a retained speedup.

Astra approves the scripts/results and reusing this authenticated block as block
one. Bind its plan/capture/replay hashes in the eventual full result. Before the
remaining nine blocks, collect the paired native block with per-target statistics
and all eight prior full-row alignment witnesses, including unscored row1,024.
The full10,000-target loss gate,100-task gate, earlier fidelity rejections and
app eligibility remain open. No long scalar-oracle rerun is implied.
[Evidence](../benchmarks/results/2026-11-04-gemma-upstream-loss1025-v1.json).
