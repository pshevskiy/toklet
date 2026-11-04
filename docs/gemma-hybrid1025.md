# Gemma4 diagnostic 1,025-position state and numerical rejection

**State, eviction/wrap, overflow and reset pass; numerical qualification is
rejected.** This does not enable ordinary Gemma Chat or change the 32-position
default. [Admission](gemma-context1025-probe.md) is separately reviewed and
committed as `614878a`.

The [native contract](../benchmarks/gemma-hybrid1025-v1/manifest.json) uses the
first 1,025 quality-corpus IDs. One schedule stops at committed counts 512, 513,
1,024 and 1,025; the other crosses both eviction boundaries in multirow calls.
Full state agrees exactly at shared endpoints 256 and 1,025 after independent
epoch checks. Every step verifies token history, all 15 physical owners' slots,
allocation and stable addresses. Appending token 1,026 preserves the full prefix;
reset reproduces all 32 initial rows and valid KV prefixes.

The 136-record capture contains every retained K/V value at four primary
milestones and eight full-vocabulary logit rows on both schedules. It occupies
104,882,176 bytes, below 128 MiB. The independent full-sequence reference stores
each producer's absolute K/V rows once. The checker independently reconstructs
retained intervals and physical slot tags; numerical failure retains the passing
state/overflow/reset verdicts.

| Contract | Compared values | Failed observations | Maximum gate ratio |
| --- | ---: | ---: | ---: |
| Mathematical FP64 |24,123,392 |4 |1.2644245636 |
| Staged FP32 |24,123,392 |8 |1.5526492736 |

The unchanged gate is `abs(delta) <= 1e-5 + 1e-4*abs(reference)`. Mathematical
K/V comparisons all pass; four mathematical logits fail at absolute position
1,024, vocabulary IDs 1822, 6212, 23958 and 74252. Selected staged logits all
pass. The eight staged K/V failures repeat four stored V values across milestones:

| Owner | Absolute position | Component | Native | Staged reference |
| --- | ---: | ---: | ---: | ---: |
|13 |433 |68 |−0.02845186740 |−0.02846741490 |
|12 |650 |0 |−0.01861946099 |−0.01863448881 |
|14 |650 |277 |0.03209404275 |0.03207981959 |
|14 |650 |296 |−0.002232567407 |−0.002216696739 |

The position-433 failure is before eviction. Both contracts' four causal
512-prefix logit rows remain byte-identical to the prior oracle. The complete
replay retains all failure coordinates, values and ratios; no threshold changes
or earlier rejection waivers follow.

Native peak RSS is 2,973,171,712 bytes and physical footprint 3,053,390,680 bytes.
The independent reference takes 6,449.57 seconds at 1,387,642,880 bytes peak RSS;
replay peaks at 370,458,624 bytes. All fit the 6 GiB process policy. These are
correctness-run costs, not generation-performance measurements. All owned
processes exit cleanly, with source/runtime/binary/input bindings and resource
guards passing. Seven oracle controls, two checker controls, formatting, strict
Clippy in both configurations, 550 default/658 Metal tests and release build pass.
The initial checker fixture failure and provisional-preview syntax failure are
retained. Final Astra approval is tracked in the evidence.

Next is one same-input V-projection/RMSNorm witness at owner13/433 and
owners12/14 at650, verifying complete projection input, raw V, normalized V and
exact cache-transfer bytes. It distinguishes local-operation behavior from the
full-sequence discrepancy without tracing every earlier layer. Mathematical and
staged qualification remain rejected. The prior four-token 27 mathematical
failures, original-BF16 cross-precision failures, full quality, free generation,
resident scheduling and app/device qualification remain open.
[Next original-quality experiment](gemma-original-quality-next.md).
[Native/replay evidence](../benchmarks/results/2026-11-04-gemma-hybrid1025-v1.json).
[Independent-reference evidence](../benchmarks/results/2026-11-04-gemma-e2b-wrap-reference-v1.json).
