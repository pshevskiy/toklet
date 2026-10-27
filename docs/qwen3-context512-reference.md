# Independent real Qwen3 C512 reference

This L4 preparation uses the existing batched CPU graph and pinned Qwen3-0.6B
eight-bit checkpoint. It preserves the established Half activation/KV, original
FP32 affine-weight and FP32-logit storage contract, with separate FP32 and FP64
arithmetic. It does not execute the native engine or establish upstream BF16
arithmetic, held-out quality or a larger usable model window.

Two512-token sequences repeat the existing17-position witness prefix. The
second changes only the final token. Each case captures embedding, final norm,
all512 vocabulary-logit rows and20 boundaries in layers0/1/27. Four complete
case/precision captures use508,297,216 bytes each. The bounded ledger includes
returned graph state, tied FP32 weights, source mapping, operator/BLAS temporaries,
comparison mappings/stripes, short-reference pages and runtime metadata within
6GiB. Graphs are saved individually; no concatenated full-capture bytearray or
all-case activation dictionary is retained.

Numerical comparisons use the existing frozen per-stage/logit gates unchanged.
Both contracts also compare their first17 positions against the previous frozen
reference. Leading511-position state/logit prefixes must remain byte-identical;
attention score/probability checks use the leading511-by-511 region, and masked
probabilities must be zero. Every position retains each contract's raw argmax ID
and margin. Any disagreement marks the common greedy oracle unavailable; a
later target cannot choose the favorable arithmetic contract.

Per-capture journals and final failure cleanup preserve completed descriptors,
partial comparisons and primary errors. Qualified runs must admit exact cold
Python/NumPy/native runtime files before numerical imports, verify actual loaded
module/native-image origins afterward, retain raw command evidence, and measure
completed physical/RSS peaks. Repeated identical captures establish
reproducibility; they do not independently prove mathematical agreement.

## Actual result, 27 October 2026

Two cold full captures have identical per-array and complete-file hashes. Two
separate cold readbacks reproduce every summary, token ID and margin. The12
portable tests pass, including coherent malformed mask, causal/signed-zero,
nonfinite, argmax-disagreement and later-forward failure controls. All six owned
command groups are absent; Rust sources and protected release binaries remain unchanged.
The copied [frozen records](../tests/fixtures/qwen3-context512-v1/README.md) retain
complete expected layouts and results; large binary captures stay in artifacts.

**Numerical context qualification fails under the unchanged gates.** Across both
cases,32 contract comparisons contain7282 failing scalar values:3641 per case,
including3250 logits. The first failing captured layer is27; layers2–26 are not
observed, so this does not locate the first divergence in the complete graph.
Maximum logit difference is0.17894530296325684, with maximum gate fraction
2.197846894771957. The historical tolerances are not expanded.

All252 old17-position comparisons pass. All126 typed causal-prefix checks and
25,116,672 masked probability zeros pass per capture set. Each of1024 case/token
positions has matching FP32/FP64 argmax IDs;2048 IDs and their margins are saved
per set. The common raw greedy witness is available, but this does not establish
full-context greedy quality or waive the numerical failures.

The admitted reference ledger is5,214,106,290 bytes within6GiB. External process
peaks are1,259,243,752 physical/2,510,831,616 RSS bytes in the first run and
1,299,450,064 physical/2,472,673,280 RSS in the second. These are reference-process
measurements, not native speed or app/device-fit evidence. Exact cold runtime
admission and actual module/native-image endpoint checks pass. The first owner
correctly stops at the numerical rejection; the recovery reuses unchanged source,
plan, unit results and first capture and permits only that exact complete
rejection classification. Both replays retain it, without a tolerance waiver.

See the [public receipt](../benchmarks/results/2026-10-27-qwen3-context512-reference-v1.json).
The separately completed [precision reader](qwen3-context512-precision.md) now
localizes every failure and identifies five unique disjoint interval pairs at
position65. Existing native target acceptance uses canonical FP32 values; the
two-oracle diagnosis does not add a dual-target requirement. The [native diagnosis](qwen3-context512-native.md) now captures and independently
replays the complete C512 matrix, retaining18,400 canonical numerical failures.
Its state/argmax and schedule comparisons pass. Next selected same-input Q/K/V
dots separate inherited-input variation from local arithmetic before considering
any precision-policy or production change. Source parity, unquantized/held-out
quality, higher/full40960 contexts, app limits and phone qualification remain
separate in the [active plan](long-reply-engine-plan.md).
