# Qwen3.5-9B consuming Metal binding

Q2.E adds a separate concrete GPU store and hybrid model factory for the pinned
uncensored9B MLX4bit source. It reuses the existing consuming upload mechanism,
Metal affine projection and embedding gather. Recurrence, convolution, attention
and session state remain on the CPU. Numerical kernels and the strict4B source
entry points are unchanged. Required checks and matched measurements pass; Astra
approves scoped retention with no actionable findings.

The cold uploader moves original packed owners into the GPU, then drops each
CPU matrix before uploading the next. It checks source/destination/temporary
allocation overlap, retained capacities and wrapper metadata with checked
arithmetic. No duplicate full CPU matrix remains. Concrete4B/9B Plans and stores
retain source separation; the9B factory resolves250slots with distinct head and
embedding during binding. Numerical dispatch stays static. The4B public store
fields, layouts and report facts are preserved.

Two actual9B captures pass the unchanged independent dual-profile gate across
33,320,448values each: zero failures, maximum gate ratio0.1874, identical lowest-ID
greedy results. All1,051stage/state/logit arrays are bitwise identical to the
reviewed CPU capture. This is raw input `[1]`, offset0, capacity512/chunk32 only.
Metal projections use the existing compensated FP32 arithmetic and original
packed Q4/group64/BF16 values. No full-model FP32 expansion is introduced.

Eight serial release cohorts on Apple M1Pro16GiB/macOS26.6.2 use3warmups and
10measured repetitions each, with reversed confirming pairs. Same-source9B
scalar CPU forward medians15.283913/15.366223s become0.236157/0.238546s with
hybrid Metal:98.45% lower latency,64.72×/64.42× faster in these scopes. CPU IQRs
are150.371/56.363ms; hybrid IQRs2.085/2.668ms. Raw samples and sample standard
deviations are retained. The4B17-input regression pairs change−3.28%/+0.34%,
with exact complete output bits and identical owner/report facts. Both4B capture
scopes also transfer all1,051arrays bitwise from reviewed Q2.C, preserving the
original independent one/context17 gates. No repeated5%4B regression appears.

Only completed ordinary forward is timed: reset, setup, loading, compilation,
upload, output comparison, logging and file writes are outside the clock.
Each process starts fresh and replays after warmup. The9B CPU/hybrid source,
weights, raw input `[1]`, position, capacities, allocator and precision match.
Metal work completes before the timer returns. These are scalar-reference versus
hybrid forward measurements, with no comparison to Ollama/vLLM or other models.
Cold hybrid loading takes22.06/22.27s and upload9.09/9.33s, separately recorded;
these costs are excluded from the forward gain. OS paging absence, request TTFT,
throughput, reliable tail percentiles, sustained runs and a1% overhead bound are
unqualified. AC power, nominal `ProcessInfo.thermalState` and Low Power Mode off
are observed before/after every cohort; boundary snapshots do not establish
continuous thermal stability.

The tests exercise exact reset/replay, zero warmed Rust allocations, cancellation,
late-head observer failure and actual GPU-head failure after248projections.
Failures commit no token and preserve the prior prefix/distribution. GPU failure
poisons execution until repair; overlap-minus-one is rejected. Successful repair
is checked by reset/replay, without claiming nonzero-prefix retry fidelity for
this9B workload. The existing independent tiny two-table graph separately checks
nonzero-prefix recovery and all3.675Mvalues under both original profiles.

The9B store reports5,036,000,192packed GPU bytes,4,245,504small CPU bytes and zero
retained CPU matrix bytes. Retained requested ownership is5,040,247,864bytes;
retained policy is5,174,465,592 and consuming-load policy5,784,265,128bytes.
The numerical session admits5,538,789,612bytes, including41,425,068GPU execution
scratch; repair overlap admits5,580,214,680bytes. Both diagnostic native lifetime
physical peaks are at most5,666,984,448bytes. The additional128MiB diagnostic
reserve and all live guards remain within6GiB. Owner admission, allowances and
physical footprint are distinct metrics.

Formatting, strict CPU/Metal all-target Clippy and667CPU/803Metal tests pass
(46/328ignored). Native62 has three arm64 slices with46exports each; Mac CPU/Metal
Csmokes pass. The app remains linked to native46; device execution is unqualified.
All38relevant Python tests,2positive/19damaged-resealed9B evidence controls and
16damaged4B report controls pass. A finite embedding rewrite with updated hashes
still fails the independent coupled gate; all original captures remain unchanged.
Initial diagnostic JSON-macro recursion, launcher/virtualenv path mistakes, an
old4B probe cfg-lint mismatch and a reader assumption that4B saved only one logit
row (it saves17) were corrected; failed logs remain in the evidence directory.
Audited cache cleanup preserves source/output/link pins. The7GB cleanup target
was missed, while minimum headroom and the2GiB live disk guard held.

All measurements use child-only `MallocLargeCache=0`, with Nano/Aggressive unset.
Default allocator, longer9B context, codec/application fit, complete requests,
summary overhead, quality and managed adoption require separate gates. Preserve
6GiBMac,512capacity/32chunk,512MiB codec policy, deferred physical iPhone and the
binding [4B128/256 rejection](qwen35-context256.md). This milestone does not
qualify those rejected contexts or change the managed route.

[Contract](../artifacts/qwen35-9b-metal-2026-11-09/acceptance-contract.json),
[checks](../artifacts/qwen35-9b-metal-2026-11-09/required-checks-2/checks.json),
[first fidelity](../artifacts/qwen35-9b-metal-2026-11-09/fidelity-1.json),
[second fidelity](../artifacts/qwen35-9b-metal-2026-11-09/fidelity-2.json),
[performance contract](../artifacts/qwen35-9b-metal-2026-11-09/performance-contract-1.json),
[raw performance/summary](../artifacts/qwen35-9b-metal-2026-11-09/performance-summary-1.json),
[evidence controls](../artifacts/qwen35-9b-metal-2026-11-09/evidence-controls-1.json),
[checks/review/commit](../artifacts/qwen35-9b-metal-2026-11-09/publication-checks.json).
