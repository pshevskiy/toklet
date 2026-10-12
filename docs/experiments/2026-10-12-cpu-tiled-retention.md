# Retain the ordered FP64 CPU tile, 12 October 2026

Retain the measured tile as the qualified opt-in CPU projection variant exposed
by `--cpu-projection tiled`. Scalar remains the default and explicit reference
variant; this decision does not change constructors, numerical sources or
qualified executables. M3.1 closes: both primary comparison pairs, all eight
numerical workloads, the full application trace, unchanged frozen quality
gates and memory/capacity checks pass their recorded acceptance scope. The
bounded public CPU/ANE feasibility experiment is already closed, so M3 is
complete and implementation proceeds to M4.

The [implementation](2026-10-10-cpu-tiled-engine.md) targets projections,
which accounted for 99.24% of the recorded scalar profile. A four-token /
four-output tile reuses activations and contiguous weight rows, preserving
feature-order FP64 accumulation, bias placement and final FP32 rounding.
Backend selection stays outside inner loops. Row-major weights, FP32
weight/KV storage, session ownership, scratch, tokenization, sampling and
stop/cancellation semantics are unchanged. Scalar functions and independent
fixtures remain available; no tolerance or quality gate is relaxed.

## Primary acceptance and workload coverage

The declared primary metric stays warm TTFT for 128 input / 128 fixed output
tokens. The [same-executable comparison](2026-10-10-cpu-primary-comparison.md)
uses effective order scalar/tiled/tiled/scalar, each with three warmups and ten
requests. Both pairs improve TTFT by 77.6400% / 77.6009%, exceeding the unchanged
5% retain threshold and maximum observed median variation of 0.3531%.
Request reductions are 52.0024% / 52.0992%. The originally overlapped scalar run
is excluded and replaced by its accepted clean run; the stale orchestration
entry must not be used as the baseline. The
[primary calculation](../../benchmarks/results/2026-10-10-cpu-primary-analysis-v1.json)
retains exact bindings, raw-derived summaries, pair arithmetic and variation.

Every following row emits 128 fixed IDs with EOS ignored, capacity 2,048 and
chunk 128. Cache preparation is outside request timing. All measured requests
retain states and monotonic events. The short row is pair 1; all other scalar /
tiled comparisons are separate historical runs and their ratios are descriptive,
not additional alternating primary comparisons. Times are seconds; decode
rates count 127 intervals after the first event.

| Workload | Scalar TTFT | Tiled TTFT | Scalar request | Tiled request | Tiled decode steps/s | Request reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [Short, 128 input](2026-10-10-cpu-primary-comparison.md) | 40.221369 | 8.993484 | 128.561790 | 61.706524 | 2.409895 | 52.0024% |
| [Medium, 512](2026-10-11-cpu-tiled-medium.md) | 163.381240 | 40.961032 | 253.195493 | 97.892741 | 2.232219 | 61.3371% |
| [Long, 1,024](2026-10-11-cpu-tiled-long.md) | 336.469968 | 85.683956 | 428.750184 | 142.368874 | 2.239058 | 66.7944% |
| [Boundary, 1,920](2026-10-11-cpu-tiled-boundary.md) | 653.578164 | 186.521378 | 749.021123 | 246.854774 | 2.103946 | 67.0430% |
| [Append, 512 + 64](2026-10-12-cpu-tiled-append.md) | 21.348085 | 5.839271 | 111.628890 | 63.166372 | 2.214753 | 43.4140% |
| [Fully cached, 512](2026-10-12-cpu-tiled-cached.md) | 0.0000007920 | 0.0000008545 | 90.000207 | 54.425495 | 2.333468 | 39.5274% |
| [Pending token, 512 + 1](2026-10-12-cpu-tiled-pending.md) | 0.702565 | 0.425491 | 90.500725 | 54.797608 | 2.335707 | 39.4506% |
| [Changed prefix, 512](2026-10-12-cpu-tiled-changed.md) | 163.551572 | 38.530052 | 253.054154 | 93.038050 | 2.332475 | 63.2339% |

Request and decode medians improve in every row. The cached first-event median
is 0.8545 microseconds versus historical scalar 0.7920, an absolute difference
of 62.5 nanoseconds. Their observed TTFT ranges overlap (scalar 0.583–1.000,
tiled 0.666–1.042 microseconds); the empty evaluation can measure 0–42 ns.
These near-floor timings do not establish a cached-TTFT percentage ranking or
contradict the measured projection benefit. This interpretation changes neither
the declared primary metric nor its threshold.

The state matrix covers an empty cache with a nonempty prompt; exact 512-token reuse plus 64 appended
tokens; fully cached distribution reuse; fixed stand-in 2077 staged outside
request timing and evaluated inside; and an edit at zero-based index 128 that
resets/replays all 512 tokens. The [boundary](2026-10-11-cpu-tiled-boundary.md)
verifies all 2,048 positions committed after evaluating the final pending ID,
then rejects another token before mutation. The separate final capacity-response
JSON is not retained: exact driver assertions and its published capacity note
are the disclosed binding, rather than a separately archived response.

The [full application trace](2026-10-12-application-tiled.md) has three warmups
and ten three-turn scenarios, median 129.141379s versus historical scalar
263.230241s (descriptive 50.9398% reduction). All thirty measured token-ID
sequences, text hashes, stops and untimed generation states match scalar,
including hidden pending EOS and reset finalization. Its timers include
transport/dispatch, tokenization/formatting, sampling, streaming and response
finalization. Later prompts use scripted assistant messages, so this is latency
and state evidence, not live tool execution or task-quality scoring.

## Correctness, quality and memory

The original functional qualifications retain selected real-model trace/greedy
and session/cache comparisons, exact scalar/tiled projection behavior and
unchanged gates. The
[numerical qualification](../../benchmarks/results/2026-10-10-cpu-tiled-qualification-v1.json)
binds isolated worker SHA-256
`3601c63ae83b0d9b20f8362bbda27781e4dd969d527918c072ccf270d2d4874d`;
the [CLI qualification](../../benchmarks/results/2026-10-10-cpu-tiled-cli-qualification-v1.json)
binds isolated CLI
`8e1be695229fb4a46e49725f06050b3296f6fb398e718fca8567bf0c6b400b43`.
Dispatch commits identify measurements; numerical build bases, committed
source hashes and build flags are recorded independently. No numerical/helper
or executable changes are made by this closure.

The [full loss comparison](2026-10-11-held-out-tiled-loss.md) scores 10,000 targets
in independent blocks. Mean NLL is 2.883725149929484 for both; every complete
logits hash, target FP32 bit pattern and per-target NLL matches. Increase 0
passes the unchanged 0.05-nats/token gate. This is chunk 1/capacity 1,025 evidence;
it does not substitute for chunk 128 generation quality.
The [full task comparison](2026-10-11-held-out-task-quality.md) instead covers
all 100 frozen cases with the actual chunk 128 CLI. Responses/token IDs/stops and
all grading outcomes are identical. Both score 2/100, so loss 0 pp passes the
unchanged 2 pp gate. All 40 literal code cases fail with SyntaxError; outputs are
not repaired. These are project-relative acceptance gates, not broad model
capability or an external quality/speed claim.

Logical numerical memory remains 1,976,131,072-byte weights, 50,331,648-byte
allocated KV and 10,436,608-byte scratch. Conservative numerical admission is
2,172,348,800 bytes. The eight candidate-row maximum physical child charge is
2,043,840,528 bytes; primary comparison's largest scalar/tiled peak difference
is 81,920 bytes, with unchanged logical layout. Full application admission is
2,353,012,560 bytes and its physical CLI peak 2,123,745,392 bytes, with the complete
logical inventory equal to scalar and summing exactly to admission. All are
below 6 GiB. Physical charges describe own children; energy and system pressure
are unavailable. The evidence supports no unexplained working-set regression
in this qualified scope, not complete system-memory ranking.

## Sustained scope, exclusions and decision

The primary tiled pair 1's ten measured requests total 617.825464s; confirming
pair 2 is separately accepted. Other uninterrupted measured totals are medium
965.754241s, long 1,424.459786s, boundary 2,469.970157s, append 629.436638s,
changed 931.148143s and application 1,292.114342s. These exceed ten minutes
within their respective runs. Cached 545.308331s and pending 548.999434s do not
add separate ten-minute timed-request qualification. Parent holds between
historical primary runs are disclosed; they do not form one uninterrupted
four-run sequence. No holds occur inside these retained request sequences.

Hardware is Apple M1 Pro / MacBookPro18,3 / 16 GiB / macOS 26.6.2 (25G83), Rust 1.90.0
and ordinary recorded optimized builds without RUSTFLAGS. CPU rows retain
startup AC observations but query the low-power policy when publishing their
completed results, including the paired primary records. Application records
have explicit startup AC/policy observations. Endpoint warning reports are recorded, but continuous temperature,
clocks, energy, system pressure, formal confidence intervals and reliable tail
percentiles are unavailable. Source hashing warms storage. Claims remain
limited to these named workloads, artifact, power observations and hardware.
The frozen Qwen/tokenizer revision is
`7ae557604adf67be50417f59c2c2f167def9a775`; every linked result binds actual
checkpoint/tokenizer hashes, precision, cache, compiler and source identity.

The [dependency-failed medium attempt](2026-10-11-cpu-tiled-medium-excluded.md),
[paused cached attempt](2026-10-12-cpu-tiled-cached-excluded.md) and overlapped
original primary scalar run remain wholly excluded. Fresh accepted runs replace
them; partial rows are never reconstructed into acceptance data. The original
held orchestration later disappeared, with exit status/time/cause unknown;
[current recovery](../../benchmarks/results/2026-10-12-cpu-tiled-cached-v2-excluded-idle.json)
explicitly proves current recorded-PID/group absence. Final application
[natural completion and idle](../../benchmarks/results/2026-10-12-application-tiled-v3-idle.json)
prove its owned chain gone, VM stopped and protected endpoints unchanged.

Retain the bounded tile, preserve scalar as default/reference, and advance to
M4. The CPU experiment adds a measured selectable mechanism; it does not add
an ANE backend or qualify Metal, Q4, long-context or iPhone execution. Those
milestones retain their separate exit criteria. Saved-data arithmetic/content,
links and whitespace checks apply to this documentation closure; no new engine
test suite or numerical run is required. Astra accepted retention and M3 closure after correcting cache-state and power-observation wording; saved evidence and numerical results are unchanged.
