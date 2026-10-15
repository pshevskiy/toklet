# M6 short-workload attention comparison

Two full eight-workload matrices and independent primary confirmations show a useful
complete numerical request reduction with Online attention on both selected models.
All four workers pass exact state, input identity, persistent memory, natural-exit
and cleanup checks. Astra accepted source, evidence and explicit short-workload
retention without findings. Longer-context release
and M6 completion remain open.

The hypothesis is that bounded online-softmax attention improves complete
numerical requests by reducing attention work and intermediate storage. The
reported primary metric is complete numerical request time. The frozen contract
names the 512/128 primary workload without selecting a unique timing scalar, so
all four recorded timing metrics and both collections are retained below. Every
metric exceeds the unchanged 5% median useful-gain threshold in both collections
and its median gap exceeds the sum of within-run MADs. Small-model prefill/TTFT
ranges overlap; no uniform or worst-case prefill gain is claimed.

## Matched execution and policy

Apple M1 Pro, macOS 26.6.2 (25G83), release Rust 1.90.0. The
[reviewed immutable worker](../../benchmarks/results/2026-10-15-m6-context-worker-qualification-v1.json)
and current compiled-source closure match byte for byte. Collector revisions are
`c0a3c7d3a51fdd250589ebe958f4d454b08525b9` in all four records.
Small uses compensated Half projections and Original normalization; larger uses
frozen Q4 v2/group8 projections and Compensated normalization. Both share the
exact selected immutable weights between separate Reference and Online sessions,
with FP16 KV, FullLogits, capacity 2048 and chunk 128. Packed/model/contract and
compiled/source identities are retained in the audit. Production defaults stay.

The primary has 512 input and 128 identical teacher-forced output IDs: three
warmups and ten measured alternating pairs, then an independent same-policy
confirmation. Other seven original rows have one warmup and three measured pairs
as an exploratory regression screen. Loading, tokenization, sampling and text
publication are excluded. Cache setup is outside continuation timing. TTFT is
numerical first-token timing, not first visible text or application latency.
Timing completes GPU work. No counter profiling, builds, tests, VM, other owned
model jobs or large identity hashes overlap the numerical collection windows.
Power endpoints show AC operation, AC low-power mode disabled and no recorded
power/thermal warning. Continuous system pressure and energy remain unavailable.

## Primary medians and dispersion

All values are seconds. Dispersion is median absolute deviation (MAD); full
minimum/maximum and all raw samples remain in the linked summary and captures.
Ten requests do not establish tail percentiles.

### Small

| Collection | Metric | Reference median ± MAD | Online median ± MAD | Median reduction |
| --- | --- | ---: | ---: | ---: |
| matrix | Prefill | 2.935538 ± 0.003798 | 2.387670 ± 0.008521 | 18.66% |
| matrix | Numerical TTFT | 2.935538 ± 0.003798 | 2.387670 ± 0.008521 | 18.66% |
| matrix | Decode | 29.401358 ± 0.007309 | 5.290397 ± 0.008250 | 82.01% |
| matrix | Complete request | 32.337806 ± 0.011139 | 7.676584 ± 0.015809 | 76.26% |
| confirmation | Prefill | 2.934170 ± 0.004769 | 2.391910 ± 0.000500 | 18.48% |
| confirmation | Numerical TTFT | 2.934171 ± 0.004769 | 2.391911 ± 0.000500 | 18.48% |
| confirmation | Decode | 29.416335 ± 0.003714 | 5.294415 ± 0.002346 | 82.00% |
| confirmation | Complete request | 32.350558 ± 0.007940 | 7.686532 ± 0.004100 | 76.24% |

### Larger

| Collection | Metric | Reference median ± MAD | Online median ± MAD | Median reduction |
| --- | --- | ---: | ---: | ---: |
| matrix | Prefill | 9.989828 ± 0.012067 | 8.585764 ± 0.010091 | 14.05% |
| matrix | Numerical TTFT | 9.989828 ± 0.012067 | 8.585765 ± 0.010092 | 14.05% |
| matrix | Decode | 68.536833 ± 0.015904 | 12.588279 ± 0.010091 | 81.63% |
| matrix | Complete request | 78.516534 ± 0.033461 | 21.183726 ± 0.019372 | 73.02% |
| confirmation | Prefill | 9.974079 ± 0.018852 | 8.572179 ± 0.011955 | 14.06% |
| confirmation | Numerical TTFT | 9.974079 ± 0.018852 | 8.572179 ± 0.011956 | 14.06% |
| confirmation | Decode | 68.546237 ± 0.007035 | 12.600152 ± 0.007106 | 81.62% |
| confirmation | Complete request | 78.503677 ± 0.023667 | 21.171542 ± 0.017460 | 73.03% |

Complete-request and decode ranges are disjoint across both collections for
both models. Their median differences greatly exceed within- and between-run
variation. Larger-model prefill/TTFT ranges are also disjoint. Small-model
prefill has one unusually fast Reference observation in each collection:
2.351454084 / 2.352250417 seconds, versus typical medians near 2.935 seconds.
Online has faster observations too; the source, tokens, modes, shapes and states
are unchanged. The cause was not isolated by endpoint telemetry. All observations
are retained, with no exclusions, tolerance changes or additional warmups.
The first collection has a minimum paired prefill gain of 3.40%, so an
every-pair 5% prefill claim would be false. Median prefill gains are 18.66/18.48%
small and 14.05/14.06% larger; complete-request gains remain strongly confirmed.

The exploratory v1 summary additionally required disjoint full ranges for every
metric: its `all_reported_metrics_confirmed_useful` diagnostic is therefore false.
That stronger diagnostic is not the frozen acceptance gate. The v2 summary keeps
the same false flag and all identical numbers, names the complete-request check
separately, and adds an exclusive output argument for reproducible replay.
No collection, sample, threshold or numerical result changed.

## Eight-workload regression screen

Complete numerical request medians below are descriptive for the seven
non-primary three-pair rows; the primary has ten pairs. Every row preserves its
exact cache and committed/pending history semantics. No median complete-request
regression appears. Fully cached TTFT accesses a prepared distribution and
performs no prefill; tiny timer differences there are not application gains.

| Workload | Small Reference / Online (s) | Larger Reference / Online (s) |
| --- | ---: | ---: |
| short | 12.348346 / 4.211044 | 30.811246 / 11.992552 |
| medium | 32.337806 / 7.676584 | 78.516534 / 21.183726 |
| long | 60.291013 / 12.584142 | 144.969574 / 33.958519 |
| boundary | 109.679010 / 22.255502 | 269.569030 / 57.521298 |
| append | 33.004221 / 5.911274 | 77.146277 / 14.235316 |
| cached | 29.405643 / 5.300707 | 68.556102 / 12.586785 |
| pending | 29.651589 / 5.346138 | 69.137517 / 12.695420 |
| changed | 32.359913 / 7.700671 | 78.536243 / 21.204128 |

## Memory, replay and decision

Small: unique paired admission 1,440,667,340 bytes; sampled joint native lifetime charge 1,276,266,392 bytes. Admission includes both session resources, metadata and a 64 MiB application allowance; shared weights are counted once. Native charge is the conservative sum of worker and collector lifetime peaks, not a reserved allowance. These 2K-capacity observations are below 6 GiB; the actual tokenizer-owning max-context application check remains separate.

Larger: unique paired admission 2,303,311,508 bytes; sampled joint native lifetime charge 2,157,644,392 bytes. Admission includes both session resources, metadata and a 64 MiB application allowance; shared weights are counted once. Native charge is the conservative sum of worker and collector lifetime peaks, not a reserved allowance. These 2K-capacity observations are below 6 GiB; the actual tokenizer-owning max-context application check remains separate.

Thirty-five portable checks pass: ten original context, four extended-wait,
seven new auditor mutation/replay, seven controlled-helper and seven Transport
checks. The [saved-data auditor](2026-10-15-m6-context-audit.md) replays all four
short records (300 events) plus two prior small-model quality records and two
budget probes: eight records, 568 events and 327 bindings. Supplemental quality
replay does not claim all higher-depth gates. No Rust/runtime source changed
since its prior accepted CPU/Metal checks; those suites were not repeated.

Retain Online as an explicit qualified short-workload optimization: complete
request reductions of 76.26/76.24% small and 73.02/73.03% larger reproduce in the
second collections, exceed 5% and observed variation, with exact state, unchanged
quality gates, bounded memory and no unexplained median workload regression.
The separate [full-loss](2026-10-15-m6-online-full-loss.md) and
[full-task](2026-10-15-m6-online-task-quality.md) gates were already accepted.
This is not a default change, external-runtime ranking, application speed claim
or long-context release. Long quality, profiles, descriptive timings, actual
max-context generation and the conditional KV decision remain before M6 closure.

[Qualification and all evidence](../../benchmarks/results/2026-10-15-m6-short-performance-qualification-v1.json)
bind code, checks, raw archives, exact replay and
[full medians/MAD/ranges/pairwise comparisons](../../benchmarks/results/2026-10-15-m6-short-summary-v2.json).
