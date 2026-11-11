# LFM3 BF16 console and external request observations

Status: Astra approves scoped observational publication, with no findings. These are scoped
latency observations on different execution profiles. Exact cross-engine
numerical/output equivalence, quality, speedup ratios and adoption remain
unqualified. The isolated finite-widening candidate is excluded.

The named hardware is an Apple M1 Pro Mac with 16 GiB unified memory. All routes
consume the authenticated mchen04 LFM3 Heretic BF16 MLX checkpoint at revision
88636d45d403e7c8ab6a40e85c9d0fa90cc5a253. Its 3,457,635,448-byte source has
SHA-256 b6ff17ac993dddf0f6f51bd872b12ae1437f4453a54a0b4878cedc7864fbc07e.
The publisher's abliterated description is a model identity, not a behavioral
guarantee. Source identity does not establish equal arithmetic across engines.

## Work and clock boundaries

Each engine has two fresh-process cohorts with three warmups and ten measured
requests each: 78 requests total, of which sixty are measured. Every request
has 512 raw-text input tokens and produces 32 output tokens at context capacity
2,048, temperature zero and no repetition penalty. Distinct first prompt tokens
avoid an identical prefix between requests. Native state resets and reuses zero
tokens. vLLM reports exact prompt IDs and zero cached prompt tokens. Ollama's
API reports counts but exposes no authoritative prompt/output IDs; its input
match stays unqualified. No chat template is applied.

The native CPU reference uses the canonical BF16 source profile and prefill
chunks of 128. vLLM uses its MLX Metal backend with one sequence, batch-token
limit 512, 64 MiB KV cache, disabled prefix caching and eager execution. Ollama
uses its MLX runner, a private model store, one loaded model and one request at
a time. Different batching and arithmetic paths remain explicit.

The external order is Ollama1, vLLM1, vLLM2, Ollama2. Native baselines were
collected in their separate serial campaign; this is not one interleaved
cross-engine campaign. Source reauthentication and cleanup bracket each external
cohort. AC power, Low Power Mode off, thermal state zero and no competing named
model/compiler work are sampled. These endpoint/sampled checks do not prove
continuous global exclusivity.

The native request clock includes console transport through parsed/logged
completion. External clocks include HTTP transport through the terminal event
and earlier wire logging. First generated text means the first nonempty client
text event. It is neither first-content nor first-visible latency. Native
after-done flags are false in both measured baseline cohorts. Loading and
engine-reported prefill/decode clocks remain separate; no client/server clock
subtraction or denominator conversion is applied.

## Completed observations

Values below are medians over ten measured requests, in seconds. All raw
samples, executable/model identities, captured report/wire digests and memory
charges are in the [machine-readable record](../benchmarks/results/2026-11-11-lfm3-engine-comparison-v1.json).

| Route | Cohort | Client request | First generated text | Exact input identity | Output IDs matching CPU |
| --- | --- | ---: | ---: | --- | --- |
| Toklet canonical CPU | 1 | 349.5412 | 310.2947 | Yes | Reference |
| Toklet canonical CPU | 2 | 349.5760 | 310.2873 | Yes | Reference |
| Ollama 0.40.15 MLX | 1 | 1.1058 | 0.4373 | Unavailable | Unavailable |
| vLLM 0.33.0 / Metal 0.33.0 | 1 | 1.1394 | 0.4411 | Yes | 3/13 |
| vLLM 0.33.0 / Metal 0.33.0 | 2 | 1.1398 | 0.4427 | Yes | 3/13 |
| Ollama 0.40.15 MLX | 2 | 1.1120 | 0.4393 | Unavailable | Unavailable |

Output comparisons include all thirteen requests, including warmups. Raw text
also matches CPU on only 3/13 requests in each external cohort. Matching counts
or a common checkpoint do not make the remaining outputs equivalent. No
cross-engine speedup ratio or quality/adoption result is qualified.

The native baseline's frozen executable SHA-256 is
1f9f9442aed1a410b0d1c1dbdf271d43fae54c4bde3a4351568e24598857ac15.
Its receipt does not recover an exact source revision; candidate-checkout
provenance must not be attributed to this baseline. The actual execution
profile, binary identity and raw results are retained. External distribution
metadata and eleven selected source files are checked; this is not a full
runtime source attestation.

All observed memory guards remain below 6 GiB. Maximum conservative charges
including the helper reserve are approximately 3.56/3.33 GiB for the native
baselines, 3.90/3.91 GiB for Ollama and 5.41/5.41 GiB for vLLM. The record
preserves exact byte counts and each accounting scope. Shared pages can be
counted more than once, and short-lived undiscovered work or final peaks can be
missed. This is not a continuous allocator cap or a memory-saving claim.

## Optional vLLM request timelines

All 26 genuine vLLM captures replay through the reviewed offline timing CLI
with network, subprocess, framework, model-open and signal effects denied.
Timing arithmetic is consistent for every capture. Across the measured cohorts,
server scheduled-to-first medians are 434.18/434.96 ms, first-to-last intervals
701.84/700.41 ms and queue intervals 0.01221/0.01290 ms. Derived decode rates
use 31 divided by the first-to-last interval: 44.17/44.26 tokens/s. Reported
overall output rates, 28.17/28.17 tokens/s, include scheduled-to-first time.
These are independent field medians and scheduler request clocks; they do not
time completed GPU operators or establish any first-visible event.
[Actual replay](../artifacts/lfm3-engine-comparison-2026-11-11/actual-vllm-timelines-2/checks.json),
[metric definitions](lfm3-vllm-timeline.md).

## Import, guards and retained failures

The original private Ollama import authenticates all 286 BF16 tensor payloads,
names, dtypes and shapes plus four JSON documents. Its pinned loader then
unconditionally transposes the convolution axes, although this MLX source
already stores the target layout. A separate private tag changes only thirteen
singleton shape entries from [2048,7,1] to [2048,1,7]; the loader restores the
original layout, with every payload byte preserved. All other shapes and JSON
files remain unchanged. The original source, tag and imported blobs remain
intact. Five adapter controls in both Python modes and actual source/import/view
authentication pass under the separate 512 MiB setup limit.
[Pinned loader](https://github.com/ollama/ollama/blob/v0.40.15/mlxrunner/model/lfm3/lfm3.go#L281),
[adapter/authentication](../artifacts/lfm3-engine-comparison-2026-11-11/ollama-conv-view-publication-1.json).

Preparation failures remain separate from successful collection: standard
private-manifest symlink handling, a missing empty output parent, sandboxed
process inspection and independently read macOS footprint counters. An owned
allocation probe reproduces current footprint exceeding an earlier lifetime
peak. Accounting preserves both raw values, charges their maximum over time
per retained identity and separately rejects decreases of the reported lifetime
peak. No memory limit or reserve is lowered.

Source authentication, PID/birth ownership, runtime admission and the launcher
have respectively 13/12/7/9 portable controls in both Python modes. Nine actual
watchdog cases and five integrated fixture cases exercise completion, failure,
deadlines, bounds and cleanup while preserving an unrelated server. The raw
collector has five controls per mode and six CLI loopback cohorts/64 synthetic
requests. Diagnostic vLLM cohorts are retained separately and substitute for
none of the four fresh cohorts. Post-run source checks and cleanup pass after
successful and failed backend execution.
[Collection evidence](../artifacts/lfm3-engine-comparison-2026-11-11/external-collection-summary-1.json),
[configuration preflight](lfm3-external-config.md).

The first report draft incorrectly assumed buffered native first text; raw
timestamps and flags corrected it before review without changing measurements.
An earlier offline replay runner had a transcribed tool hash; the corrected
hash matches the committed tool. Both drafts/failures remain in the evidence.
Telemetry overhead, sustained/tail latency, engine equivalence, managed model
adoption and the wider roadmap remain separate and unqualified by this report.

[Checks, review and publication receipt](../artifacts/lfm3-engine-comparison-2026-11-11/comparison-publication-checks.json).
