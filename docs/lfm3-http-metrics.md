# LFM3 external benchmark stream metrics

`tools/bench/lfm3_http_metrics.py` records one explicit benchmark request against
an owned loopback Ollama or vLLM server. The caller owns model authentication,
startup, process-memory and power guards, and the complete-job timeout. The
client's socket timeout bounds read inactivity; it is not a wall-time watchdog.
This utility changes no numerical kernel, native ABI, managed route or default
request instrumentation.

| Observation | Meaning |
| --- | --- |
| Request duration | Client monotonic time before JSON serialization/send through the terminal Ollama event or vLLM `[DONE]`; valid only after complete stream validation |
| First generated text | First nonempty response or thinking field for Ollama; first nonempty raw completion text for vLLM |
| First content | First Ollama response field; unavailable for raw vLLM text without a qualified reasoning/content parser |
| First token batch | First vLLM frame carrying reported output IDs; unavailable for Ollama, with no conversion from text chunks to sampled tokens |
| Engine work | Ollama's reported prompt/eval counts and independent nanosecond durations; vLLM's reported IDs and usage counts. No engine phase clock is inferred from SSE arrival |
| Cache reuse | Explicit returned cached-prompt count. A missing field remains unavailable and fails the zero-reuse comparison gate |
| Input match | Exact vLLM prompt IDs, prompt count and zero reported reuse; Ollama's generate endpoint does not establish exact prompt IDs |
| Failure | Raw framing and partial observations survive a handled HTTP/protocol error. An observed terminal event followed by malformed data does not become a completed request |

Ollama's API defines its own eval-count/rate denominator. The client preserves
that definition rather than converting it to Toklet's sampled, emitted, committed
or pending counters. [Versioned API](https://github.com/ollama/ollama/blob/v0.40.15/docs/api.md#generate-a-completion).
Neither API establishes a KV publication journal or displayed-text timing.
Uncached prefill rate requires the explicit cache count and a nonzero API
prefill duration. Every unavailable client duration/rate carries a reason.
The report always leaves server weight identity, model quality and performance
qualification false: its input-match flag is a narrower observation.

## Frozen requests and evidence

The CLI requires the complete workload SHA256, one of its thirteen 512-token
public inputs, and its unchanged greedy/32-output-cap policy. It adds
`add_special_tokens=false` for vLLM because the inspected completion protocol
defaults that field to true. Returned IDs still have to match the frozen native
IDs. Ollama receives `keep_alive=10m` so a loaded-model cohort does not acquire
per-request unloading. These frontend/residency additions are recorded explicitly;
the frozen prompt IDs, stop IDs, sampling policy and native workload stay fixed.

Each new output directory contains `report.json` and `wire.jsonl`. CLI exit zero
means the stream completed validation; comparison still requires its input-match
flag and the caller's independent identity, quality and resource gates. The report
binds the tool and workload hashes, trial, expected candidate and effective
request. Raw evidence records monotonic line-arrival times and exact Base64
bytes, including malformed UTF-8 and rejected truncated prefixes. Admission is
bounded by 1 MiB per line/frame, 8 MiB per response and 4,096 events, with one
extra byte to detect overflow. Rejected bytes are recorded before validation;
non-200 response bodies retain their bounded exact Base64 bytes too.
Prompt/output counts respect the context/output caps, durations fit uint64, and
integer stop IDs must fit the vocabulary. Responses with duplicate JSON
keys, nonfinite numeric values, invalid token/counter types, changed identities,
missing terminals or inconsistent usage are rejected.

Twenty-eight portable controls pass in normal and optimized Python modes, alongside
the eight unchanged configuration controls. Twelve explicit synthetic loopback
HTTP/CLI cases pass: both complete protocols, cached/BOS mismatches, both truncated
streams, binary HTTP rejection, malformed UTF-8, extreme counters, a floating
stop ID and an unterminated line. The conservative serial lifetime
RSS charge is 75,677,696 bytes under 512 MiB. The initial malformed `done` fixture
failed before reaching validation; its log is retained and the fixture is fixed.
[Portable checks](../artifacts/lfm3-engine-comparison-2026-11-11/http-metrics-checks-4.json),
[loopback controls](../artifacts/lfm3-engine-comparison-2026-11-11/http-loopback-smoke-3/checks.json),
[resource guard](../artifacts/lfm3-engine-comparison-2026-11-11/http-loopback-smoke-3-guard.json).

Replaying 26 saved genuine Ollama 0.40.15 Gemma Q4 responses preserves text,
counters and the original receiver event timestamps. Missing prompt IDs and
reported cache reuse correctly prevent matched-input qualification. This is
schema/parsed-object replay, not new frame timing or LFM3 execution. The first
helper assumed the historical options were flat; the corrected helper reads the
stored request's nested options and leaves all original inputs unchanged.
[Replay](../artifacts/lfm3-engine-comparison-2026-11-11/http-legacy-replay-3.json).

The loopback server emits synthetic observations and loads no model. Actual
Ollama/vLLM startup, runtime numerical/quality checks and serial performance
comparison remain separate work. Fresh byte comparison verifies 5,177 relevant
Rust/native/fixture/tool inputs unchanged, so required fmt, strict CPU/Metal
Clippy, 733 CPU/879 Metal-enabled tests and eleven examples are reused with their
original ignored/device scopes. [Input/check binding](../artifacts/lfm3-engine-comparison-2026-11-11/http-required-reuse-1.json).
Astra requested fixes for lost rejected bytes, oversized-counter report failure
and an untyped stop-ID comparison. All three fixes and added controls pass;
the follow-up approves scoped utility retention with no remaining findings.
[Review](../artifacts/lfm3-engine-comparison-2026-11-11/http-observer-astra-review-2.json),
[commit and publication checks](../artifacts/lfm3-engine-comparison-2026-11-11/http-observer-publication-checks.json).
