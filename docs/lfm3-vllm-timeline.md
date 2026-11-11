# Offline vLLM request timing

The [timeline reader](../tools/bench/lfm3_vllm_timeline.py) interprets optional
server timing fields from a sealed [HTTP capture](lfm3-http-metrics.md).
Run it after collection:

```sh
python3 tools/bench/lfm3_vllm_timeline.py \
  --capture /path/to/capture/report.json \
  --capture-sha256 CAPTURE_SHA256 \
  --output /path/to/request-timeline.json
```

It verifies the capture hash, replays exact Base64 wire bytes through the
unchanged collector parser, and reconciles timestamps, text, IDs and counters.
The source capture retains its client clocks and qualification flags. The
reader adds no inference instrumentation or request-path work.

## Clock boundaries

The installed vLLM 0.33.0 implementation defines these optional fields:

| API field | Unit and boundary |
| --- | --- |
| `queue_time_ms` | Milliseconds from queued to scheduled |
| `time_to_first_token_ms` | Milliseconds from scheduled to first output token; excludes queue wait |
| `generation_time_ms` | Milliseconds from first to last output token; excludes prefill/TTFT |
| `mean_itl_ms` | First-to-last interval divided by `N-1`, for `N>1` |
| `tokens_per_second` | All `N` output tokens over scheduling-to-last interval; includes prefill/TTFT |

The reader derives decode rate as `(N-1)/(generation_time_ms/1000)` only when
`N>1` and the interval is positive and finite. For example, four output tokens,
50 ms TTFT and a 1,500 ms first-to-last interval give 2 decode tokens/s and
about 2.58 overall output tokens/s. These rates have different denominators.
[Pinned implementation and interpretation](../artifacts/lfm3-engine-comparison-2026-11-11/vllm-timeline-source-contract-1.json).

The server attaches metrics to the final empty-choices usage chunk only when
one prompt and one output stream are attributable. Missing timestamps or disabled
metric emission can leave fields absent or null. Missing values remain unset;
zero/one-token requests and zero intervals do not produce a decode rate.
These are server-reported scheduler/request observations. They establish neither
GPU operator durations nor a client/server clock conversion. Content-only
latency, numerical quality and performance qualification remain separate.

## Validation and limits

The reader checks the API's ITL and overall-throughput arithmetic when the
required fields exist. The arithmetic tolerance is `1e-12` relative and
`1e-9` absolute in the compared field's units. This checks serialization and
API arithmetic; it changes no model quality or numerical gate. Inconsistency
preserves reported values, sets `arithmetic_consistent=false`, and withholds the
derived decode rate. CLI exit zero means a complete typed interpretation;
callers must inspect that consistency field before using a derived result.

The capture report is limited to 8 MiB; unused provenance/extensions stay in
the sealed source and are discarded from the replay working set. The JSONL
wire journal is limited to 96 MiB and each serialized entry to 8 MiB, accounting
for its Base64 and decoded-text copies. The original 8 MiB raw-wire, 1 MiB
frame and 4,096-event limits still apply to actual reads. Inputs must be stable
regular files. Malformed, duplicate-key, nonfinite, truncated, changed or
multi-stream evidence is rejected. The existing 32-output/context limits and
strict token/counter/clock types remain unchanged.

The schema interpretation is qualified against pinned vLLM 0.33 source.
The reader does not attest an executing runtime version, checkpoint or model
quality and always leaves its performance qualification false. The original
collector's unavailable-engine-clock note describes that collector's scope;
this additive report separately interprets optional server fields when present.

Twenty-four portable controls pass in normal and optimized Python modes.
Five explicit cases execute only the pinned primary timing function, without
importing vLLM or a model. Nine serial offline CLI cases pass with process,
network, framework and model-loading effects denied; they include the prior
synthetic loopback capture and retain all source bytes. The conservative serial
RSS charge with helper reserve is 96,501,760 bytes under 512 MiB. An audit
namespace omitted `int`, and an initial stricter check called the existing
token-ID helper with a wrong argument; both failures and source snapshots are
retained, and the corrected checks pass. Astra found an empty decoded entry
that could hide unread journal bytes, and numeric equality that could accept
wrongly typed recorded stop metadata. Both fixes have portable and actual CLI
regressions; neither invalid capture can produce a completed interpretation.
[Controls and CI composition](../artifacts/lfm3-engine-comparison-2026-11-11/vllm-timeline-controls-4.json),
[primary-source audit](../artifacts/lfm3-engine-comparison-2026-11-11/vllm-timeline-primary-audit-4.json),
[offline CLI cases](../artifacts/lfm3-engine-comparison-2026-11-11/vllm-timeline-cli-controls-3/checks.json).

The new [CI job](../.github/workflows/http-metrics.yml) runs both HTTP utilities'
52 portable tests in normal and optimized modes without model downloads or Apple
hardware. A fresh comparison verifies 5,177 relevant Cargo/Rust/native/fixture
files byte-identical to passing check inputs. Required fmt, both strict Clippy,
733 CPU/879 Metal-enabled tests and eleven examples reuse their recorded scopes,
including 52/340 ignored tests. This adds no numerical/device qualification.
[Check binding](../artifacts/lfm3-engine-comparison-2026-11-11/vllm-timeline-required-reuse-1.json).

Astra's follow-up approves scoped utility retention with no remaining findings.
[Review](../artifacts/lfm3-engine-comparison-2026-11-11/vllm-timeline-astra-review-2.json),
[commit and publication checks](../artifacts/lfm3-engine-comparison-2026-11-11/vllm-timeline-publication-checks.json).
