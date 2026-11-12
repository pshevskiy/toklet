# LFM3 external-engine configuration preflight

The installed Transformers 5.20.2 rejects the original pinned LFM3 candidate's
configuration: `lfm3` is absent from its built-in AutoConfig mapping, and the
candidate omits an AutoConfig declaration. This blocks a same-checkpoint vLLM
comparison before model loading. The installed versions are vLLM 0.33.0+cpu,
vLLM-Metal 0.33.0, MLX 0.32.4 and MLX-VLM 0.7.11.
[Original admission probe](../artifacts/lfm3-engine-comparison-2026-11-11/config-probe-result.json).

`tools/bench/lfm3_external_config.py` verifies the complete 3,457,635,448-byte
candidate and its six configuration/text/code assets against the existing
authenticated source contract. It copies metadata into a new directory and adds
only `auto_map.AutoConfig`, pointing at the unchanged pinned parent configuration
class. The MLX-LM variant preserves the publisher's custom module. The MLX-VLM
variant additionally removes `model_file`, explicitly selecting the installed
built-in LFM3 implementation. Every numerical configuration field stays unchanged.
The original checkpoint is never written. The derived directory contains no
weights, weight links or model instance and cannot load a model.
[MLX-LM receipt](../artifacts/lfm3-engine-comparison-2026-11-11/mlx-lm-config-1/receipt.json),
[MLX-VLM receipt](../artifacts/lfm3-engine-comparison-2026-11-11/mlx-vlm-config-1/receipt.json).

The offline installed-runtime probe admits both local AutoConfig variants.
MLX-VLM's actual module/config selection rejects the publisher's MLX-LM module
because it has `ModelArgs` rather than `ModelConfig`. The built-in variant passes
that selection. No model constructor is called; an audit hook prohibits model
weight-file opens. The guarded process and supervisor's conservative lifetime
RSS sum is 536,625,152 bytes, below the separate 512 MiB reference budget.
This result qualifies configuration preparation only: vLLM engine startup,
complete loading, numerical correspondence and memory/performance qualification
are still required.
[Installed probe](../artifacts/lfm3-engine-comparison-2026-11-11/installed-config-probe-result-1.json),
[resource guard](../artifacts/lfm3-engine-comparison-2026-11-11/installed-config-probe-1-guard.json).

Eight portable controls cover source preservation, declared dispatch-only changes,
the absence of weights/links, changed weights/tokenizers/configuration code/
contracts, and existing or in-checkpoint destinations. An initial synthetic
fixture used an incorrect byte count; the failed log is retained and the fixture
now derives its actual size. An additional growth-after-stat control verifies
the actual metadata read stays bounded. Astra approves with no findings.
Check results and review are recorded in the
[publication receipt](../artifacts/lfm3-engine-comparison-2026-11-11/publication-checks.json).
Cargo/native inputs are unchanged; prior required checks are reused by exact
input comparison rather than rerunning unrelated numerical suites.

Next, run the exact BF16 weights through external startup and matched workloads
after the resident quality jobs finish, under the unchanged 6 GiB ceiling. The
built-in MLX-VLM graph is a different arithmetic implementation; configuration
admission does not establish its equivalence to the publisher or Toklet graph.
Retain token/output differences and each timing boundary. Ollama's private-store
import must independently authenticate every imported tensor before inference.
Its versioned [create/blob API](https://github.com/ollama/ollama/blob/v0.40.15/docs/api.md#create-a-model)
allows uploading source files directly to an owned server without using the CLI's
default model cache. The reviewed preflight qualifies configuration preparation only.

Follow-on setup now has an independently verified private Ollama BF16 import and
a separately owned vLLM weight-file view. The common raw-text workload freezes
thirteen 512-token inputs and a 32-sample output cap before any timing. Greedy
sampling explicitly disables the source's repetition penalty and temperature
defaults. A Transformers/tokenizer preparation attempt exceeds 512 MiB; the
standalone-tokenizer retry fits and records wrapper validation as still open.
The setup is provisional pending full startup, serial measurement, final checks
and Astra review of the complete comparison.
[Current setup evidence](../artifacts/lfm3-engine-comparison-2026-11-11/comparison-setup-state-1.json).

The owned-server startup plan now freezes private endpoints, model views,
installed source/version identities and explicit greedy/context/KV options.
Ollama and vLLM remain serial under 6 GiB and wait for the native performance
cohorts. vLLM's optional per-request timeline metrics are distinct from completed
GPU-operation clocks and will retain their installed definitions. Startup and
inference have not run under this plan.
[Frozen startup plan](../artifacts/lfm3-engine-comparison-2026-11-11/external-startup-plan-5.json).

Preparation now includes tested process birth identities, descendant/reparenting
tracking, PID-reuse rejection, conservative memory accounting and listener
ownership. Eight portable controls pass in both Python modes; a real inert Mac
server confirms the libproc ABI and owned endpoint, within 48 MiB. A separate
persistent 3+10 HTTP collector passes five controls in both modes. It preserves
stream failures and unmatched-input reasons without promoting collection to
model or performance qualification. Six actual CLI loopback cohorts now pass
64 synthetic requests: both complete protocols, cached/BOS mismatches, a
truncated stream and binary HTTP rejection. All raw response bytes match;
partial streams retain unavailable completed durations. The conservative
serial RSS charge including helper reserve is 89,440,256 bytes under 512 MiB.
The owned watchdog now passes nine live Mac lifecycle controls: complete
collection, startup exit/timeout, stalled health headers, bounded logs, memory
and client-timeout rejection, an existing listener and failed admission. Cleanup
reaps owned children and releases their endpoint while a separate same-program
server stays live. The corrected retry charges three concurrent fixture children
and admits terminal root RSS; its conservative control charge is 141,328,384
bytes under 512 MiB. The original undercounted harness estimate is retained as
unqualified. Source/model byte authentication now passes eight portable controls
in both modes. It rehashes the exact prepared view or original imported manifest
and blobs against the authenticated tensor proof; it does not replace that
original payload validator. A refreshed runtime gate passes seven controls in
both modes, including newly forked owned workers and replacement PIDs. The live
gate correctly rejects external startup while native timing is active. Large
weight rehashes remain deferred. The integrated launcher passes nine predecessor
controls in both modes and an actual deferred run with process starts, network,
model-package imports and safetensor opens denied. Isolated venv metadata confirms
eight versions and eleven selected source files, without importing an engine.
Five fixture-only integration cases cover complete collection, rejected startup,
post-run corruption after both success and failure, and wrong versions. The fixture deliberately
scopes runtime rows to its own inert processes; it does not qualify global
admission or a model. Failed backend executions reauthenticate bytes after
confirmed cleanup; unknown cleanup blocks post-run reads. Two new mocked checks
initially selected the wrong positional argument; their failed source/logs remain
and the corrected checks pass without a launcher-source change. The serial
external queue now waits for complete native collection and analysis, then runs
fresh Ollama/vLLM cohorts under the frozen limits. It stops on a failed cohort
for inspection and never commits or adopts automatically. The plan uses isolated
venv Python and explicit MLX selection; numerical/model/workload settings stay fixed.
[Preparation and queue checks](../artifacts/lfm3-engine-comparison-2026-11-11/comparison-preparation-state-7.json).

The first actual external launch attempt stops during model authentication before any engine starts.
The private Ollama v2 tag is a standard symlink to the expected pinned manifest
blob; the original strict regular-file reader rejects it. The scoped correction
requires that exact private content-addressed target and stable pointer while
keeping other link/type/hash gates. Thirteen portable controls pass in both
Python modes. Actual source/import/view reauthentication passes with all 291
referenced blobs and original hashes, within 512 MiB. This authenticates bytes;
a fresh launcher/serial attempt and all engine qualifications remain pending.
[Failed preflight](../artifacts/lfm3-engine-comparison-2026-11-11/external-cohorts-1/ollama-run1/result.json),
[corrected controls](../artifacts/lfm3-engine-comparison-2026-11-11/external-auth-controls-2.json),
[actual byte authentication](../artifacts/lfm3-engine-comparison-2026-11-11/external-auth-real-2.json).

The next retries retain a missing empty output-parent failure and a sandboxed
lifecycle failure, both corrected before useful model work. The real watchdog
then catches a non-atomic macOS footprint pair. Apple's XNU gathers the lifetime
peak before the current footprint; an owned allocation probe reproduces a
current value above that earlier peak. Accounting now retains raw values and
charges the maximum of reported lifetime and all sampled current footprints per
identity, while separately rejecting a decreasing reported peak. Twelve controls
in both modes, the bounded live allocation check, nine watchdog cases and five
integrated launcher cases pass; no cap or reserve changes.
[Apple read order](https://github.com/apple-oss-distributions/xnu/blob/f6217f891ac0bb64f3d375211650a4c1ff8ca1ea/bsd/kern/kern_resource.c#L3167),
[counter controls](../artifacts/lfm3-engine-comparison-2026-11-11/external-process-controls-2.json),
[actual allocation probe](../artifacts/lfm3-engine-comparison-2026-11-11/external-footprint-live-2.json).

Ollama's actual first request then fails in its MLX runner with convolution
weights `(2048,1,7)` and 2,048 input channels, before any validated benchmark
sample. Cleanup and post-run byte authentication pass. This is a failure of the
pinned artifact/import/backend combination, not a general Ollama rejection.
The same frozen vLLM configuration independently completes its first 3+10 cohort
with exact prompt/cache admission and cleanup/source receipts; the second cohort
is running. Numerical correspondence and complete paired performance remain open.
[Ollama attempt](../artifacts/lfm3-engine-comparison-2026-11-11/external-cohorts-3/ollama-run1/result.json),
[vLLM first cohort](../artifacts/lfm3-engine-comparison-2026-11-11/external-cohorts-3/vllm-run1/result.json).

The pinned Ollama 0.40.15 source unconditionally transposes the last two short-
convolution axes. A separate private tag adapts only thirteen singleton shape
entries from `[2048,7,1]` to `[2048,1,7]`, so that loader restores the original MLX
layout. All 286 BF16 payloads/names/dtypes, other shapes and four JSON documents
remain identical; original import/source authentication passes again. Five
adapter controls in both Python modes and the actual guarded publication pass;
peak setup RSS plus reserve is 65,683,456 bytes.
[Loader source](https://github.com/ollama/ollama/blob/v0.40.15/mlxrunner/model/lfm3/lfm3.go#L281),
[adapter controls](../artifacts/lfm3-engine-comparison-2026-11-11/ollama-conv-view-controls-1.json),
[actual publication](../artifacts/lfm3-engine-comparison-2026-11-11/ollama-conv-view-publication-1.json).

A fresh four-cohort order completes all 52 requests (40 measured), with source
reauthentication, cleanup, AC/LPM/thermal admission and observed memory below
6 GiB. Client request medians are Ollama 1.1058/1.1120 s and vLLM 1.1394/1.1398 s.
Ollama exposes no authoritative prompt/output IDs. vLLM's prompt IDs/cache match,
but only 3/13 output-ID/text rows equal the CPU reference in each cohort. These
are scoped observations; cross-engine speedup, numerical/quality equivalence and
managed adoption remain unqualified. The two diagnostic vLLM cohorts are retained
separately and do not replace any fresh cohort. All 26 genuine fresh vLLM captures
also replay through the reviewed timing CLI with side effects denied; all timing
arithmetic is consistent under 512 MiB. A transcribed SHA in the first offline
runner is corrected against the committed tool; its failed script/output remain.
[Collection and limitations](../artifacts/lfm3-engine-comparison-2026-11-11/external-collection-summary-1.json),
[actual timing replay](../artifacts/lfm3-engine-comparison-2026-11-11/actual-vllm-timelines-2/checks.json).

The subsequent [request observation report](lfm3-engine-comparison.md) now
records complete private-import authentication, four fresh external cohorts,
their guards and retained failures. That separate scope preserves missing
Ollama prompt/output IDs, observed cross-engine output differences and distinct
CPU/API/server clocks. It does not qualify numerical equivalence, quality,
speedup ratios or managed adoption.
