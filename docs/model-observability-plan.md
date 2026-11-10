# Model roadmap: observability before optimization

Updated 9 November 2026. Selected order: **Gemma 4 E2B → Qwen3.5-4B →
Qwen3.5-9B**. Qwen2.5-1.5B remains a historical regression reference, not the
next adoption target. The [O1 opt-in pilot](o1-gemma-request-summaries.md) now
implements available worker-boundary observations. The additive
[O2.A phase API/report](o2-gemma-cost-report.md) now exposes completed native
phases. [O2.B1 numerical coverage](o2-gemma-workloads.md) adds two matched
cold/continuation batches and a ten-minute sustained run; UI/device/other-workload
gaps remain explicit and the 1% summary overhead bound is unqualified.
[O2.B2.A publication metrics](o2-ui-publication.md) correlate Send with controller
handling; visible-frame qualification remains separate. [Current coverage](model-coverage.md) describes
what actually runs. [PLAN](../PLAN.md#validation-and-performance-policy) retains
the binding numerical, quality, memory and measurement gates.

## Delivery order

| Stage | Delivery | Acceptance before advancing |
| --- | --- | --- |
| O1 | Versioned request summaries on the existing managed Gemma 4 E2B BF16 Mac route; disabled/summary modes and bounded JSONL export | Metric definitions below; authoritative success/rejection/error/Stop/replay state; clock and memory scope; collector pressure and exporter-failure checks; telemetry overhead qualified or explicitly opt-in |
| O2 | Gemma baseline, phase breakdown and offline comparison report | Exact artifact/build/workload manifests, raw samples, missing-field reasons and state counters; separate cold load, warm generation, continuation and sustained runs; identify the largest measured cost |
| G1 | Gemma 4 performance work and preferred uncensored MLX artifact evaluation | One measured change at a time; preserve qualified BF16/G2 reference. New weights/quantization get independent fidelity, task-quality and memory gates before managed adoption |
| Q1 | Qwen3.5-4B, preferred uncensored affine MLX 4-bit | Inspect actual graph/packing first; qualify loader, hybrid recurrent/attention state, template/stops, capacity and cancellation; expose the same request metrics before optimization |
| Q2 | Qwen3.5-9B using the qualified mechanisms from Q1 | Separate artifact/shape/quality/admission evidence; repeat baseline and overhead checks. Keep the 6 GiB ceiling; a failed fit does not justify silent paging or a budget increase |
| S1 | Five under-4B candidates below, one at a time | Measured quality/latency/memory shortlist, then explicit individual adoption decisions. If Q2 is memory-blocked, record it and evaluate this smaller-model track |

The remaining [under-30B candidates](models-under-30b-2026-11-07.md) are research
options for later: Gemma 12B/26B, Qwen3.8-27B, Devstral Small 2, LFM2.5-24B,
Ministral 14B, gpt-oss-20b and Granite 8B. They do not displace the selected order
or expand the current working-set budget. iPhone remains deferred.

## Preferred MLX artifacts

Prefer **uncensored/abliterated MLX** deployments, with a reproducible lineage
and a verified numerical/quality result. These are publisher descriptions, not
an assurance of capability or refusal behavior. Preserve the original checkpoint
as the reference; record the modified checkpoint as a distinct model identity.

| Main route | Candidate, inspected revision prefix | Format and adoption work |
| --- | --- | --- |
| Gemma 4 E2B first | [deadbydawn101/gemma-4-E2B-Heretic-Uncensored-mlx-4bit](https://huggingface.co/deadbydawn101/gemma-4-E2B-Heretic-Uncensored-mlx-4bit), `cde01efc` | Affine 4-bit, group 64; derived from p-e-w's Heretic ARA checkpoint. Evaluate after O1/O2 on the qualified official BF16 route. Existing Gemma Q4/QAT quality failures remain binding for their artifacts; they neither qualify nor automatically reject this different artifact |
| Qwen3.5-4B second | [avan-ag/Qwen3.5-4B-Uncensored-MLX-4bit](https://huggingface.co/avan-ag/Qwen3.5-4B-Uncensored-MLX-4bit), `fb98fbc5` | Affine 4-bit, group 64; DavidAU fine-tuned lineage, not simply stock Qwen quantization. Preserve this distinction in quality and timing comparisons |
| Qwen3.5-9B third | [lukey03/Qwen3.5-9B-abliterated-MLX-4bit](https://huggingface.co/lukey03/Qwen3.5-9B-abliterated-MLX-4bit), `8ff61b6a` | Affine 4-bit, group 64, text-only configuration. Inspect exact packed tensors and memory before admission; do not transfer multimodal claims from the official model |

Full inspected revisions/configuration are in the
[artifact record](../artifacts/model-observability-plan-2026-11-07/artifact-candidates.json).
At implementation freeze every consumed file hash, tokenizer/template, converter
version, quantization mode/group/exclusions and source revision. Prefer direct
conversion from high-precision modified weights over re-quantizing GGUF-derived
weights. An MXFP4/NVFP4 model is not interchangeable with affine Q4.

Qualify two differences separately: the modified high-precision checkpoint versus
the original on held-out tasks, then quantized/native execution versus that exact
modified checkpoint. Keep the existing loss/task gates; report intentional text
behavior changes separately without using them to excuse unrelated capability
loss. An MLX repository is an artifact/external-baseline candidate, not evidence
of Toklet loader, kernel or managed-app support. No blanket family adoption.

## Five candidates strictly below 4B total parameters

Ranked research shortlist for capability, local efficiency and useful model
coverage, not a common-harness leaderboard. Counts below are full unquantized
checkpoint tensor-element counts from the publisher's Hub metadata, including
stored embeddings and modality tensors; neither active nor effective counts.
The [metadata receipt](../artifacts/model-observability-plan-2026-11-07/model-metadata.json)
and artifact record preserve revisions. Recheck unique tensor ownership/aliases
and every derivative's actual shapes at admission.

| Rank | Official reference | Total parameters | Reason to evaluate | Uncensored / MLX path |
| --- | --- | ---: | --- | --- |
| 1 | [MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B) | 2,516,756,480 | Strong small reasoning, coding and tool-use candidate; ordinary Llama-style model type | Prefer accessible [Heretic MLX 8-bit](https://huggingface.co/bumblebuttpow/MiniCPM5-2B-heretic-abliterated-MLX-8bit), `50ab1b22`, with declared insraq parent `8d83eaf4`; independently qualify. Original mondk `8ebe35e0` 4-bit now returns401; retain its historical preflight |
| 2 | [LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B) | 1,728,801,536 | Recent compact agent/instruction-following model; useful performance experiment with a new architecture | Prefer the separate [mchen04 Heretic BF16 MLX](https://huggingface.co/mchen04/LFM3-1.7B-Heretic-Abliterated-MLX-bf16), `88636d45`, then independently qualify bounded Q4 conversion. Its declared Vortecks BF16 parent is pinned; payload/quality/admission are open. The previous `095df4e8` GGUF-to-F32 artifact still exceeds6GiB in weights alone and remains rejected |
| 3 | [Granite 4.2 3B](https://huggingface.co/ibm-granite/granite-4.2-3b) | 3,659,737,600 | Dense reasoning/tool-use baseline with adjustable thinking effort | [Heretic source](https://huggingface.co/saidutta69/granite-4.2-3b-heretic), `db83ced6`; uncensored MLX conversion pending. [Official MLX Q4](https://huggingface.co/ibm-granite/granite-4.2-3b-q4-mlx), `0c6f39b1`, is a reference fallback; strict config translation and qualification remain open |
| 4 | [Ministral 3 3B Reasoning](https://huggingface.co/mistralai/Ministral-3-3B-Reasoning-2512) | 3,849,090,048 | Compact STEM and vision reasoning; count includes vision encoder | [Heretic source](https://huggingface.co/xtools-at/Ministral-3-3B-Reasoning-UNCENSORED-heretic), `46fb9a42`; uncensored MLX conversion pending. [MLX 4-bit reference](https://huggingface.co/mlx-community/Ministral-3-3B-Reasoning-2512-4bit), `2cd2087a`, has a stale index; its single-file header is diagnostic only |
| 5 | [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) | 2,274,069,824 | Small vision/reasoning candidate that can reuse mechanisms qualified for Qwen3.5-4B/9B | [TheCluster uncensored MLX BF16](https://huggingface.co/TheCluster/Qwen3.5-2B-Uncensored-HauhauCS-Aggressive-MLX-bf16), `78efacde`; header differences include15omitted MTP tensors and18norm vectors narrowed to BF16. Authenticate/qualify these changes before affine Q4 conversion |

Gemma 4 E2B's effective name does not make it under 4B total; it remains the
main first model. Qwen3.5-4B is also outside this strict tier. Nanbeige4.2-3B was
investigated but has 4,169,800,704 stored parameters at `b82e54bd`, so is excluded.
Do not use quantized container word counts to claim a smaller parameter class.
Availability of a publisher's conversion is not a passing local benchmark.

The [MiniCPM5 metadata preflight](../artifacts/minicpm5-preflight-2026-11-09/preflight-summary.json)
finds296packed matrices/85small tensors in the pinned1.416GB conversion; its
2.517B logical parameters match the official declaration. FullFP32 expansion
would require10.067GB, so use a bounded independent reference. The declared
modified BF16 parent's current anonymous API requests return401; cached listings
do not authenticate its availability or lineage. No payload was downloaded,
and source/codec/numerical/quality/admission gates and model order stay unchanged.

A10November availability check now gets401 for that conversion's pinned API,
HEAD and config as well as its declared parent; official/search controls succeed.
No weights were downloaded. The separate [accessible8bit preflight](../artifacts/minicpm5-heretic8-preflight-2026-11-10/preflight-summary.json)
verifies973physical entries/381logical descriptors, the2.674GBdeclared object,
exact index and unaligned110386-byte payload origin. Its nested and flat default
RoPE declarations both say5e6; S1.A adds explicit source admission without
rewriting publisher files or inventing defaults. The parent config declaresBF16,
while the conversion card calls itFP16; exact conversion lineage still needs proof.
Full source, model execution, codec, physical fit, quality/request/adoption and
any quantization tradeoff remain unqualified. A separate ARA4bit alternative has
mixed3/4/5/6/8-bit overrides and no declared parent in its short card; existing
affine layouts represent those widths, but no execution or lineage is qualified.
[Implementation contract](../artifacts/minicpm5-default-rope-2026-11-10/acceptance-contract.json).

The [first short console vertical](minicpm5-console.md) authenticates the 8-bit
publisher object and passes 52.9M dual-profile native comparisons plus state and
fresh package checks. Matched readback cohorts find no 5% request benefit;
existing defaults stay unchanged. Quantized conversion lineage, original
quality, codec/deeper numerical and adoption remain open; this is bounded Mac
evaluation rather than blanket MiniCPM support.

The [bounded MiniCPM5 CLI profile](minicpm5-profile.md) supplies calibrated
per-layer/operator intervals with complete-logit parity and explicit diagnostic
inflation. Its [exact-reduction RMS store optimization](minicpm5-rms-store.md)
passes the original numerical gates and reduces warm short CLI request time
21.51%/21.50% in two runs. Astra approves the explicit Mac opt-in; defaults and
broader model/adoption gates stay fixed.

[Bounded parent authentication](minicpm5-parent-auth.md) now verifies both full
BF16 parent object hashes (5,033,557,128bytes), all381logical tensors and85exact
copied norms. It retains1,349tensor-block digests without storing the payload.
This authenticates source identity. The separate
[bounded matrix reproduction](minicpm5-matrix-lineage.md) now reconstructs all
296 matrices/888 affine arrays with zero byte mismatches while freshly
authenticating both parent objects, under a frozen GPU/BF16 recipe. Checks and
Astra review pass; publisher's exact binary remains unknown. Original quality,
deeper numerical and adoption
stay open. The separate
[bounded basic codec/text CLI](minicpm5-text-codec.md) now passes original-template/
tokenizer/stream and persistent-cache gates, fresh required/native checks and
two complete-logit native text parity workloads. Eight matched request cohorts
clear the scoped5%screen with18.3–18.9% and11.8–12.3%request gains for the prior
RMS opt-in. Astra approves scoped retention;
[review/commit](../artifacts/minicpm5-text-codec-2026-11-10/publication-checks.json).
No model adoption is implied.

The [complete MiniCPM5 quality inputs](minicpm5-quality-inputs.md) now freeze all
10,000 targets and original 100 tasks/caps for separate stock-to-modified and
modified-to-native comparisons. Full cold source replay and altered-input
controls and Astra review pass. All corpus IDs and 100 task renders/IDs
match pinned stock metadata, whose weights remain undownloaded. Loss requires
1,025 positions and the largest task 559; the current 512 diagnostic limit stays
fixed until separately qualified. No quality score or adoption follows.

The [LFM3 metadata preflight](../artifacts/lfm3-preflight-2026-11-09/preflight-summary.json)
checks286BF16 tensors/3.458GB for the new candidate and594physical tensors/0.973GB
for the official Q4 reference. Its conversion manifest links to an accessible,
pinned modified BF16 parent's expected hash; complete payload bytes are still
unauthenticated. This graph needs7-tap convolution, rotating256-token windows,
sandwich/QK norms, per-head attention gates and layer-specific partial RoPE.
Existing LFM2 support and publisher parity results do not qualify those mechanisms.

The [remaining small-model audit](../artifacts/small-model-metadata-2026-11-09/preflight-summary.json)
checks53metadata files and12canonical/diagnostic headers across8pinned sources.
Granite's BF16 payload is7.319GB; its original index understates physical bytes by
514MB, while the inspected header owns both untied tables. Ministral's text-only
BF16 weights already require6.858GB. Both exceed6GiB before session/application
owners and need separately qualified bounded conversion. Ministral also needs
3,072hidden/4,096query width and YaRN source semantics; the core model represents
independent widths, but its current source adapter does not admit this model.
The selected Ministral Q4 index has stale filenames, tensor inventory and size;
its header diagnostic changes no source-admission policy.
Qwen2B's removed15MTP tensors explain60,828,160fewer stored parameters. Its18norm
precision changes and convolution/vision layouts are metadata observations,
not tensor-byte parity. Separate24-layer/16-value-head binding, codec, numerical/
quality, physical fit and request/overhead gates remain open. No payloads were
downloaded and model order/context caps stay unchanged.

The isolated [all-source FP32 Metal projection gate](qwen35-gpu-projections.md) passes.
The [concrete executor](qwen35-metal-executor.md) passes its scoped coupled
state/recovery checks and both matched forward performance cohorts; Astra
approves scoped retention with no findings.
Next are admitted contexts, codec/quality and the same
completed-work/request metrics. An isolated operator result does not admit a
managed route; codec, quality, admitted-context and full-request gates still apply.

Pinned [4B codec metadata](../artifacts/qwen35-codec-preflight-2026-11-08/preflight-summary.json)
has a distinct tokenizer; the old dense Qwen3 codec cannot be reused by
model-family name. After performance collection, the tokenizer payload was
authenticated. Model/generation EOS 248044 resolves to `<|endoftext|>`, while
the tokenizer's named EOS resolves to `<|im_end|>` at 248046. The
[bounded text codec](qwen35-text-codec.md) now reproduces the selected upstream
union of both IDs, exact template/IDs and stream behavior. Its controlled phase
baselines pass only under an explicit process profile; default repeated
reconstruction fails the component memory cap. Astra approves this text scope;
controller, deeper-context, request/quality/adoption gates remain open. No metadata
is rewritten. [Checks/review/commit](../artifacts/qwen35-text-codec-2026-11-08/publication-checks.json).
Separate [persistent-codec stress](../artifacts/qwen35-context256-2026-11-09/codec-cache-stress-2/validated-final.json)
now fits the512MiB component target in default/controlled singleton processes on
the recorded corpus. Astra approves this additional singleton scope; repeated construction and
aggregate admission remain open.
The [bounded generation state bridge](qwen35-generation-state.md) is retained:
pending/committed history, conservative hybrid replay and detailed memory/work
snapshots feed the existing controller. Required/source checks and both scoped
release baselines pass; Astra approves this state bridge with no findings. This
grants no managed route.
[Checks/review/commit](../artifacts/qwen35-generation-state-2026-11-09/publication-checks.json).
The closed [128/256-input evaluation](qwen35-context256.md) rejects the4B
trajectory under the original dual-profile gate:13,505failed values across76.98M
comparisons, maxratio34.066. Greedy IDs agree, while3,132checkpoint coordinates
have disjoint idealFP64/stagedFP32 tolerance intervals. No output could pass both
there; neither tolerance nor profile requirement changes. Native state/recovery
controls pass. Astra approves scoped retention of the rejection diagnostics
with no findings; [checks/review/commit](../artifacts/qwen35-context256-2026-11-09/publication-checks.json).
No request, quality or managed-route qualification follows. The separate reference
prototype matches all38.49M originalFP32 values bitwise with closed endpoints and
32damaged-evidence controls; it changes no collector or engine behavior.
The [9B header preflight](../artifacts/qwen35-9b-preflight-2026-11-08/preflight-summary.json)
finds 250 packed matrices with a distinct, untied output head and a 5.038 GB
container. It requires separate head binding, bounded reference-reader support
above 4 GiB and measured cold/steady admission; full source-byte identity is
now verified below.
Header metadata alone does not establish fit or runtime qualification.
The [9B planning estimate](../artifacts/qwen35-9b-preflight-2026-11-08/admission-estimate-2026-11-09.json)
puts current-mechanism numerical ownership near5.54GB and cold upload near5.78GB,
before the changed head metadata, codec and application owners. These are
hypotheses, not admission evidence. Its5.04GB source requires at least7.19GB free
disk, plus other concurrent growth, to preserve the2GiB guard. After reclaiming
3.14GB of verified regenerable compiler cache, all5.038GB and927header tensors
pass [independent byte/header/config/index verification](../artifacts/qwen35-9b-payload-2026-11-09/verified.json).
The [source contract](../artifacts/qwen35-9b-payload-2026-11-09/contract.json) and
[growth audit](../artifacts/qwen35-9b-payload-2026-11-09/growth-audit.json) retain
the2GiB disk guard; recorded start/live headroom covers1GiB of pending capture/
evidence growth. This authenticates publisher bytes only. The numerical parser,
untied-head binding, oracle ceiling, codec/quality/physical fit and request metrics
remain open; staging changes no implementation order or managed route.
The [formula control](../artifacts/qwen35-9b-preflight-2026-11-08/admission-estimate-formula-control.json)
reproduces the existing4B requested-owner ledger; actual9B physical fit stays open.
The [phase admission hypothesis](../artifacts/qwen35-9b-preflight-2026-11-08/phase-admission-plan-2026-11-09.json)
loads weights before constructing the codec so upload temporaries and codec
construction do not overlap. It preserves the512MiB codec allowance and estimates
a6.08GB peak before application/changed-head metadata. This sequence is not yet
implemented or physically qualified; allocator retention still needs measurement.

Q2.A retains a separate [pinned9B cold metadata recipe](qwen35-9b-cold.md)
with distinct head/embedding storage and complete source descriptor coverage.
The4B source/API remains strict. Required652CPU/786Metal, fmt/both strictClippy
pass with actual4B source/corruption controls and14release metadata cohorts;
native59 three-slice/46export and Mac Csmoke checks pass. Astra approves scoped
retention with no findings; [checks/review/commit](../artifacts/qwen35-9b-cold-2026-11-09/publication-checks.json).
No9B payload reader,
numerical head binding, memory admission or managed route follows from this work.

Q2.B retains [separate9B CPU payload ownership](qwen35-9b-storage.md), with
complete source authentication, original affine components and exact promoted
small values before publication. Required661CPU/795Metal and actual independent
927component/750row checks pass at5.053GBpeak in controlled child processes.
Six loading cohorts/60samples pass (9B22.731/22.675s;4B−1.02%/−0.18%); no cache-
residency or speedup claim. Astra approves after a reader timing fix;
[checks/review/commit](../artifacts/qwen35-9b-storage-2026-11-09/publication-checks.json).
Full model/default allocator/GPU/codec/
request/quality/adoption gates stay open and4B eligibility does not expand.

Q2.C's [explicit untied head](qwen35-untied-head.md) preserves tied interfaces,
with a separate diagnostic9B CPU factory. Required667CPU/801Metal, tiny dual-profile
CPU/Metal graphs, native61/Csmokes and both bitwise4B capture scopes pass. Six
release cohorts/60samples show4B+0.13%/−2.43% (no repeated5% regression) and
9BCPU15.480/15.546s at5.31GBpeak. Astra approves scoped retention, with no findings. Full9B oracle,
GPU/default allocator/codec/appfit, request/quality/adoption gates stay open;
4B128/256 rejection remains binding.

Q2.D's [independent9B one-input oracle](qwen35-9b-oracle.md) passes both captures:
33.32Mvalues each, maxratio0.1874, zero failures and matching greedy IDs. Separate
bounded >4GiB reads preserve the strict4B path; native peaks5.311GB/reference530MB.
Required667CPU/801Metal,33Python and19damaged-evidence controls pass. Native61/
engine baseline reuse is explicitly audited for test-only Rust changes; helper
measurements imply no engine gain; hardware/thermal queries were unavailable.
Astra approves scoped retention, no blocking findings. Full9B context/GPU/
default allocator/codec/appfit/request/quality/adoption and4B rejection stay binding.

Q2.E's [consuming9B GPU store and hybrid binding](qwen35-9b-metal.md) preserve
strict4B APIs and numerical kernels. Both real9B captures pass33.32Mvalues each,
maxratio0.1874, with5.667GBpeak and zero retained CPU matrix bytes. Required
667CPU/803Metal,38Python, native62/Csmokes and35damaged-evidence controls pass;
both4B capture scopes stay bit-exact. Eight release cohorts/80samples show9B
scalar15.284/15.366s→hybrid236/239ms (64.72×/64.42×) for raw input1 only;
4B−3.28%/+0.34%. AC/nominal thermal/Low Power Mode off are boundary observations.
Astra approves scoped retention, no findings. Context/default allocator/codec/appfit/request/overhead/
quality/adoption stay open;4B128/256 rejection remains binding.

Q2.F's [9B context17 qualification](qwen35-9b-context17.md) passes both captures:
33.85M comparisons each, zero failures, max ratio 0.4141, identical complete bodies
and 5.667 GB peak. Real nonzero-prefix failure/cancel/poison/repair/retry, partitions
and zero-allocation checks pass. Required 667 CPU / 803 Metal, 46 Python and
25 damaged-evidence controls pass; native62 reuse is audited for test-only Rust
changes. Two release context17 baselines are 4.093 / 4.121 s, full outputs equal,
with nominal AC/thermal/low-power boundary observations. No optimization, request,
tail, sustained or overhead claim. Astra approves scoped retention, no findings. Deeper 9B contexts,
default allocator/codec/appfit/quality/adoption and the 4B 128/256 restriction remain.

Q2.G0's [separate exact recurrence reference](qwen35-reference-vector.md) matches
all67.17M one/context17 FP32/FP64 values bitwise. Scalar sources, frozen fixtures,
default readers and gates stay unchanged. Two matched reversed cohorts lower
complete helper-step time60.02/60.25% FP64 and95.55/95.23% FP32, including
conversion/validation, with108.3MB helper peak. This is a reference component gain;
engine speed, deeper contexts and adoption remain unqualified. All175 portable Qwen tests and evidence controls pass; required Rust/native checks
are reused only after input audits. Astra approves scoped retention with no findings;
[checks/review/commit](../artifacts/qwen35-reference-vector-2026-11-09/publication-checks.json).

Q2.G's [9B128/256 evaluation](qwen35-9b-context256.md) rejects deeper coupled
fidelity:45,332/79.35M comparisons fail per capture, with14,847 disjoint original
profile intervals. Native functional/recovery checks pass at5.667GBpeak;3positive/
39damage controls pass. Both original profiles/gates remain binding. Required
Rust checks,190portable Python tests, native build/ABI and two strict release-probe
builds pass. The first performance attempt rejects its environment before model
startup. Retry2 saves raw preflight before admission and passes12metric/preflight
tests. After power correction,40measured/12warm samples close at nominal boundaries:
128medians32.068/30.848s,256medians68.399/63.330s, exact outputs,5.676GB maximum
conservative charge. First256 cohort is noisy; all samples are retained.1positive/
17performance controls pass. Astra approves scoped diagnostic retention/commit,
no findings. These
are descriptive baselines; numerical rejection, overhead and adoption restrictions remain.

## Metric contract

O1 must make each handled request observable before performance changes. Reuse
[`generation::Report`](../src/generation.rs), completed
[`metal::Timing`](../src/metal/mod.rs), worker events and owner-memory reports.
Use a schema version, request/attempt/session IDs and owner generation. Attach
immutable manifests once: engine/build, device/OS/power mode, artifact/revision/
hashes, model lineage, tokenizer, quantization/KV format, backend/kernel choices,
context/chunk/output capacity, cache state, prompt/token hashes in benchmark
captures, sampling/seed, thinking mode/budget and instrumentation mode.

| Metric fields | Definition / unit | Collection and interpretation |
| --- | --- | --- |
| `request_total_ms`, `native_total_ms` | User submission to terminal UI state; native generation start to terminal worker outcome, respectively | Monotonic host clock; nested intervals, not additive. Record outcome and observed partial duration on errors/cancellation |
| `ttft_event_ms`, `ttft_visible_ms`, `first_answer_visible_ms` | Submission to first emitted token event, first nonempty displayed text, and first displayed final-answer text | Reasoning text may precede the answer; preserve thinking mode. No token/text means null with reason. A callback is not a render acknowledgment |
| `queue_ms`, `tokenization_ms`, `output_decode_ms`, `ui_delivery_ms` | Submission→worker start; formatting/tokenization; token→text decoding; correlated publication→render acknowledgment | Separate boundaries; missing render/phase probes remain null. Never infer a UI or GPU clock from another clock without calibration |
| `load_ms`, `conversion_ms`, `compile_ms`, `ready_ms` | Completed model acquisition/load phases and process start→ready | Separate download from disk load; new process is not proof of cold filesystem cache. Label cache state and component overlap |
| `prefill_ms`, `prefill_tokens_per_s` | Completed prefill wall duration; newly evaluated prompt rows divided by prefill seconds | Exclude reused prefix rows from numerator; report reuse and evaluated rows separately. Fully cached/no-evaluation case is not infinite throughput |
| `decode_ms`, `decode_tokens_per_s` | Completed decode phase duration; `(N-1)/(last_token_time-first_token_time)` for N emitted model-token events >1 | Follow PLAN's formula, excluding first-token prefill from the rate. Same event source and stop policy; N≤1/zero interval gives null. Count thinking/answer tokens separately where tokenizer semantics support it |
| `token_interval_ms`, `ui_update_interval_ms` | Successive model-token event gaps; successive visible update gaps | Distinct distributions; chunked callbacks cannot establish per-token latency. Bounded aggregates/histograms, no unbounded token trace |
| `sampling_ms`, `host_encode_ms`, `gpu_completed_ms`, `host_wait_ms`, `readback_ms` | Existing phase scopes; GPU start→completion on its own clock | Host sampling or cached-ID fetch only; fused GPU selection stays charged to evaluation. GPU execution overlaps host wait; do not sum them as exclusive phases |
| `prompt_tokens`, `reused_prefix_tokens`, `prefill_rows`, `decode_evaluations`, `sampled_tokens`, `emitted_tokens`, `committed_tokens`, `pending_tokens` | Authoritative work/state counts, with prompt/decode scopes documented | Commit only after successful work; no fabricated KV advancement on partial failure. Record cache action/replay reason and replayed rows |
| `weights_owned_bytes`, `kv_allocated_bytes`, `kv_valid_positions`, `scratch_bytes`, `telemetry_bytes`, `app_allowance_bytes` | Unique physical owners and separately labelled admission allowances | Count shared storage once; preserve capacity versus valid state; include hybrid recurrent/convolution state and file-backed PLE explicitly |
| `footprint_current_bytes`, `process_peak_bytes`, `load_peak_bytes`, `memory_pressure` | OS-measured process observations with instrument/scope/time | Lifetime peak is not request peak. Sampled maxima are lower bounds; unavailable isolated load peak is null. Admission estimates are not measured footprint |
| `command_buffers`, `dispatches`, `host_waits`, `readback_bytes`, `buffer_growths`, `pipeline_builds`, `steady_state_allocations` | Existing counters or cheap operation-boundary counters | Mark unsupported counters unavailable; distinguish Rust numerical allocations from native/app allocations. No per-element events or extra GPU waits |
| `terminal_status`, `failed_phase`, `error_category`, `cancel_to_terminal_ms`, `replay_required` | One authoritative outcome for success/Stop/cancel/reject/error; cancellation signal→terminal acknowledgment | Offline success/failure/rejection/cancellation rates use attempted-request denominator. Process death is incomplete, not a fabricated terminal event |
| `telemetry_dropped`, `export_errors`, `collector_peak_bytes`, `summary_overhead_pct` | Record completeness/resource counters; `100*(median_summary/median_disabled-1)` separately for TTFT and total time | Bound both collection and retention; dropping telemetry never changes inference/session state |

Every observation includes availability and instrument/scope; **unknown is not
zero**. Counters and stable enums at request/chunk boundaries stay outside
numerical hot loops. No routine prompts, output text, token arrays, string maps,
per-token JSON or synchronous disk writes. Keep IDs in records, not aggregate
metric labels. Optional energy/J per token and device-bandwidth counters require
an identified, qualified instrument; modelled bytes are not measured DRAM traffic.

## Measurement and acceptance gates

1. **O1 correctness and bounded overhead first.** Check terminal outcomes,
   empty/no-output and single-token cases, pending/committed state, fully cached
   prompts, reset/replay, counter overflow, ring wrap, dropped records and failed
   export. Freeze field units and reconcile counters with authoritative state.
   Maximum 16 KiB collector/request, 1 MiB process ring and 4 KiB serialized
   summary; account for these bytes. No new steady-state numerical allocation,
   GPU command or wait. Qualify ≤1% median TTFT and request-time overhead in paired
   disabled/summary trials; if noise cannot resolve it, collect more evidence or
   keep summary mode opt-in, not declared qualified.
2. **Baseline before optimization, on every selected artifact.** Release builds,
   named Mac and power/thermal conditions; three warmups, ten measured trials,
   raw samples, median and dispersion, then an independent confirmation run.
   Token-matched numerical and user-visible app comparisons are separate tracks.
   Record tokens and wall time together so reduced reasoning output cannot be
   misreported as a faster kernel. Before further UI work, run console engine
   checks and compare with current stable Ollama and vLLM-Metal on this Mac.
   Record exact engine versions, authenticated model/tokenizer identity,
   precision/KV policy, prompt/output work and cache hits. Separate client TTFT
   and request wall time from completed native/server phases, loading and
   tokenization. Use the same weights where supported; label quantized or
   otherwise unmatched configurations separately. Preserve the 6 GiB gate and
   explicit unsupported/admission results. Use MLX/external engines as references,
   not proof that Toklet can execute the same artifact.
3. **Admitted workload matrix.** Start the existing Gemma pilot at its qualified
   context/chunk/reply limits, including a 384-input/128-output case only if its
   actual admission passes. For each new model run 128/512/1,024-input plus
   128-output cases only when admitted; report unsupported cases, never truncate.
   Include empty-session warm runs, exact append, fully cached/pending tokens,
   prefix edits/replay, boundary rejection, Stop, failed execution/output and
   separate new-process loading. Continue sustained evidence for at least ten
   minutes where required. No automatic 128K/256K context expansion.
4. **Useful reports first.** JSONL plus offline table of phase share, TTFT,
   first-visible/final-answer time, prefill/decode rate, request/continuation time,
   memory and outcomes. Report p50 with sample count; ten requests do not qualify
   p95/p99. Collect a separately sized tail study and uncertainty before making
   tail claims. Diagnostic operator capture is opt-in and excluded from acceptance
   timing; capture coverage/overflow is explicit.
5. **Retain one measured improvement.** Declare primary metric and hypothesis
   before changing code; target ≥5% beyond noise and confirm separately, with no
   unexplained required-workload regression. Preserve quality, numerical/session
   semantics and the 6 GiB engine ceiling including load transients. Uncensored
   and quantized variants need their own gates; prior BF16 results do not transfer.
   Record retain/revert decision and exact evidence/commit.
6. **Required implementation checks and review.** Cargo fmt/clippy/tests with
   relevant supported features, changed model/session/GPU checks, Swift/app
   lifecycle checks, real Mac baseline and overhead trials; retain full logs.
   Submit one combined diff/invariants/check-results request to Astra; resolve
   findings before code completion/commit. The original planning commit was
   documentation-only; the O1 implementation has its own linked check/review
   receipt and does not qualify missing native/UI metrics.

Source/metadata inspection and documentation checks are recorded in
[checks](../artifacts/model-observability-plan-2026-11-07/checks.json). No new
model, speed, overhead or memory qualification is claimed by this plan.
