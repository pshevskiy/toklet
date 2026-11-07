# Console engine comparison on Apple M1 Pro

Ollama 0.40.15 and vLLM 0.33.0 / Metal 0.33.0 are installed and tested.
Two independent short cohorts per engine pass: three warmups and ten measured
requests each, with raw samples and median/IQR retained. All78 requests return
`The capital of France is **Paris**.` from the 15-token prompt. This is a
reference baseline with different weights, precision, caching and timing boundaries;
it does not establish an engine speedup factor or model adoption quality.

| Engine / weights | Request median, runs 1 / 2 | First-token/content median, runs 1 / 2 | Cached prompt | Peak bytes, largest run |
| --- | --- | --- | --- | --- |
| Toklet, official BF16/G2 | 2.184 / 2.202 s | 392 / 395 ms | 0 / 15 | 4,919,169,320 |
| Ollama 0.40.15, uncensored MLX 4bit | 121 / 119 ms | 20.9 / 20.7 ms | 14 / 15 | 3,852,391,488 |
| vLLM-Metal 0.33.0, same uncensored MLX 4bit | 147 / 149 ms | 50.7 / 51.5 ms | 0 / 15 | 5,921,790,960 |

Toklet times generation from prepared token IDs; its first-token event precedes
text decoding, and request time excludes final detokenization/console printing.
Ollama times the HTTP client through first nonempty content and stream completion,
including tokenization/detokenization. vLLM uses its reported frontend first-token
duration and the blocking `LLM.generate` call on prepared IDs. No subtraction mixes
vLLM wall-clock arrival with monotonic engine timestamps. UI visibility is unobserved.

Toklet performs 8 completed decode evaluations after its initial sample, giving
4.48 / 4.44 evaluations/s. vLLM generates the identical 8 visible token IDs plus
stop 106; its 8 post-first tokens give80.66 / 80.88 tokens/s. Ollama reports 8 emitted
tokens and71.39 / 72.48 API eval tokens/s; actual sampled IDs/hidden stop accounting
are unavailable. These rates retain their engine-specific denominators.

Each external request records AC power and nominal thermal state before/after.
Engines run serially after setup; confirmation cohorts reload the engine process.
Ollama cohort 1 shares the resident model/control process; cohort 2 reloads it.
vLLM control and both cohorts each use a separate engine process.
External memory is a conservative sum of owned process footprints and can count
shared pages twice; native memory is a single-process lifetime peak. The sum of observed
per-process lifetime peaks also stays below 6 GiB: at most 3,899,184,048 bytes for
Ollama and 5,948,005,408 for vLLM. Different accounting prevents an exact memory
ratio. The 250 ms watchdog observes live/discovered processes; short-lived children
or final peaks can be missed. It provides sampled guard coverage, not a continuous
allocator hard cap. vLLM has an explicit 64 MiB KV allocation; observed external KV dtypes and
committed/pending journal states remain unknown. Toklet verifies 24 history,
23 committed and 1 pending token, cancellation and admission preserving state.

## Identity, admission and installation

Toklet uses authenticated `google/gemma-4-E2B-it` revision
`70af34e20bd4b7a91f0de6b22675850c43922a03`, BF16/G2, context 512/chunk 32,
with uncommitted O3 source pinned in its cohort manifests; profiling is disabled.
The original BF16 external configurations both fail the tested 6 GiB guard before
inference. The vLLM planner also reports 9.26 GB model storage plus 0.71 GB overhead.
Those failures yield no speed result or assertion about every possible configuration.

Per user choice, both external references instead use
[`deadbydawn101/gemma-4-E2B-Heretic-Uncensored-mlx-4bit`](https://huggingface.co/deadbydawn101/gemma-4-E2B-Heretic-Uncensored-mlx-4bit/tree/cde01efcb11f8a12e88a953b752fc16062aa1486),
revision `cde01efcb11f8a12e88a953b752fc16062aa1486`, affine4/group64. All nine files,
2,649 tensors and packing ranges pass bounded validation. The 3,581,100,878-byte
weight file SHA256 is `d4a2995ce3fdafbbd4a3376bc8524a8d26bcbf0f51197701b4e21ab27faa37e1`.
Tokenizer bytes match the official snapshot; weights are a modified heretic
checkpoint, so one exact response is no quality/adoption qualification.

Toklet rejects this checkpoint's format: extra inactive MoE metadata, redundant
shared-layer K/V tensors and BF16 per-layer projection differ from its qualified
native affine4 representation. A narrow metadata experiment passed its two unit
checks but exposed the next layout rejection; it was reverted. Strict production
loading and numerical gates are unchanged. Native support needs separate validation.

The [official Ollama release](https://github.com/ollama/ollama/releases/tag/v0.40.15)
archive checksum, matching Team 3MU9H2V9Y9 signature and Gatekeeper notarization pass.
The previous app is backed up in the setup evidence. vLLM follows the
[official Metal installation](https://docs.vllm.ai/projects/vllm-metal/en/latest/installation/),
using a repository-local environment with MLX 0.32.4 and stable Torch 2.14.0;
`pip check`, MetalPlatform and a real MLX GPU arithmetic control pass. The normal
core wheel version `0.33.0+cpu` does not describe the selected MLX GPU backend.
The experimental MPS nightly dependency override conflicted with core requirements;
its logs remain, and the default MLX environment was restored to a clean dependency set.

## Evidence and next work

[Full report / dispersion / input hashes](../artifacts/engine-comparison-2026-11-07/console-comparison-report.json),
[offline checks](../artifacts/engine-comparison-2026-11-07/report-checks.json),
[installation/research](../artifacts/engine-comparison-2026-11-07/research.json),
[bitwise Ollama import provenance](../artifacts/engine-comparison-2026-11-07/ollama/import-provenance.json),
[native raw cohorts](../artifacts/console-engine-2026-11-07/toklet-console-report.json).
This documentation/report task changes no production engine source; Cargo tests
are not rerun for publication. Native required checks are in the pinned cohort
records; O3's later changes still require their own final checks and Astra review.

Next: finish O3's bounded profile, identify actual dispatch/copy/wait costs, then
measure one optimization under the unchanged retain gate. The reference gap
motivates that investigation; it does not isolate its cause. Longer contexts,
longer outputs, sustained/tail performance, model quality, matched-precision
ranking and summary/profile overhead remain separate qualifications.
