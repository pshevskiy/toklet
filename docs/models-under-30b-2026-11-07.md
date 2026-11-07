# Current models under 30B: research shortlist

Researched 7 November 2026. Scope: downloadable language/chat models with fewer
than 30 billion **total** parameters, including multimodal models. Rank is a
judgment for Toklet combining capability, coding/tool use and local efficiency;
it is not a common-harness leaderboard. Prefer current official checkpoints over
superseded versions and unqualified community fine-tunes. Benchmarks below are
vendor reports with different harnesses, reasoning budgets and precision.

| Rank | Official model / primary source | Approximate total parameters | Reason to evaluate |
| --- | --- | --- | --- |
| 1 | [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | 27B language model; full artifact rounds to 28B | Strongest overall candidate in this survey: reported GPQA Diamond 89.2 and LiveCodeBench v6 90.3; coding, tools and vision. |
| 2 | [Gemma 4 26B-A4B IT](https://huggingface.co/google/gemma-4-26B-A4B-it) | 25.2B language + about 0.55B vision; 3.8B active language parameters | Strong quality with sparse execution; reported GPQA Diamond 82.3 and AIME 2026 88.3. Full expert weights still require storage/memory. |
| 3 | [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | 9B language; full artifact rounds to 10B | My quality/memory balance pick: reasoning and vision, reported GPQA Diamond 81.7. |
| 4 | [Gemma 4 12B IT](https://huggingface.co/google/gemma-4-12B-it) | 11.95B | Unified text/image/audio model without separate modality encoders; reported GPQA Diamond 78.8. |
| 5 | [Devstral Small 2 24B](https://huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512) | About 24B | Coding-agent specialist; reported SWE-bench Verified 68.0. Different benchmark from Qwen's SWE-bench Pro. |
| 6 | [LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B) | 24B; 2.45B active | Promising performance-first agent candidate; hybrid convolution/attention MoE and official MLX 4-bit artifact. Apple speed claims need local reproduction. |
| 7 | [Ministral 3 14B Reasoning](https://huggingface.co/mistralai/Ministral-3-14B-Reasoning-2512) | 13.5B language + 0.4B vision | Math/STEM and visual reasoning; reported AIME25 85.0 and GPQA Diamond 71.2. |
| 8 | [gpt-oss-20b](https://developers.openai.com/api/docs/models/gpt-oss-20b) | About 21B; 3.6B active | Text reasoning and tools with configurable effort. Official OpenAI documentation confirms size and capabilities. |
| 9 | [Granite 4.2 8B](https://huggingface.co/ibm-granite/granite-4.2-8b) | About 8B | Dense transformer with reasoning, low-effort and non-thinking modes; attractive small tool-use baseline. |
| 10 | [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | 4B language; full artifact rounds to 5B | Compact reasoning/vision candidate; reported GPQA Diamond 76.2. Best starting point here for tight memory. |

Parameter names are rounded. Qwen's full artifact counts differ from language
model names. Active parameters determine only part of execution cost; they do
not replace total weights in memory planning. The Gemma E2B/E4B labels similarly
mean effective parameters, not total size. Models advertised at 30B or above
are outside this shortlist; none is admitted solely through an active count.

## Implications for Toklet

The subsequent [selected model/metric plan](model-observability-plan.md) takes
priority over this survey ranking: Gemma 4 E2B first, then Qwen3.5-4B and 9B,
with observability before optimization and a preference for qualified uncensored
MLX artifacts. The suggestions below remain research context.

- Quality target for a larger machine: Qwen3.8-27B.
- Performance-first MoE experiments: Gemma 4 26B-A4B and LFM2.5-24B-A2B.
- First new candidates for the current Mac: Qwen3.5-4B, then Qwen3.5-9B;
  Granite 4.2 8B is an alternative for tool use. These require implementation
  inspection, exact artifact admission and independent numerical/quality gates.
- The [plan](../PLAN.md#validation-and-performance-policy) retains a 6 GiB engine
  working-set ceiling. Ideal four-bit weights alone require about 12-14 GB for
  24-28B models, before scales, higher-precision tensors, KV, scratch and loading
  transients. Even a nominal 9B language model needs careful admission.
- No Toklet execution, latency or quality qualification is implied by this
  research. Existing qualification remains in [model coverage](model-coverage.md).
  Current O1 status is in the [pilot receipt](o1-gemma-request-summaries.md).

Also inspected recent specialist releases, including
[Nemotron 3.5 Bolt E4B](https://research.nvidia.com/labs/nemotron/Nemotron-3.5-Bolt-E4B-4GB/)
and [LFM3](https://www.liquid.ai/blog/lfm3). Their conversational/edge focus is
interesting, but does not establish a higher general capability ranking. NVIDIA
device timing is not evidence of Apple Silicon timing.

Documentation validation: [check record](../artifacts/models-under-30b-2026-11-07/checks.json).
