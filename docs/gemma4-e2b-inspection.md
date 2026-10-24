# Gemma4 E2B text inspection

G1 inspection, 24 October2026. This maps the first concrete Gemma4 candidate;
it does not implement its graph or admit loading/generation. The
[priority plan](gemma4-implementation-plan.md) remains authoritative. The next
code task is the scalar tanh-GELU/GeGLU target after its separately
[frozen independent witness](gemma-gelu-reference.md).

## Immutable inputs and scope

- Official `google/gemma-4-E2B-it` config revision
  `3e22461f65e89153144f8adb70e3b8c2cc9845a7`, SHA256
  `1b28f3d2c3100f6c594754b81107428bd7b822a7f48272ca681dae9d2ec38330`.
- Converted `mlx-community/gemma-4-e2b-it-4bit` revision
  `238767527555cb75a05732a84dff5d6ba0dd6809`, config SHA256
  `6397cb6eca41b911d1dcab74e17941351057bd759284052a2331918ff6f9246c`.
- MLX-LM `0d4d3d04f8ab7ce4a51e7209e8615755a48ba94d`: inert wrapper,
  text model, positional helpers, cache and attention/mask source. Actual installed
  MLX0.32.3 norm/activation definitions are captured from the already admitted
  runtime, without numerical imports or model execution.

Nine successful owned HTTPS commands capture the immutable inventory, six config/
tokenizer/index/template files and exact ranges for an eight-byte container
length plus 318220-byte header. No tensor payload is downloaded. Git/LFS identities
of full metadata files are checked; a range hash is not a verified whole-shard
hash. The declared shard is 3550670554 bytes, LFS SHA256
`038e39a37a7667373d2c3991375446b10c96ae1d717a68674870343db376b76e`.
Its full payload and finite/quantized values remain unverified.

The [receipt](../benchmarks/results/2026-10-24-gemma4-inspection-v1.json) binds
source, captures, independent structural replay and retained failed attempts.
The first attempt rejected a redirect larger than a small metadata file. A
premature retry was corrected by stopping the obsolete owned acquisition; another
attempt failed HTTP/2 framing. The final serial retry keeps curl configuration
disabled, uses HTTP/1.1, bounds redirect transport separately and preserves exact
final sizes, status, ranges and hashes. No failed attempt contributes acceptance.

## Graph mechanisms to implement

The official and converted nested text configurations match. The inspected text
variant has hidden1536,35 layers, vocabulary262144,8 query heads,1 KV head and
shared K/V on the final20 layers. It is dense, tied-output, bias-free, causal text;
MoE and K=V variants are disabled. The multimodal wrapper and its tensor namespace
must be admitted explicitly; initial support will execute text only.

| Mechanism | Exact inspected behavior | Native work |
| --- | --- | --- |
| Embedding | Token embedding scaled by sqrt(1536); tied final projection | Reuse qualified packed gather/linear only after new-shape gates |
| Layer schedule | Repeated four sliding layers then one full layer, seven times | Distinct validated local/full layer descriptions |
| Attention geometry | Local head256, full head512,8 query/1 KV heads; attention scale1.0 | Independent geometry/numerical witnesses; do not inherit Qwen scaling |
| Norms | Direct learned mean-RMS scales on Q/K and decoder paths; value RMS has no learned scale | Independent order/epsilon/weight-convention checks |
| RoPE | Local default base10000; global proportional base1000000 with `partial_rotary_factor=0.25` (frequency `factor=1.0`) | The proportional helper builds frequencies using full head512 denominator and infinite inactive frequencies; freeze exact paired-coordinate/offset behavior, not a guessed contiguous partial rotation |
| Sliding cache | Window512 with no retained prefix; local causal masks and absolute offsets | Bounded rotating state, wrapped reads and chunk/prefix/cancellation fixtures |
| Shared KV | First15 layers own cache; subsequent local/full layers reuse producer13/14 respectively | One owner per producer, explicit read-only consumer mapping; no allocation/update per shared consumer |
| Feed-forward | Tanh-GELU GeGLU; intermediate6144 for first15,12288 for final20 | New independent GELU/GeGLU oracle and scalar/Metal operations |
| Residual/norm order | Input norm→attention→post-attention norm→residual; pre-feed-forward norm→MLP→post-feed-forward norm→residual | Explicit Gemma4 layer graph and stage witnesses |
| Per-layer inputs | Embedding `[262144,35*256]`, scale16; projection of scaled hidden→35×256 with hidden^-0.5 scale and width256 RMS; sum with embedding and scale1/sqrt(2) | Dedicated gather/projection/layout/scale witnesses and bounded scratch |
| Per-layer gating | Project hidden to256→tanh-GELU→multiply per-layer input→project to1536→post norm→residual | Exact order/weight mapping; no SiLU substitution |
| Layer/final output | Learned scalar after every decoder layer, final mean-RMS norm, tied logits, tanh soft cap30 | Scalar broadcast and logit capping witnesses |

Defaults and absent options need explicit validation: no projection biases,
attention dropout0, cache enabled, causal rather than bidirectional execution,
known local/full schedule, no experts, declared direct norm convention and exact
quantization metadata. Different depth/head/PLE/KV-sharing/rope/MoE options remain
unimplemented until independently described and tested. Image, audio, video,
clipped modality linears and alternate input-embedding matching are outside the
first text graph; their weights must be recognized as excluded components.

## Tensor and representation mapping

The complete header and index agree on2511 tensors. Span arithmetic, dtype widths,
ordered gap-free payload coverage and exact index-to-shard names pass structural
replay. The text component is exactly1096 tensors:278 U32 packed matrices,
556 BF16 affine scale/bias arrays and262 BF16 learned vectors/scalars. These map
to540 logical tensors from the inspected graph, including both MLP widths.
All text affine matrices use declared4-bit/group64 packing: physical weight
`[out,in/8]`, scale/bias `[out,in/64]`. Value norm has no missing learned tensor;
shared consumers have no private K/V projections or K norm. No extra text tensor
is silently ignored. Layout metadata does not qualify reconstruction values.

| Stored component | Header payload bytes |
| --- | ---: |
| Text | 2603981382 |
| Audio tower | 609650176 |
| Vision tower | 334730112 |
| Audio embedding/projector | 1327104 |
| Vision embedding/projector | 663552 |

The per-layer token embedding alone uses1321205760 stored bytes, including its
affine metadata. “E2B” is not the full stored parameter/memory count. Text-only
loading must avoid resident modality allocations and redundant full CPU/GPU
copies, while acquisition storage and load conversion peaks still count.

## Tokenizer, chat and stops

The32169626-byte tokenizer has262144 BPE vocabulary entries,514906 merges and24
added special tokens. Its normalizer replaces spaces with `▁`; decoding applies
`Replace(▁→space)`, ByteFallback and Fuse, with no initial-space Strip. Existing
Mistral decoding/template semantics cannot be inherited wholesale. The streamed
fallback-run mechanism may be reusable only after Gemma-specific fixtures.

The captured17336-byte Jinja template supports system/developer, model turns,
thinking channels, tools and multimodal content. Freeze a bounded basic-text
subset with explicit unsupported-option rejection and independent prompt/roundtrip
witnesses before Chat admission; do not execute arbitrary repository code.
The official wrapper stops are1/106, converted wrapper/generation stops are
1/106/50, while the nested text config names1. Resolve and test the actual selected
profile's semantics. Thinking/turn/tool markers, hidden terminal tokens, whitespace,
Unicode/byte fallback, literal controls and multi-turn history need their own
checks. The tokenizer's enormous placeholder model_max_length is not a usable
context capability; use validated position semantics and allocated capacity.

## Memory/context findings and handoff

These are stored-byte and proposed-layout derivations, not an allocator plan,
reservation, measured process peak or device admission. Text storage alone is
about2.425GiB. With FP16 K/V,3 full producers at head512 need6144 bytes per total
position;12 local producers at window512/head256 need6291456 bytes. At capacity
131072 that totals811597824 bytes (774MiB). Stored text plus these two cache terms
is3415579206 bytes, about3.181GiB, before alignment, scratch, staging, tokenizer,
UI/history, reply buffers and load overlap. No phone fit follows. KV compression
or other layouts require independent numerical/quality and memory qualification.

The declared131072 total positions cannot hold131072 output tokens and a nonempty
prompt. No context extension or different long-context checkpoint is admitted by
this inspection. Gemma4 shorter replies and the separate long-reply route have
distinct acceptance gates. Current native eligibility,256/128 output caps and512
Chat positions remain unchanged. G2–G7, real payload integrity, native scalar/
Metal execution, app/tokenizer/quality and physical fit remain open. Deliver the
full [handoff contract](gemma4-implementation-plan.md#mobile-handoff-contract)
before installing a new candidate on the phone.
