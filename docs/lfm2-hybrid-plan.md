# B4 candidate: a shared short-convolution state block

29 October2026. This is the next mechanism after B3's bounded Gemma3 checks,
not an executable-model admission. Astra's cached primary-source research
identifies [mlx-community/LFM2-350M-4bit](https://huggingface.co/mlx-community/LFM2-350M-4bit/tree/18dc72abf3b2337f9123cfd6eeeb58dfa7947066)
revision `18dc72abf3b2337f9123cfd6eeeb58dfa7947066` as a practical representative
for recurrent/hybrid state. No tensor payload has been downloaded or executed.

## Why this mechanism

The pinned configuration has 16 layers: full attention at2/5/8/10/12/14 and
ten length3 depthwise short-convolution layers. Hidden width1024, attention
heads16/KV8/head64, actual FF width4608 and tied vocabulary65536 fit a compact
model. Projection,
ordinary split-half RoPE, Q/K RMS, SwiGLU, residual and full KV mechanisms have
existing counterparts. The new mechanism is a bounded convolution history that
must publish atomically with all attention KV and token state.

The authenticated header inventory has93 Q4/group64/BF16 affine matrices,
45 direct-format norm tensors and ten BF16 convolution tensors `[1024,3,1]`:
334 physical descriptors/148 logical owners. “Direct-format” describes the
stored tensor, not a qualified mathematical gamma interpretation. The advertised
weight file is199,527,820 bytes, payload199,491,072 bytes, LFS SHA256
`558353ad098c1230a8e2ea508cd34ef311a1b97a2bc4f47498f4df8229dc5c0f`.
It still needs full-file verification.

The source planning ledger is4,519,922,060 bytes and native cold ledger
3,552,479,116 bytes, each below the existing6GiB ceiling. These include explicit
conversion/framework allowances and are estimates, not observed process fit.
The declared128000 positions do not imply an initial implemented/qualified
window. Propose a bounded512-position Mac vertical; freeze its exact inputs and
limits before execution.

## Small reusable block and ownership

Resolve the fused input projection into checked B/C/x row spans at loading.
For each channel, the source mechanism computes `u[t] = B[t] * x[t]`, then
`z[t] = w[0]*u[t-2] + w[1]*u[t-1] + w[2]*u[t]`, and projects `C[t] * z[t]`.
Initial missing history is zero. Preserve the primary source's exact coefficient
orientation, operation/store precision and mask semantics in independent
fixtures before adopting this expression as a numerical contract.

Keep immutable convolution weights with the model. Session-owned history stores
the two preceding u rows per channel: `[history_row, hidden]`, contiguous across
hidden channels. Use separate committed/candidate banks and reuse bounded chunk
scratch. The ten layers require40,960 Half bytes per bank before guards and
alignment; FP32 would require81,920. Choose and qualify precision explicitly.
Reserve this storage once; avoid per-token allocation or whole-context copies.

A successful full step publishes candidate convolution state, attention KV and
committed positions together. Any failure/cancellation must retain the previous
valid prefix or require explicit replay. A convolution-layer success alone must
not advance the session. Reset, cached-prefix reuse, pending sampled tokens and
model/settings changes obey the same session identity rules as KV.

Use concrete functions over slices and a small layer-mechanism enum selected
outside numerical loops. Keep scalar reference and resident backend leaves
separate. Extend existing projection/norm/attention blocks only where actual
shapes require it; do not add a second graph framework or model-name fallback.

## Ordered tasks and acceptance

| Task | Delivery | Required evidence |
| --- | --- | --- |
| B4a | Strict config/header/source plan and text inventory | Original row ownership, convolution orientation, actual FF width, direct gamma provenance, unsupported-field rejection, exact memory equations/overflow controls |
| B4b | Scalar short-convolution operation and transactional history | Independent initial/one-/two-/three-row and multi-chunk fixtures; coefficient orientation, mask/tail, exact prefix/reset, failure/retry/publication controls, steady-state allocations |
| B4c | Tiny hybrid graph using shared projections/norm/attention/MLP | Independent complete stages and physical state across mixed attention/convolution layers; whole-step rollback, causal and partition checks |
| B4d | Resident backend block and hybrid composition | Original resident weights, persistent history/scratch, independent leaf/tiny comparisons, guards/status/recovery, physical-owner accounting and completed-work measurements |
| B4e | Acquire and qualify the fitting real representative | Immutable full-file hashes, fixed source/native/text protocol, bounded512 generation/session comparisons and measured process fit, independent Astra review |

Each code task gets relevant ordinary checks, Astra review, progress/improvement
updates and a commit. App admission/context controls follow through B5; original
unquantized10k/100 adoption quality remains M8 closure work. No need to qualify
128000 positions or add MoE/MLA before delivering this concrete mechanism.

## Known source and text questions

The card declares mlx-lm0.26.0, but its tag
`9ee2b7358f5e258af7b31a8561acfbbe56ad5085` lacks `mlx_lm/models/lfm2.py`.
The exact conversion runtime therefore remains unknown. The inspected
[MLX-LM implementation](https://github.com/ml-explore/mlx-lm/blob/1f7057ca530f8a594d4aab50bac41a2ed5e5c5c7/mlx_lm/models/lfm2.py)
is pinned separately; do not label it the conversion revision or infer norm
semantics from BF16 dtype. Choose an explicit current pinned-loader same-artifact
profile after authenticating its complete load/sanitize/quantization path,
`nn.RMSNorm` implementation and actual payload/header. Its direct-weight norm
semantics can be independently corroborated by the inspected Transformers
implementation. Historical converter identity may remain unknown; it does not
require recreating that converter to qualify the explicitly declared profile.
Resolve this interpretation before exporting a real numerical reference.

The tokenizer maps64,400 IDs while the head has65,536 rows; unmapped IDs must
have explicit behavior. BOS1 is present in both template and tokenizer
postprocessing, so prevent duplication. The declared stop union is7; token2
is not automatically a stop. Template rendering, decoding and chat support need
their own fixtures, outside the numerical block. No arbitrary template execution
or silent vocabulary clipping is authorized by successful graph loading.

Cached research lives under `artifacts/new-goal-b4-lfm2-research-v1`, including
primary source files, immutable metadata URLs, header prefix and
`metadata-audit.json`. Its full-file identity and physical fit fields explicitly
remain unqualified.

Metadata audit SHA256:
`bc1cba0789d34e5a73dead22f24c9dcd0532d4c0fe5c4973cb3b312a6861a4cc`;
inspected MLX-LM model source SHA256:
`558dcbec61442b1f7216c0122d08d44083ea71d1510be23f4f934e62b0f71376`.
