# Next model: bounded Granite sparse MoE

5 November 2026. Astra selects Granite3.1-1B-A400M-Instruct to add sparse
feed-forward execution while reusing the reviewed Granite scalar boundaries,
RMSNorm, ordinary RoPE, GQA and SiLU. This document freezes the prospective
arithmetic and work order. Router and selected-expert scalar leaves are complete;
full sparse-MoE model execution is not implemented yet.

The [official pinned config](https://huggingface.co/ibm-granite/granite-3.1-1b-a400m-instruct/blob/0da7a48b0276d500ce5922fd2b33944091fc6c09/config.json)
declares 24 layers, hidden 1024, expert width 512, 32 experts/top 8,
16 query/8 KV heads and a tied vocabulary of 49,155. Granite scales are
embedding 12, attention 0.015625, residual 0.22 and logits divisor 6.

Public discovery returned 45 matching repositories, none tagged MLX. This is a
bounded search result, not proof no conversion exists. Prefer the authenticated
original BF16 artifact and bounded conversion into final resident Half buffers;
creating another quantized artifact is unnecessary for this first mechanism.
Stock MLX can load the original fused expert layout through its existing
sanitizer. The native adapter must validate those fused slices explicitly.

The pinned safetensors header has 218 BF16 tensors: combined expert gate/up
`[32,1024,1024]`, expert output `[32,1024,512]`, router `[32,1024]`, plus shared
attention/norm/embedding tensors. Each expert's gate and up halves occupy the
second axis; they are not two contiguous whole-model tables. Preserve physical
expert strides or verify any one-time packing with an independent reconstruction
oracle. No duplicate full CPU model is retained alongside GPU weights.

The [header/owner screen](../artifacts/granitemoe-discovery-2026-11-05/admission.json)
validates immutable API/config identity, exact tensor names/shapes, contiguous
spans and file length using only eight header-length bytes and 26,384 header
bytes. Weight payload has not been acquired. The conservative ledger includes
Half weights/KV, selected-expert and attention scratch guards, bounded conversion,
driver, tokenizer/text and application allowances. A fitting screen does not
replace exact native admission or observed process peaks.

Astra's native profile is frozen before target implementation:

- Project router scores with the qualified Half projection/store boundary;
  reject nonfinite scores. Rank descending score, then ascending expert ID;
  select exactly eight distinct experts and retain this ranked order.
- Apply selected-only FP32 softmax: subtract maximum, exponentiate, explicitly
  sum in ascending slot order and divide. No full-router probability
  normalization followed by unnormalized selection.
- Keep existing Half expert projection/SiLU boundaries. Multiply outputs by
  FP32 gates, sum in ranked order in FP32, then store Half once before the
  reviewed Granite residual scaling.

MLX `argpartition` ordering is not this native contract. Its SwitchGLU may sort
work when the routing-index count reaches 64, then scatters results back. Compare
routing sets separately from numerical output; preserve any disagreement rather
than fitting tie or reduction order after execution.

Proceed in reviewable tasks, each checked and reviewed by Astra before commit:

1. Independent router/selected-expert fixtures and scalar implementation. Cover
   boundary/all-equal/negative ties, expert-axis permutations, selected-only
   normalization, reduction-order cancellation, nonfinite rejection and batches
   of 7/8/9 tokens across the stock work-sort threshold.
   Scalar stage complete: [router](moe-router-block.md) and
   [selected experts/reduction](moe-expert-blocks.md), with Astra review and checks.
2. Strict source adapter, fused expert slice mapping, bounded BF16 conversion
   and exact owner planning. Reject unsupported source variants before allocation.
3. Resident Metal routing/expert blocks with no host readback or numerical
   hot-loop dynamic dispatch; tiny intermediate, padding and cached-state checks.
4. Pinned stock/native real 512-position execution, chunk/reset/overflow,
   full decision rows and process peaks. Quality, managed Chat, full declared
   context, performance tuning and iPhone remain separate gates.

Avoid a generic MoE framework. Add the concrete router/selected-expert mechanism
with immutable weights and session-owned scratch; generalize only for another
verified consumer.

Astra reviewed the 218-tensor inventory, expert-axis split, ledger and prospective
arithmetic. The screen is 4,332,298,240 bytes, leaving 2,110,152,704 below 6GiB,
conditional on bounded direct conversion. Header inspection does not authenticate
weight content; verify the recorded LFS SHA at acquisition. Local link and
whitespace checks pass; production Rust inputs are unchanged by this task.
