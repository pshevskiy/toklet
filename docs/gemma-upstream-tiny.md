# Pinned upstream BF16 tiny streaming control

The tiny resident and streamed models produce bit-identical BF16 layer outputs
and logits. This validates storage integration around the stock forward pass;
it does not execute or qualify the real checkpoint.

The frozen fixture uses Transformers5.7.0/PyTorch2.9.1, CPU eager attention,
BF16 parameters, eval/inference mode and no generation cache. Its four layers
have local/global dimensions64/128, two query heads, one KV head, window4 and
shared suffix2. Hidden/MLP/PLE widths are64; the shared MLPs are double-wide.
Nine repeated IDs cross the local window, with a separate changed-final-ID
causal control. All506,376 weight bytes/67tensors were frozen before inference.

The complete model skeleton starts on meta. Only root normalization/projection
weights and CPU rotary buffers are materialized. Compact stock scaled embeddings
read requested tied/PLE rows; both embedded inputs and PLE are supplied explicitly,
avoiding a vocabulary-wide reverse lookup. A layer lease loads the stock decoder's
parameters and persistent scalar, preserves the shared-KV dictionary, then releases
its weights. A vocabulary-row head uses the same tied table and the root retains
its original softcap. Numerical operators are unchanged.

Both controls pass exact layer/logit bits, lowest-ID greedy choices, causal
prefix8, head tile4 with tail1, CPU-FP32 rotary and BF16 scaling buffers, and
partial-layer-read cleanup/recovery while retaining the exception. Shared
consumers2→0 and3→1 reuse the original producer tensor identities and bytes.
The streamed harness owns at most one materialized decoder-layer weight set;
the tiny resident reference and frozen raw fixture remain separate test inputs.
This structural bound is not a real-model fit claim.

The owned sealed run retains20 arrays/19,656bytes, with peak RSS383,270,912bytes
and physical footprint281,789,328bytes below6GiB. Source/runtime/input identities,
actual Python/native origins and clean process exit pass. PyTorch's two synthetic
module namespaces receive explicit implementation-file identity checks. Its
generated temporary source is retained by content hash. The initial inventory
failure is preserved; the input weights and equivalence gates are unchanged.
Final Astra review is tracked in the evidence.

Next is the descriptive real four-token upstream BF16 comparison under the
[prospective policy](gemma-original-quality-next.md). Original-source11,
packed27 and long-context4mathematical/8staged failure observations remain open.
No Rust kernel, model/app eligibility, quality gate or tolerance changes follow.
[Evidence](../benchmarks/results/2026-11-04-gemma-upstream-tiny-v1.json).
