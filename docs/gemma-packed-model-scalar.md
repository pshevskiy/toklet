# Native borrowed packed Gemma reference graph

The [receipt](../benchmarks/results/2026-10-25-gemma-packed-model-scalar-v1.json)
binds the native execution of the frozen
[packed graph reference](gemma-packed-graph-reference.md). This is a bounded
scalar foundation for G3. It reads the synthetic packed bytes directly and does
not load or qualify a real checkpoint, device, app profile or new context limit.

## Layout and ownership

`model::gemma_packed::Model::new` admits only hidden64, vocabulary17, four
alternating local/full layers, Q2/KV1, local head64/rotary64 and full
head128/rotary32, PLE64, MLP64/128, two producer layers and32 total positions.
The public constructor cannot select another geometry. A private layout holds
the derived dimensions, rotary parameters, window and checked scratch counts;
forward code consumes that layout. A separately reviewed real-checkpoint
adapter can later populate the same execution layout after qualification.

Every matrix is a borrowed, contiguous row-major packed4/group64 MLX affine
matrix with BF16 coefficient values. Embedding `[17,64]` doubles as the tied
head. PLE lookup is `[17,256]`, PLE projection `[256,64]`; each layer has concrete
Q/O, FF gate/up/down and PLE gate/projection fields. Norm vectors are direct
FP32 scales. Only the first two layers own K/V projections and K norms.
Missing or unexpected owners, matrix shapes/representation, norm sizes/finite
values and nonfinite scalars reject during construction. No tensor maps,
full expanded matrix or trait-object dispatch occur in forward execution.

The model borrows immutable storage. Each session borrows its exact model and
owns two caches, persistent producer K/V staging and23 scratch buffers. The
local cache is a four-position ring; the full cache admits32 positions. Shared
layers consume the current proposals of their corresponding producers, including
uncommitted tokens in the same chunk. Borrowed proposals keep both staging
owners live until the final all-owner commit. Activations are token-major;
K/V cache storage remains `[kv_heads,capacity,head_dim]`.

## Numerical and publication behavior

Row fetch and projections reconstruct packed coefficients into FP32 values;
projection products accumulate in feature order using FP64 and store FP32.
Activations and KV stay FP32. This follows the explicitly promoted-FP32 source
contract; original-BF16 graph execution remains unqualified. The scalar kernels
preserve learned Q/K norms, unlearned V norm, proportional rotary, scale1 causal
attention, embedding scaling, PLE, GeGLU, norm/residual/scalar order and softcap30.

Public logits, committed input history and every cache advance only after all
layers, final normalization, tied head and softcap succeed. Full cache-batch
validation precedes writes. Failure leaves the prior published prefix and
logits intact; discardable scratch and producer staging may change. Reset
preflights both caches before discarding history/logits and reuses storage.
Sampling, pending sampled tokens, general prefix reuse and cancellation APIs
are later tasks; committed history here contains only successfully input tokens.

## Validation and accounting

Four unit tests compare full prefill, one-token decode and mixed chunks against
both frozen mathematical contracts under unchanged `1e-5 + 1e-4×abs(reference)`
gates. They check selected states in every layer, producer staging, physical
cache slots, public logits/history and stable buffer identities. Seven injected
late boundaries preserve an existing prefix and recover exactly. Empty/invalid
input, capacity exhaustion and actual finite-scalar arithmetic overflow preserve
published state. Model-boundary tests reject wrong matrix representations,
shapes, norms, ownership and scalars.

A separate integration test measures96 warmed decode steps, three resets and
accounting calls with zero Rust allocations. A rejected budget allocates no
session buffers. Logical model bytes count packed payload once for the tied
head, norms, scalar/metadata structures and actual vector capacities. Session
bytes include caches, staging and all scratch capacities; admission checks the
combined requirement before numerical allocation and rechecks actual capacity.
Caller container padding, tokenizer/app state, test-fixture storage and process
memory are outside this numerical count. It is not device-fit evidence.

Required formatting, strict default/Metal Clippy, default/Metal tests and both
iOS compilation targets are recorded in the receipt. Astra reviews source and
exact final evidence before commit. No performance claim is made from these
debug correctness runs.

Actual E2B payload/value admission and geometry, generation/tokenizer/chat
semantics, resident Metal, real numerical/quality/performance measurements,
Mac/iPhone memory admission and app integration remain open. The source packed
GPU graph's15 intermediate discrepancies, earlier dense graph's15 discrepancies
and8 long-rotary discrepancies remain separate, unwaived records.
