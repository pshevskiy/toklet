# Qwen3.5 text candidate: implementation investigation

This is design work for L4/E2/E4, not an implemented graph or selected production
model. Testing has resumed; the independent unmasked gated-delta reference is
checked separately. No downloaded weight payload, new model
admission, long-context numerical result or phone fit is established here.
The [long-reply plan](long-reply-engine-plan.md) remains authoritative.

The already authenticated [architecture inventory](mlx-architectures.md) pins
MLX-LM0d4d3d04. Initial graph inspection read the selected source archive as inert
bytes. The separate gated-delta comparison now executes its authenticated
operator source. Relevant graph sources are [Qwen3.5](https://github.com/ml-explore/mlx-lm/blob/0d4d3d04f8ab7ce4a51e7209e8615755a48ba94d/mlx_lm/models/qwen3_5.py),
[Qwen3 Next attention](https://github.com/ml-explore/mlx-lm/blob/0d4d3d04f8ab7ce4a51e7209e8615755a48ba94d/mlx_lm/models/qwen3_next.py)
and [gated delta](https://github.com/ml-explore/mlx-lm/blob/0d4d3d04f8ab7ce4a51e7209e8615755a48ba94d/mlx_lm/models/gated_delta.py).
Installed Python packages are not substituted for that immutable source identity.

## Concrete missing mechanisms

The official [0.8B config](https://huggingface.co/Qwen/Qwen3.5-0.8B/raw/eb706f593d2d43c90a10271199c10b07ced7569a/config.json)
has hidden1024,24 layers,18 linear and6 full attention, query heads8, KV heads2,
head width256, feed-forward3584 and vocabulary248320. It declares262144 positions.
Full attention uses per-head Q/K normalization and a query projection carrying
both queries and their output gate. That split is within each head, so treating
all queries as one leading half would use the wrong layout. Rotary covers64 of
256 head coordinates. The remaining coordinates must stay unchanged.

Linear layers require causal depthwise convolution, normalized Q/K, scalar decay
and update gates, a persistent FP32 matrix state, gated normalization and output
projection. The update must consume the decayed state before correction/readout.
Its normalization epsilon and readout scale must follow the recurrent reference;
they cannot be inherited from the transformer's Q/K RMSNorm helper.

The wrapper and text graph have different tensor namespaces. Conversion can
change convolution axes and normalization weight conventions, and some source
artifacts include vision or prediction-head weights. Freeze the actual supported
conversion's tensor metadata and reconstruction convention before loading it.
Text-only support must explicitly account for any ignored component and cannot
claim its execution. The tokenizer/template needs independent admission too.

## Small modules and implementation order

No empty module or general tensor framework is added for this investigation.
Introduce each module when its concrete implementation exists:

1. A tiny independent convolution/gated-delta reference with nonzero initial
   state, multiple heads, different key/value widths, chunk/decode equivalence,
   masked positions and terminal state witnesses. Gate saturation, normalization
   epsilon and rounding need deliberate inputs. Include more value than key
   heads and freeze the exact grouping: value headh selects key head
   floor(h/(value_heads/key_heads)), with a checked divisibility requirement.
   Freeze each fixture before its target. The
   [unmasked gated-delta reference](gated-delta-oracle-preparation.md) is now
   frozen with independent FP64 math and passing pinned upstream CPU/GPU
   comparisons. Convolution, masked outputs and graph-level witnesses remain open.
2. Ordinary scalar slice functions for the recurrence and convolution. Explicitly
   document contiguous layouts: activations[tokens,heads,dim], convolution
   history[kernel-1,channels], matrix state[value_heads,value_dim,key_dim].
   Keep immutable weights separate from mutable state/scratch.
   The [normalized scalar recurrence](gated-delta-scalar.md) is checked against
   the frozen unmasked witness. The separate
   [raw Q/K normalizer](gated-delta-normalization.md) also passes its frozen and
   composed-state gates. [Scalar gate construction](gated-delta-gates.md) also
   passes the full raw-input pipeline; causal convolution remains pending;
   no hybrid session is implied.
3. Partial rotary and the attention-gate layout as separate dense mechanisms.
   Preserve existing full-rotary Qwen2/Llama/Mistral/Qwen3 fixtures. The text
   position oracle must establish rotary convention/offset behavior, including
   source configuration fields associated with multimodal positions.
4. A named `mlx/qwen35.rs` graph adapter for bounded nested config, tensor names,
   declared layer schedule, tied output, weight conventions and rejected options.
   Share validated artifact/affine readers where the actual layouts match.
   Compose existing projections and new mechanisms with concrete layer variants.
5. A separate scalar hybrid graph/session, then resident Metal convolution/state
   kernels and graph scheduling. Select layer/backend types at operation
   boundaries; keep per-element loops statically dispatched. Existing affine
   projection kernels are reuse candidates, subject to the new shape gates.
6. Shared load/session plans include each real allocation and temporary overlap.
   Only then consider a larger context ceiling, explicit capabilities, actual
   long-context quality and end-to-end Mac/phone trials.

Masked output and masked state need separate oracle policies. The pinned
operation reference preserves inactive state but still computes its readout;
the masked Metal branch writes zero output. Those paths cannot be assumed
equivalent on inactive outputs. Start with an explicitly unmasked baseline or
name and qualify a selected masked policy; preserve both upstream witnesses and
do not hide the difference by changing tolerances.

The existing `mlx/qwen.rs` reader already understands floating F16/BF16/F32 and
MLX affine metadata. That representation support is a reuse opportunity, not
proof that a hybrid tensor set or conversion has been admitted. Existing scalar
`model::Model` is a dense reference and should not silently acquire recurrent
state fields that every old graph must carry. A second concrete graph justifies
shared operators; it does not justify a universal graph runtime.

## State validity and cancellation

A valid hybrid prefix includes convolution and recurrent state for exactly the
committed numerical positions as well as the full-attention KV prefix. Sampled
pending output remains separate. Reject prefix reuse when either representation
is stale or model/position semantics differ. A fully cached prompt and an
uncommitted sampled token need explicit fixtures.

A failed multi-layer step must retain a valid previous prefix or mark the state
as requiring replay. A naive in-place recurrent write before later layers can
succeed is insufficient. Evaluate tentative buffers versus explicit invalidation
and replay after failure; include whichever extra allocations are retained in
admission. A cancellation observed between complete steps must preserve the
last committed state. Do not copy a whole long cache for every generated token.

## Memory arithmetic to qualify later

These are derivations for proposed F16 full-attention KV and FP32 recurrent state,
not engine allocation reports or device budgets. For capacityC, six full layers
would have6×2(K/V)×2(KV heads)×C×256×2 bytes of KV. AtC131072 that is1.5GiB;
atC262144 it is3GiB. Eighteen recurrent matrices at16×128×128×4 bytes add18MiB.
This excludes weights, convolution history, scratch, padding/tables, tentative
state, tokenizer, response buffers, UI history and loading/conversion overlap.

A131072-token reply needs additional positions for prompt/template/history.
No capacity or phone process allowance follows from these equations. Inspect the
immutable candidate's complete headers and dtype exclusions, then compare actual
plans/owners and measured memory separately on each device. Retain the existing
Mac budget and qualify a mobile limit independently.

The named missing mechanisms above are mapped for implementation. The unmasked
gated-delta reference is checked and frozen before its native target. The other
operator/graph oracles, scalar/Metal execution, real artifact selection, text
integration and every long-context/device gate remain open. An unavailable
required check cannot grant completion.
