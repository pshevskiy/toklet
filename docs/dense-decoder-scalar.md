# Scalar dense decoder finish

The standalone FP32 reference now composes the
[frozen dense decoder order](gemma-decoder-order-reference.md) from immutable
hidden state and an externally computed attention result. Input normalization
uses the existing RMS function before attention; this new function begins at
post-attention normalization. Neither entry owns or mutates KV/session state.
The [acceptance receipt](../benchmarks/results/2026-10-24-dense-decoder-scalar-v1.json)
binds source, fixtures, check logs and final evidence.

## Shapes, ownership and numerical behavior

`DenseDecoderShape` gives flattened independent token rows, hidden width H,
FFN width I and positive finite epsilon. `DenseDecoderWeights` borrows direct
learned `[H]` post-attention/pre-FF/post-FF RMS scales and row-major FFN gate/up/down
weights `[I,H]`, `[I,H]`, `[H,I]`. An optional complete `PerLayerGateWeights`
borrows `[P,H]` gate, `[H,P]` projection and `[H]` post-norm weights. An optional
finite scalar is applied after the last residual. These are dense FP32 weights;
packed quantized execution needs a separate integration that avoids expanding or
retaining this reference weight set.

Hidden and supplied attention result are `[tokens,H]`; PLE input, when supplied,
is already composed `[tokens,P]`. PLE weights without an input skip the gate.
Input without PLE weights rejects. Partially missing modules are unrepresentable;
a checkpoint adapter must reject incomplete inventories. The frozen source's
partial-module call guards compare against the coherent disabled native branch
without granting adapter support for those states.

Caller-owned scratch has two `[tokens,H]` buffers, separate gate and product
buffers each `[tokens,max(I,P)]`, and an up buffer `[tokens,I]`. Without PLE,
max(I,P) is I. With PLE weights but absent input, that P capacity is still
required and validated. Output is `[tokens,H]` and holds the evolving residual.
All buffers use ordinary FP32 alignment. The function allocates no storage,
clones no weights and has no dynamic numerical dispatch. Input slices remain
immutable; callers can reuse the same storage for later independent chunks.

Post-attention RMS precedes the hidden residual. The FFN pre-normalizes that sum,
projects gate/up, applies tanh GELU and product, projects down, post-normalizes,
then adds the saved residual. A present PLE branch gates this updated sum,
applies tanh GELU and the supplied PLE product, projects to H, post-normalizes,
and adds its saved residual. The final scalar multiplies the whole sum.
Existing FP64-product/accumulation linear, FP64 RMS and tanh-GELU kernels round
to FP32 at operator boundaries. Residuals, products and scalar multiply remain
separate checked FP32 operations; no algebraic cancellation or reassociation is
introduced.

All sizes, lengths, epsilon, scalar and finite weights/inputs are admitted before
any output/scratch write, including PLE weights skipped for missing input.
Zero dimensions reject; empty sequences make no call. Checked size arithmetic
precedes allocation at the caller. Numerical overflow can leave partial scratch
and output; discard all of them and replay from immutable inputs. This function
has no model/session commit or rollback role.

## Verification and limits

Six methods authenticate the accepted fixture and compare input normalization,
finish output and final scratch stages for all11 nonempty cases against both
ideal and staged contracts. They also check immutable inputs, repeated exact
chunks and stable buffer pointers, each late invalid operand/scalar/scratch,
missing-input/disabled branches, final scalar absence/signed zero, invalid epsilon
and size/zero boundaries. Seven overflow scenarios prove successful prerequisites
and the exact intended failure at attention norm/residual, FF gate/product/down,
PLE product and final scalar; fresh replay recovers the clean result.

Formatting, strict default/Metal Clippy,257 CPU tests,279 Metal-feature tests
and iOS/simulator library compile checks pass. Feature suites do not execute the
ignored GPU/device tests. The implementation is
[here](../src/cpu/dense_decoder.rs), with
[regressions](../src/cpu/dense_decoder/tests.rs).

This is a dense FP32 composition reference, not a complete Gemma decoder,
quantized/BF16 checkpoint, Metal graph, tokenizer/session, device fit, quality,
performance or long-reply qualification. Constructor/config selection and the
[eight upstream rotary discrepancies](gemma-rotary-precision.md) remain open.
No ABI, protected binary, app capability/limit or installed package changes.
