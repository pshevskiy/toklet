# M3 public CPU/ANE feasibility conclusion

The bounded public-runtime experiment ends with supported CPU slice execution
and inconclusive ANE acceleration. Retain the reproducible probe and its
independent fixtures; adopt no Core ML/ANE engine backend. Continue the engine
through the complete Metal path after CPU candidate qualification. This closes
M3.2/M3.3's research outcome, not complete stateful ANE generation.

The accepted [Core ML slice report](2026-10-10-coreml-slices.md) already retains
the exact graphs, conversion/load costs, runtime policies, native records,
placement observations, memory scope and unchanged numerical gates. Its
[driver](../../tools/experiments/coreml_probe.py) and
[frozen oracle](../../tests/fixtures/coreml-slice-v1/manifest.json) implement
the permitted supported-slice route in PLAN.md. No further experiment or
reconstruction is needed for this conclusion.

Four graphs cover FP32/FP16 biased projection and causal grouped-query attention
with persistent FP16 K/V, including explicit FP32 attention arithmetic. Both
`CPU_ONLY` and `CPU_AND_NE` are measured. FP32 projection and both attention
graphs pass the frozen five-case gate with three warmups and ten measured
sequences per case. Attention qualifies independent interleaved state, offsets,
all 32 fixture positions, exact cache readback, poisoned future suffix and
causal masking. Projection uses `[1,32]` activations with row-major `[32,32]`
weights; separate K/V state is `[2,32,8]` with contiguous head dimensions.

FP16 projection fails the first warmup under both policies: maximum absolute
error 0.0018783807754516602 and frozen tolerance fraction 1.5727969659617609.
Retain both failures and reject that graph under the original gate. Conversion
success does not qualify the numerical result, and no tolerance changes follow.

Every nonconstant operation's public compute plan prefers CPU. Some operations
list Neural Engine support, but actual ANE execution is unobserved. CPU-only
execution follows its explicit policy. Policy differences in prediction timing
cannot establish acceleration without actual placement and a complete decoder.
The recorded environment lacks full Xcode/Instruments execution tracing;
this conclusion describes that experiment's availability, not a new inventory
of tools installed later.

Own-process physical peaks are 237,290,528–242,025,576 bytes; build peak is
233,030,616 bytes. Total Core ML service/device memory, energy and full-model
working set remain unavailable. Stateful K/V totals 2,048 logical bytes.
Prediction times include synchronous public Python/native boundary conversions
and output allocation, but exclude fixture I/O, state creation and diagnostic
readback. Tiny slice timings are not request TTFT, token generation or a
full-engine comparison. Source/tool/dependency/package identities, M1 Pro
hardware, recorded AC/thermal conditions and per-graph timings remain in the
accepted report rather than being recombined into a new performance result.

The experiment's retain/reject/inconclusive decision is complete. It has no
qualified full decoder, stateful Qwen generation, held-out quality evaluation,
complete memory accounting or engine integration. Additional ANE research can
be a later explicitly scoped experiment; it is not an open dependency for M4.
No claim is made that ANE is generally unsupported or uncompetitive. M3's whole
milestone also requires M3.1's independently reviewed CPU retention decision.

This documentation task references accepted evidence without new conversion,
native execution, numerical tests or model downloads. Astra accepted the documentation closure without findings after checking
milestone scope against the accepted report. All three report links, combined
local document links and whitespace checks pass.
