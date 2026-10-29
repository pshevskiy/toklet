# LFM2 cold metadata plan

29 October 2026. B4a validates the pinned LFM2-350M MLX export's config,
index and original safetensors header. It does not execute or admit the model.
The [source and implementation plan](lfm2-hybrid-plan.md) records the selected
loader semantics and remaining numerical work; the [machine result](../benchmarks/results/2026-10-29-lfm2-cold-plan-v1.json)
pins code review, fixture and complete check evidence.

`mlx::lfm2::inspect` rejects duplicate JSON keys, unsupported config/alias values,
missing or extra tensors, wrong shapes/dtypes, gaps, overlaps, invalid bounds and
overflow. The selected inventory has 93 affine matrices, 45 direct BF16 norms
and ten `[1024,3,1]` depthwise coefficient owners: 148 logical owners and 334
physical descriptors. Thirty B/C/x row views borrow their fused parents.
The shared metadata helper is renamed from `gemma_metadata` to `tensor_metadata`;
its helper bodies and existing Gemma public reexports are preserved.

Six full-attention recipes have separate KV owner IDs; ten short-convolution
recipes have separate history IDs. Hidden width is 1024, actual FF width 4608,
and attention uses 16 query heads, eight KV heads and head dimension 64.
Declared context is 128000; implemented and qualified capacities are absent.
At a hypothetical capacity 512, Half numerical state payload is 6,373,376 bytes:
6,291,456 KV plus two 40,960-byte convolution banks. This pure equation excludes
guards, staging, scratch, weights and application/framework allowances. It is
neither memory admission nor measured process fit.

Six composite metadata controls and full package checks pass. Formatting and
strict all-target Clippy pass for default and Metal configurations. Default tests
pass 438 with 20 ignored; Metal-feature tests pass 519 with 169 ignored. These
counts overlap and are not hardware qualification. No LFM2 payload, model or
explicit GPU execution was performed.

The first full Metal run exposed a stale compatibility test: it rejected ordinary
Llama theta 12345 despite earlier structural admission. Astra independently
checked the history and reviewed a test-only repair. A genuinely unsupported
positive theta 0.5 now verifies rejection before GPU creation for all four
families; a separate metadata-only Llama 12345 control verifies compatible but
untested status, pending device planning and false text/quality qualification.
Production logic and numerical gates are unchanged. The failed first receipt is
retained alongside the passing retry.

B4b continues with the existing scalar convolution primitive and transactional
history integration. Tiny hybrid composition, resident leaves, actual payload
verification and bounded real-model checks remain open. Native direct-gain RMS,
including epsilon 1e-5 at both 64 and 1024, and actual GQA 16:8/D64 need their own
numerical qualification. Chat, full declared context, iPhone and performance are
not established by this cold task.
