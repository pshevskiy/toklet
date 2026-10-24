# Real Qwen3 generation callback recovery

24 October 2026. This tests-only task extends the
[selected real Mac qualification](mlx-qwen3-real-metal.md). Production numerical,
generation, native bridge, CLI and Swift code are unchanged. The frozen
checkpoint, reference and numerical gates are unchanged.

The [explicit GPU test](../tests/mlx_qwen3_real_model_metal.rs) invokes the public
`generation::generate` API on the real Qwen3 checkpoint using both full-logit and
token-only readbacks. It proves:

- Cancellation raised by the first successful output callback returns Cancelled
  after one sampled/emitted token. Four prompt tokens remain committed and the
  sampled token remains pending. A pre-cancelled Reset leaves that entire pending
  state unchanged and invokes no callback.
- Continue consumes the pending token once, emits the remaining frozen IDs and
  matches uninterrupted execution with the same chunk/readback recipe. After
  discarding the final pending sample, complete full-logit distributions agree
  exactly and pass the frozen step3 gate. Token-only publishes no full logits
  and must have an available, correct token selection.
- A real callback error returns the original typed Output error, retaining one
  pending sampled ID and the valid committed prefix. Continuation consumes it
  once and matches a fresh two-step generation, including the full distribution
  and frozen step1 gate where available. A failed call returns no Report; no
  emitted-token count is inferred for that error.
- A hidden stop ID is sampled into pending history without invoking the output
  callback. Its report distinguishes one sampled token from zero emitted tokens.
  Continuation evaluates it once and produces the expected next ID.
- Invalid reset tokens, output-count overflow and invalid stop IDs preserve the
  complete evaluated state, including its available logits/token selection.

Both readback witnesses pass on Apple M1 Pro in an optimized build. Formatting,
strict default/Metal all-target Clippy, 162 CPU tests and 181 Metal-feature tests
pass; 100 tests remain explicitly ignored in the ordinary Metal-feature suite.
The earlier four real-model GPU tests and shared regressions are inherited
from their accepted receipt: their bodies and all production sources are proven
unchanged. They were not rerun for this follow-up.

Astra required explicit available-token assertions on two recovery paths so
`None == None` could not pass. The initial passing checks and intermediate test
source are retained separately from the final checked source. The
[receipt](../benchmarks/results/2026-10-24-qwen3-real-generation-v1.json) binds
both check versions, the actual GPU output and prior-source identities.

This qualifies deterministic callback-triggered generation cancellation and
recovery on the selected Mac checkpoint. It does not test cancellation during
GPU execution or prefill, physical UI cancellation, iPhone lifecycle, held-out
quality or performance. Those scopes, chat/thinking integration and broader
architecture/discovery work remain open in the
[active plan](engine-expansion-plan.md).
