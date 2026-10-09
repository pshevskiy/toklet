# M2.2 real-model FP32 forward qualification, 9 October 2026

The scalar decoder passes all full-array probes against pinned Transformers
5.7.0 / PyTorch 2.9.1 CPU FP32 eager execution, using the pinned Qwen checkpoint
and disabled dropout/cache. Cases cover four tokens, a changed future token,
three tokens at absolute position offset five, and one token. Each compares
embedding, all-position logits, final normalization, and 18 intermediate arrays
at layers 0, 1 and 23. All-position argmax choices match; the ordinary final-only
forward output exactly matches the diagnostic path's final row.

| Case | Maximum logit absolute error | Maximum intermediate tolerance fraction |
| --- | ---: | ---: |
| short | 0.000785828 | 0.634608 |
| future_changed | 0.000785828 | 0.634608 |
| offset | 0.000694752 | 0.489321 |
| single | 0.000130177 | 0.085088 |

[Detailed numerical metrics](2026-10-09-qwen-forward.json) retain every compared
array's maximum absolute error, relative error with a `1e-6` floor, and tolerance
fraction. Short teacher-forced target NLL differences have magnitude at most
0.000217482 nats, below the separate `1e-3` short-probe check. These few targets
do not qualify the frozen 10,000-token held-out loss or 100-task quality suite.
No latency or longer-context result is claimed.

The model stores weights/activations in FP32. Projection products and reductions
now use FP64 accumulation, with one FP32 rounding after bias. RMSNorm and
softmax also use FP64 intermediates; rotary combination, activations, attention
dots/context and residuals remain FP32. The original FP32 projection functions
are preserved as independent scalar operations. This precision choice addresses
accumulation through 24 layers; its latency cost remains for M3 measurement.

The initial real-FP32 policy and its revision are retained explicitly:

- A provisional uniform `1e-4 + 1e-4 * abs(expected)` gate failed the independent
  Transformers FP32 versus all-FP64 envelope itself, before Rust was evaluated.
  The largest observed final-normalization difference was 0.00320436.
- [V1](../../tests/fixtures/qwen-v1/manifest.json) froze separate per-case limits
  `max(1e-4, 2 * independent_FP32_FP64_max_absolute_error)` before Rust evaluation.
  FP32 projection accumulation narrowly failed single-token final normalization.
  FP64 projection accumulation resolved that issue but narrowly failed one
  near-zero single-token logit: -0.0065194573 versus -0.006417036, fraction 1.01768.
- Astra independently found that the same causal first-token logits in single
  and multi-token Transformers execution differ by up to 0.000655651, or 5.47
  times the v1 single-case gate. At the reported logit, two equivalent reference
  executions differed by 0.000550106. The gate did not cover the oracle's own
  shape-dependent accumulation. No layout or numerical implementation error was found.
- [V2](../../tests/fixtures/qwen-v2/manifest.json) is a new version. For each named
  operation **and layer**, it freezes the maximum independent FP32/FP64 envelope
  across the four cases, plus measured equivalent-prefix FP32 shape variation.
  Absolute limit is `max(1e-4, 2 * pooled_FP32_FP64_max_abs,
  2 * equivalent_prefix_FP32_max_abs)`; relative limit stays `1e-4`. Early and late
  layers retain distinct limits. Limits derive exclusively from independent
  reference calculations; Rust differences do not enter their calculation.
  The unchanged implementation then passed all v2 comparisons.

Tiny-v1 remains unchanged. V1 remains available as policy-audit evidence and is
not the accepted pretrained qualification. No FP16 or quantized gates change.

The exporter reads the exact frozen BF16 source values into FP32 and captures
actual Transformers module/eager-attention outputs. Its separate functional pass
computes all 24 layers in FP64 from those same weight values, including reductions
and rotary angles. It also evaluates identical valid prefixes in different
shapes. Full arrays stay in ignored `artifacts/`; 228 selected probes per version
plus source identities, tool versions, gate derivations, array hashes and argmax
are versioned. Every committed probe also anchors the complete array's SHA-256
and byte count. The explicit Rust suite verifies the actual checkpoint SHA-256
and full/selected reference identities before comparison. Regeneration after
adding those anchors reproduced every full array, selected value, and gate.

Reproduce with the locked Python environment and explicitly prepared model:

```sh
.venv/bin/python tools/reference/qwen.py --output artifacts/qwen-forward-v2 --probes-output tests/fixtures/qwen-v2
python3 tools/reference/qwen.py --check tests/fixtures/qwen-v1
python3 tools/reference/qwen.py --check tests/fixtures/qwen-v2
cargo test --locked --offline --release --test qwen_forward pinned_qwen_fp32_matches_full_transformers_forward -- --ignored --nocapture
```

Both export directories must be new. Regeneration in an existing checkout uses
fresh candidate directories; compare with frozen selected values before placing
the full local reference at the explicit suite's path. Ordinary Rust tests need
neither the model nor Python. Qualification ran on Apple M1 Pro, macOS 26.6.2,
Rust 1.90.0, using pre-task revision `8b9d892` plus this task's reviewed changes.
The reference manifests retain exact exporter hashes and pinned source identities.
