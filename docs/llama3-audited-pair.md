# Pinned Llama3 audited native pair

The pinned Llama 3.2 1B distribution now reuses the dense consumed-source loader,
model assembly and complete chat identity report. The additive C entry point is
`toklet_model_load_mlx_llama3_audited`. Its loaded context is capped at 512;
the model declaration remains 131,072 and is not a qualification result.

The six-file manifest uses the existing config/container/index/tokenizer/policy/
ancillary order. Source metadata and every physical container range must be
successfully consumed before the assembled model receives its receipt. The
complete factory checks that receipt and passes the exact authenticated
text buffer into the shared codec. The direct pinned Llama codec is a thin
wrapper around the same implementation. No Jinja is executed.

The legacy config allowlist stays unchanged: generic Llama loading does not
enable either legacy Chat or complete audited Chat. The audited pair uses
MlxAffine projections, compensated normalization, Reference attention and the
existing scaled-RoPE path. Its separate text allowance is 256 MiB. The identity
report has distinct `llama3_audited_v1` and `llama3-plain-20240726-v1` policies;
model ownership, source receipts and all six asset roles must agree.

Cold controls compare independent artifact/text pins, reject cross-profile
receipts, excessive context and each mismatched identity asset. The explicit
Mac checks exercise both the generic rejection and audited pair. The audited
request uses C-returned prompt IDs and stop IDs, checks actual identity fields,
rejects a same-length tokenizer mutation in a private text-only copy, checks
allowance rejection and exact owner retention, and runs the existing 512-token
boundary and cancellation/replay journey. No original cache files are changed.

The selected Apple framework remains the prior 39-export package until a new
three-platform package and managed app route pass their own checks. Full adoption
still requires the original unquantized 10,000-target and 100-task gates. App
activation, ordinary discovery, iPhone, larger contexts and throughput remain
separate. This source task changes no numerical kernel or quality threshold.

The corrected cold regression run passes548default and656Metal tests. Final
strict Clippy in both configurations, formatting,13targeted cold controls and
the explicit cached tokenizer/streaming check pass. The current debug Metal
library links from C and exports exactly the40header symbols. A first compile
missed the streaming profile branch; a later test-only FFI reference typo was
corrected. Both failed receipts are retained.

Both explicit Mac modes exit cleanly and recheck all six files after releasing
native owners. The audited identity records374verified container reads. Both
runs admit511+1, reject511+2 unchanged, cancel with one pending ID and recover
with IDs8586/596. Neither run observes a hidden stop.
[Evidence](../benchmarks/results/2026-11-03-llama3-audited-pair-v1.json).
