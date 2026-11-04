# SmolLM3 per-layer rotary selection

5 November 2026. Block implementation passes Astra review; real checkpoint execution is
not yet qualified. Reuse the dense RMS/SiLU/GQA/affine blocks and add one source
mechanism: choose which layers apply ordinary rotary position transforms.

`no_rope_layers` is an explicit integer0/1 mask: **1 rotates,0 bypasses**. Parse
its exact layer count once, require all declared attention layers to be full
causal, and reject ambiguous values or unsupported positional scaling. An
explicit mask takes precedence over the export's positive interval; it is not
replaced by a hardcoded every-four pattern. Existing families default to all
rotate. `PositionPlan` still defines the rotation convention and retains its
existing scaled-Llama restriction.

The pinned SmolLM3 export omits `tie_word_embeddings`. Default true only for this
adapter and require a single embedding table when the field was omitted.
Explicit tied/untied tiny variants exercise the existing output ownership rules.
The converted unsplit export's `pretraining_tp:2` is accepted as metadata after
full tensor-shape validation; no tensor-parallel execution is introduced.

Scalar loading owns an immutable bounded mask, counted in model metadata. The
scalar layer loop passes projected/post-normalization Q/K directly to attention
and cache on bypass layers. Metal resolves a boolean into each resident layer
at loading, selecting the existing input buffers without rotary dispatch or a
copy. Snapshots point at the selected logical outputs. Weights stay immutable;
KV remains session-owned. SmolLM3 native policy stays Reference attention and
512 positions, with no managed Chat or iPhone admission.

The independent tiny fixture reuses unchanged dense arithmetic primitives with
fresh mixed-layer composition. Two source variants use opposite masks, offsets
0/5/509, causal future-tail changes and complete39-stage inventories. FP32-ordered
and FP64-wide source calculations retain Half storage and the existing
`max(1e-4,2*source-envelope)+1e-3*abs(reference)` policy, frozen before target runs.
Fresh generation reproduces the complete fixture byte-for-byte. Scalar checks
cover identity, a hand-calculated nonzero rotary pair, cached continuation and
reset; malformed masks, attention declarations, tying and scaling reject.
Metal passes1,680 snapshots/110,208 comparisons and bit-exact bypass identity,
including nonzero-offset cached chunk schedules. Ordinary/scaled Llama controls
also pass; four explicit GPU tests,554 default/662 Metal tests, formatting,
strict Clippy and release build pass. The first full check caught three direct
Source test fixtures missing the new field; corrected None initializers preserve
old behavior. That failed run remains.

[Checks and evidence](../benchmarks/results/2026-11-05-smollm3-rotary-selection-v1.json).
[New-model queue](mac-coverage-performance-next.md).
