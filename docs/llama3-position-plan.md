# Immutable positional plans and scalar Llama3 graph wiring

M8 B1b, 28 October 2026. Retained after checks and independent Astra review.
The [evidence](../benchmarks/results/2026-10-28-llama3-position-plan-v1.json)
binds source, fixtures, successful checks and the preserved initial failed check.
This follows the [scalar positional primitive](llama3-rotary-core.md); scaled
Metal and real-checkpoint support remain later [TODO items](../TODO.md).

`model::PositionPlan` stores Ordinary or validated Llama3 mathematical semantics
in immutable model state. Existing constructors default to Ordinary. A consuming
`with_position_plan` selects semantics before a session borrows the model, so a
live committed KV prefix cannot silently change its positional policy. All four
CPU Q/K rotation sites use the plan, including QK-normalized graphs. Selection
occurs at the operation boundary; no dynamic dispatch is added inside kernels.
The shared `Config` and physical KV layout remain unchanged.

The Llama adapter accepts explicit `llama3` scaling with all required parameters.
It rejects unknown fields, invalid values and unsupported types. `rope_type` and
legacy `type` aliases may appear separately or together with the same value;
conflicting or null aliases reject. An absent/null scaling declaration retains
Ordinary. Other architecture adapters retain their existing scaling restrictions.
The exact parsed plan passes from Source into the diagnostic scalar loader's
owned Model. Numerical compatibility remains distinct from text and quality.

The shared source-aware Metal check rejects scaled plans before device/pipeline
creation, even when their geometry matches an admitted ordinary tiny model.
Metadata compatibility, runtime resource/capacity inspection and ordinary Metal
loading all use it. Existing Qwen diagnostics cannot admit a scaled Llama through
their architecture/parser checks. This prevents new CPU parsing from silently
enabling the ordinary rotary shader for a scaled artifact.

## Independent graph and session checks

A new stdlib-only oracle reads hash-pinned existing tiny tied/untied safetensors;
the original fixture and producer remain unchanged. It reconstructs weights and
evaluates its own dense graph, using Decimal80 positional frequencies and FP64
mathematical operations rounded at FP32 storage boundaries. Four graphs cover
tied/untied heads, four tokens and absolute offsets 0/5. Scaling factor 8, low 1,
high 4 and original 64 exercise all three frequency bands at tiny width 8.

The frozen 2560 values cover both layers' Q/K rotations and all logits. The
existing `1e-5 + 1e-4 * abs(expected)` gate is unchanged. Full scalar execution
and cached chunks 1/3 pass; ordinary logits differ meaningfully, proving the new
plan is consumed. Exact cached-prefix reuse, reset replay and rejected-token
state preservation pass. A fresh independent export is byte-identical and all
five original fixture inputs authenticate.

Final checks pass: formatting, strict default/Metal all-target Clippy, 383 default
tests, 260 Metal-library tests and targeted Llama suites of 9 default/10 Metal
tests. Counts overlap. The Metal integration checks exercise early rejection,
not scaled GPU numerical execution. Astra authenticates all 262 bindings,
14 outputs and seven closed command groups with no blocking findings.

The first targeted graph tests passed, then strict Clippy's test compilation
found a synthetic Source initializer missing its new positional field. Setting
its existing semantics to Ordinary fixes that helper; the complete rerun passes.
The failed receipt is retained. No numerical code, fixture or tolerance changed
to resolve it.

No real weights were downloaded, no real-model or physical phone run occurred,
and no text profile, enlarged context or quality qualification is granted. Next
implement scaled Metal arithmetic and structural safety admission, then perform
the separately bounded real-model/text/quality work.
