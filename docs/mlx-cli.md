# Explicit native MLX CLI execution

A Metal-enabled macOS build can run a compatible local MLX Qwen artifact through
Toklet's Rust engine:

```sh
cargo run --release --features metal -- \
  --model /absolute/path/to/local-model \
  --backend metal --metal-projection mlx-affine \
  --metal-attention reference \
  --context-capacity 32 --prefill-chunk 3 \
  --tokens 1,7,13,3 --max-new-tokens 4 --json
```

The IDs and capacity above are for the tiny fixture; use valid inputs and admitted
capacity for the selected model. Text input needs its local tokenizer/chat assets
as with existing CLI paths. This command does not search or download models.

`mlx-affine` explicitly selects `Model::load_mlx`, resident affine projections,
compensated normalization, Half activations/KV and full-logits readback. Attention
is reference by default; `--metal-attention online` is separately selectable.
The default CPU/Half/Q4 paths are unchanged. CPU selection, missing Metal support,
duplicate selection and conflicting `--q4-weights` reject without a fallback.
Supported configuration, geometry, payload and GPU-domain validation remain in
[the source](mlx-qwen-source.md) and [the graph loader](mlx-metal-model.md).

JSON statistics and interactive readiness report `MlxAffine`, `Compensated`,
`FP16` KV, and `MLX affine + FP16 exclusions` weights. They describe quantization
as per-tensor bits/groups; no single global group or Toklet Q4 format version is
invented for mixed MLX artifacts. Logical memory and application admission reuse
the existing accounting. As documented for the graph loader, conversion timing
includes payload reading, validation, upload and conversion together.

The [explicit release subprocess test](../tests/mlx_cli.rs) runs ten local Mac
processes: four greedy generation, four hidden-stop and two persistent interactive
sessions over reference/online attention and one-/three-token chunks. Generation
uses the independently frozen `[47,47,47,47]` tiny IDs. Interactive checks require
seven exact emitted IDs per session, complete evaluated/pending counts across
cache reuse, both continuations and reset/restart, input rejection and correct
readiness/request metadata. Stop checks require one sampled token, no emitted
token, four committed tokens and one pending token. The tiny logical weight
allocation remains 18,152 bytes and the reported memory parts sum to admission.

The first test version passed but had weaker state assertions. Its source and raw
logs are retained; Astra requested the exact state/ID checks above, then the
strengthened version was checked again. No runtime or numerical-gate change was
needed for that review refinement. Formatting, strict CPU/Metal-feature Clippy,
ordinary Rust tests and the explicit GPU subprocess test are recorded in the
[task receipt](../benchmarks/results/2026-10-22-native-mlx-cli-v1.json).

This exposes the selected tiny implementation through the CLI. It does not
qualify real checkpoint quality, speed, process peaks or iPhone execution. Native
Apple app selection, architecture expansion and device-aware discovery/downloads
remain open in the [M8 plan](mlx-implementation-plan.md).
