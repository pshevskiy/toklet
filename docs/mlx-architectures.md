# Pinned MLX architecture reference inventory

22 October 2026; engine revision `1c53464`. This inventory starts M8.3 by recording the complete Python model-source trees of the language and vision reference projects. It does not add a native decoder or infer compatibility from upstream names. The selected Qwen2.5-0.5B eight-bit checkpoint remains the qualified native MLX path; other configurations and families require their own graph implementation, admission and numerical/device evidence.

| Reference snapshot | Model Python modules | Static root `Model` exports | Literal loader aliases |
| --- | ---: | ---: | ---: |
| [MLX-LM `0d4d3d04`](https://github.com/ml-explore/mlx-lm/tree/0d4d3d04f8ab7ce4a51e7209e8615755a48ba94d/mlx_lm/models) | 137 | 126 | 12 model types and one model-type/architecture pair |
| [MLX-VLM `abd3b2d7`](https://github.com/Blaizzy/mlx-vlm/tree/abd3b2d7b728cb5372d4e092260da325e5ded209/mlx_vlm/models) | 1,399 | 232 | 32 model types |

These are static source/export counts, not counts of unique graph families, supported checkpoints or runtime-qualified architectures. Nested language/vision/helper exports remain in the module inventory but do not count as root loader entry points. Conditional/dynamic Python behavior is not executed or certified. Language and vision names overlap. The machine-readable [full inventory](mlx-architecture-reference-v1.json) lists every selected path, SHA256, Git blob identity and mode, each observed root identifier and its source, direct `Model`/`ModelArgs`/`ModelConfig` bindings, and literal loader remappings.

The [stdlib-only collector](../tools/reference/mlx_architectures.py) consumes literal-SHA-bound commit/tree metadata and source archives. It reconstructs every recursive Git tree object and checks the pinned commit's root, rejects truncated or ambiguous descriptors, and requires exact selected path/blob/size/mode equality. Removing one model module must reject. It bounds the complete gzip expansion before tar parsing, including PAX/GNU headers, and never imports downloaded source or evaluates its annotations. Selected source bodies are parsed only as syntax trees.

The Git-verified subsets contain 141 language and 1,403 vision files, including loaders and license/configuration documentation. The full vision archive timed out after 120 seconds and remains explicitly incomplete. The successful alternative downloads only the authenticated selection from immutable HTTPS source URLs. All 1,403 transfers use finite owned curl processes, reject redirects, and retain their original causes and separate capture errors. Any unresolved cleanup/capture error makes collection fail; a clean transient transfer failure may retry. The actual collection needs no retries. All worker processes finish before publication. Dangling output paths reject before mutation.

Git tracks executable versus nonexecutable blobs. The completed language archive includes an unrelated `CLAUDE.md` symlink and group-write tar permissions. The assembler never follows or extracts unrelated entries; it verifies selected file bytes and executable status, records original tar modes, and emits canonical Git permissions into the deterministic subset. The strict inventory consumes these canonical regular files. No instructions from downloaded repositories are adopted.

Ten parser tests and six mocked acquisition tests pass. They cover missing modules, altered Git objects/modes, malformed archives, compressed oversized headers, traversal, truncation, source-path replacement, code that must never execute, immutable HTTPS arguments, redirects, dangling outputs, retry-cause preservation and fatal cleanup/capture failures. The [collection evidence](../benchmarks/results/2026-10-22-mlx-architecture-reference-v1.json) binds 5,691 inputs/evidence files, both subset-construction receipts, the exact command and metadata hashes. All 50 numerical/Swift/Cargo sources and protected CPU binaries match the preceding accepted lifecycle task. Its Rust 1.90 formatting, strict Clippy and 148 ordinary Metal-feature passes/74 ignored cases are inherited for those unchanged sources; this task adds no numerical or hardware trial.

The implementation queue remains ordered by mechanisms:

| Next mechanism | Required work before claiming support |
| --- | --- |
| Dense Llama/Mistral-style decoder | Inspect actual config/weight names, optional biases, tied output and rotary variants; freeze an independent tiny oracle before composing native execution. Each variant needs explicit rejection or qualification. |
| Additional dense families | Add the required normalization, activation, positions and cache semantics from their actual graphs; shared attention shape alone does not establish compatibility. |
| Mixture of experts | Implement routing, expert weight layout, bounded dispatch and accumulation with independent selection/numerical fixtures. |
| Recurrent and hybrid graphs | Add their real state ownership and update/replay semantics; ordinary transformer KV reuse is insufficient. |
| Multimodal graphs | Implement actual media preprocessing, encoders, connectors, positions and token insertion, then test real media input on both target devices. |

M8.3 remains open. Discovery/download product behavior follows native engine compatibility under M8.4; wider downloaded-model/device qualification remains M8.5. This reference snapshot does not upgrade installed MLX packages, change the native backend, accept the rejected four-bit checkpoint, qualify chat templates, or establish model fit on either device.

Astra accepted the code, reference inventory and documentation with no remaining blockers after independently checking all 5,691 bindings, two source copies, both reconstructed Git roots, 1,544 selected files, 1,403 owned transfers, 16 test results and 50 unchanged numerical/Swift/Cargo sources. Privacy, 270 local links and whitespace pass. Acceptance covers static reference coverage only; native architecture expansion and M8.3–M8.5 remain open.
