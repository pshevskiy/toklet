# M9 — Measured performance improvements

Planned next milestone after the active [M8 shared-block and capacity work](shared-block-model-plan.md).
Requested by the user on 28 October 2026. Source:
[Toklet next performance improvement plan](https://chatgpt.com/space/page_18e155f0450c8191a76010b2b26832a5),
Page sequence 2, updated `2026-10-28T05:53:35.133190Z`.
The repository sequence and architecture obligations below make that Page
actionable without starting its optimization experiments during M8.

## First delivery boundary

**P0 → one measured P1 candidate → P2.** A second candidate and the later
conditional techniques are separate decisions. M9 closes with a qualified
adoption or a documented rejection; shipping every technique is not its exit
criterion. Each code task requires Astra review, appropriate checks, an
improvement-record update and a commit.

| Task | Delivery and decision |
| --- | --- |
| P0: baseline | Refresh the pinned MLX8 configuration on the completed M8 revision. Collect bounded operator profiles and separate uninstrumented completed-work timings. Cover every relevant prefill chunk or label sampled offsets explicitly. Rank normalization versus affine projection cost and freeze the primary metric before coding. |
| P1A or P1B | Choose parallel Compensated RMSNorm if normalization has a useful cost budget, or a four-token native MLX affine prefill tile if projection reuse is more promising. Preserve the current path and change one mechanism. If neither is useful, record why and select the observed bottleneck. |
| P2: decision | Apply the unchanged numerical, held-out quality, session/state and memory gates. Confirm useful end-to-end performance independently with matched conditions. Adopt only the validated device/configuration or record the rejection and remove unused complexity. |

The initial Page target is `mlx-community/Qwen2.5-0.5B-Instruct-8bit`, revision
`6474d4c1a05bd7bd634a9c993560e4e7272eda3d`: affine8/group64, Half activations
and exclusions, F16 KV, Compensated normalization, Online attention, FullLogits,
chunk 128. The app matrix uses capacity 1152, a 1.5 GiB numerical budget and a
separate 2.5 GiB process trial cap. Frozen loss/task recipes keep their own
capacities. Revalidate source hashes, current eligibility and constructor
accounting before running. This is distinct from Qwen3's native Reference
diagnostics and from Toklet Q4 packing.

The Page's baseline revision `2f50fdf6c06220b0c8ccc61d6c53fd840ab9aed2` and its
measurements are historical. They are not current-M8 comparison samples. Start
with Mac M1 Pro, then collect the separate iPhone 14 Pro evidence when available;
missing phone evidence remains open and cannot be inferred from Mac results.
The Page's 5–10 focused engineering days estimates the first cycle with both
devices available; it is not a delivery date or an estimate for every later task.
Its graph-coverage description also predates the current four dense family
adapters and partial Gemma hybrid/shared-state work. Inspect and reuse current
code and scoped evidence rather than restarting those implementations.

## Architecture obligations during M8

Keep family wiring and numerical semantics independent from backend scheduling.
The following are extension boundaries, not a request to add placeholder enum
variants, unused traits or a general graph interpreter. Implement a new variant
only with its mechanism, fixtures and concrete consumer.

| Boundary | Preserve now | Later consumer and required evidence |
| --- | --- | --- |
| Normalization | The family selects epsilon, gain convention, residual order and rounding boundaries. Kernel scheduling is selected at an operation/session boundary. Existing serial/reference implementations remain available. | P1A changes reduction order even with compensation. It needs independent numerical/quality checks and an explicit candidate selector, without silently redefining Compensated semantics. |
| Projection | Resolve weight representation, row spans, strides, dimensions and reconstruction math during loading/planning. Distinguish single-token decode from multi-row prefill without family-name dispatch in inner loops. | P1B shares reconstructed weights across four rows while preserving each row's arithmetic, tails and guards. Existing MLX bit widths and unsupported layouts keep their qualified path. Small query/verification scheduling is a later concrete use. |
| Attention | Model semantics specify Q/K/V dimensions, head mapping, mask, positions, retention and producer/consumer routes. Reference/Online or partitioning is an execution choice with separate eligibility. | P3 requires stable split-state merging, device-specific crossover and explicit scratch. A Qwen result cannot enable Llama Online or change full attention to a truncated window. |
| Physical state and capacity | Count unique physical owners, implemented storage, guards, scratch and phase overlap. Keep weight quantization separate from KV representation. Parameterize the solver with actual constructor recipes. | P4 must add real cache encoding, writes, reads and fused reconstruction plus scales/metadata/workspace accounting before offering I8 KV. An estimated smaller byte count is not codec support. |
| Session transaction | Track sampled history and committed positions separately. Cache identity includes weights, positional and numerical semantics. Failure leaves a valid prefix or requires explicit replay. | P5 needs bounded tentative state and per-position target decisions. Full-KV logical truncation does not restore overwritten ring slots or recurrent state; those need concrete shadow/checkpoint/replay handling before verification is admitted. |
| Outputs and profiling | Keep full diagnostic logits distinct from ordinary token selection. Profile labels identify operation, layer and chunk/position coverage, with bounded storage and completed GPU timing. | P0 must not describe the last successful chunk as a full prompt profile. G2/P5 can reduce work only when the requested outputs and all future state remain correct. Profile instrumentation stays outside acceptance timings. |
| Capabilities | Report graph, packing, backend mode, device fit, text protocol and validation separately. Persist settings and cache keys against artifact/plan identity. | An optimization is enabled only for its qualified shape/device/mode. Missing counters, codec, rollback or output support receives a specific rejection rather than a silent semantic fallback. |

B0 supplies checked physical sizing and the pure range solver; it does not
implement every future state kind or codec. B1–B4 add mechanisms as actual
architecture compositions need them. B5 presents only implemented settings.
These boundaries let M9 reuse the same blocks instead of adding a second engine.
The B0 binary search requires nondecreasing allocation sizes. A future adaptive
split/tile schedule must reserve a checked nondecreasing workspace envelope over
the admitted capacity/query range, or use a separate finite piecewise solver.
Do not pass an unproven depth-dependent workspace function to that search.

## Follow-on queue and limits

- **P3:** partitioned Online decode attention, after a qualified configuration
  and measured long-context bottleneck. Charge all split/merge workspace.
- **R1:** rotary frequency/phase reuse only when a profile justifies storage and
  complexity; it does not change the model's positional transform.
- **P4:** integrated I8 KV only for a defined capacity need, with full codec,
  attention, quality, state and peak-memory qualification.
- **P5:** bounded strict-greedy verifier, then a separate prompt-lookup proposer.
  Verify every target decision including lowest-ID ties and zero/partial/full
  acceptance, EOS, cancellation and capacity boundaries. Investigate batched
  versus sequential near-tie winners; do not introduce relaxed acceptance.
- **Gemma G0–G3:** inventory and qualify the selected checkpoint first. Reuse M8
  implementation/evidence where valid, then consider architecture-defined shared
  KV fast prefill and a compatible MTP assistant. Inter-layer sharing is distinct
  from prompt-prefix caching; preserve a full-position scoring path.
- **Llama Online:** reproduce and resolve its frozen correctness failure as a
  separate task. Neither M8 Llama breadth nor a Qwen optimization waives it.

Paged KV, continuous batching, eviction, trained speculative systems, novel
low-bit caches and new hardware backends stay deferred until a concrete workload
and prerequisite implementation justify them. No research headline is a Toklet
speed or quality forecast.

## Acceptance policy

Follow [PLAN.md](../PLAN.md): freeze identities, token IDs, cache state,
arithmetic, timing boundaries and the primary metric. Use release builds,
three warmups and ten measured repetitions for initial short comparisons,
retain raw samples and dispersion, and independently confirm a promising gain.
The default useful-gain gate is at least 5%, exceeding observed variation,
without an unexplained required-workload regression. Preserve existing loss/task
gates and investigate numerical discrepancies rather than loosening tolerances.

Count actual process peaks as well as planned buffers. Keep loading, profiling,
prefill, decode, sampling and visible app latency distinct. Physical phone
adoption includes its sustained/thermal and application checks; unavailable
energy remains unavailable. Scope the checks to the mechanism changed, screen
cheap correctness/usefulness first, and stop a candidate that misses a frozen
gate or merely moves work outside the measured interval.
