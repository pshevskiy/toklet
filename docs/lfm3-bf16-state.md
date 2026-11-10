# LFM3 transactional BF16 state

Status: Astra approves scoped CPU state retention, with no findings.
This milestone adds state ownership for the authenticated 24-layer LFM3 topology.
Complete model execution, CLI adoption, quality and
working-set qualification remain open.

The owner contains 13 width-2,048 convolution histories, six full KV caches and
five 256-token sliding KV caches. Histories retain six raw BF16 `B*x` rows.
K/V remain separate head-major `[4, capacity, 128]` buffers. A sliding window
includes the current query and at most 255 previous positions. Preparation
borrows old logical spans plus the entire new chunk; no overwrite happens until
every query has consumed that view and the caller publishes the successful step.

Storage uses a sealed specialization for `f32` and `bf16`. Existing FP32 public
entrypoints remain concrete, and their arithmetic bodies are preserved. BF16
storage has its own capacity dtype. No BF16 attention arithmetic or numerical
dynamic dispatch is introduced. The BF16 history calls the previously qualified
[convolution leaf](lfm3-binary-conv-source.md); its candidate is copied only after
numerical success. Failure may leave partial caller output, never committed
history or a published candidate.

All 24 proposals must match their owner identity, generation, committed offset
and chunk size before any owner changes. Commit then has only infallible copies
and scalar updates. Reset prevalidates every owner before clearing histories and
invalidating logical KV. Owners are exposed immutably, so callers cannot publish
one family independently. Changed or shortened token prefixes require reset and
replay; sampled/pending token policy belongs to the later session integration.

Memory admission counts unique BF16 buffers and actual vector capacities,
including all owner headers. Local caches physically retain 256 rows even when
the admitted context is shorter; the enclosing owner enforces that smaller
overall capacity. At capacity 2,048, history payload is 319,488 B and KV payload
is 27,787,264 B; complete owner allocation is 28,108,984 B including headers.
Weights, candidate banks, caller input/output, scratch, tokenizer
and application state remain separate. This component allocation does not prove
that a whole model fits the unchanged 6 GiB engine ceiling.

The [frozen source contract](../artifacts/lfm3-bf16-state-2026-11-11/prospective-contract-1.json)
precedes new fixture outcomes. An independent integer/BF16 producer covers
seven chunk schedules, 60 history observations and 120 KV observations,
including wrap boundaries, short/medium/long decode and capacity exhaustion.
The compact checked-in fixture preserves every word/range/mask digest and hashes
the exact slot-ordered absolute-position arrays. The full fixture is retained
as [raw evidence](../artifacts/lfm3-bf16-state-2026-11-11/fixture-1.json).

The checkpoint declares Transformers 5.20.2, whereas the shared installed
5.7.0 cache API predates it. Official authenticated 5.20.2 and Tokenizers 0.23.1
wheels are extracted into an isolated reference target; the shared environment
is unchanged. Actual declared-parent and MLX CPU/Metal state probes each match
737,280 history words, 73,808,896 KV-view words and 17,733,167 mask cells exactly.
MLX's physical ring and the parent's 255 stored rows are compared by logical
positions rather than an assumed identical physical layout. MLX-LM imports Torch
despite the requested environment flags; the recorded runtime and memory audit
include it. These logical-state results do not reverse the earlier rejected
CPU/Metal arithmetic correspondence.

Fresh formatting, both strict all-target Clippy configurations, 714 CPU tests,
860 Metal tests and nine example tests pass. Eight optimized state tests cover
the fixtures, all 13 learned banks, whole-state publication, abort/retry/reset,
foreign/stale proposals, shape/capacity/generation overflow and every finite or
nonfinite BF16 pattern. Debug and release allocation tests each exercise 24
transitions, 312 history proposals, 264 KV proposals and 96 rejections with zero
Rust allocations. Native76 builds three arm64 slices with 46 exports each;
strict C links, existing CPU/Metal tiny-model execution smokes and all 34
compiled console negative controls pass. Four portable tests pass in ordinary
and optimized Python modes. These are scoped checks; iPhone runtime and a full
LFM3 GPU/model path are unqualified.

The [frozen timing contract](../artifacts/lfm3-bf16-state-2026-11-11/performance-contract-1.json)
requires two reversed-order runs: 48 existing FP32 compatibility cohorts and
18 BF16 state cohorts. Each FP32 matched median must stay within the original
5% regression bound. BF16 has no existing model baseline: construction, first
transition, warmed history/KV/publication phases, raw samples and memory are
reported separately. All 24 matched FP32 pairs pass, with median changes from
-9.05% to +3.79%. Eighteen BF16 cohorts measure 0.338–0.365 / 7.119–7.254 / 27.706–28.644 ms at
1/32/128 tokens across offsets 0/511/1,023; these ranges are component medians.
The [result index](../benchmarks/results/2026-11-11-lfm3-bf16-state-v1.json) links
the raw pairs/samples and dispersion. Reset and committed-depth seeding occur outside transition
clocks. Synthetic shared input/KV staging is not a model graph or actual model
scratch layout. No complete request, TTFT, token throughput, quality or dtype
speedup claim follows from these component timings.

Evidence lives in
[`artifacts/lfm3-bf16-state-2026-11-11/`](../artifacts/lfm3-bf16-state-2026-11-11/).
Next qualify BF16 projections/attention, compose
the source graph, connect model loading/session/CLI, and run the unchanged full
quality, memory and matched-engine performance gates. Observability's opt-in 1%
summary overhead bound is still unqualified; interactive frames and iPhone
runtime remain deferred.
