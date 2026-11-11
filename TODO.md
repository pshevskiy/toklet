# Toklet progress checklist

Updated **11 November 2026**. **Improvement goal active; earlier Mac milestones
complete only in their recorded scopes.** iPhone runtime work remains deferred.
Read the current task, relevant plan criteria and targeted source. Completed
history is in the [dated archive](docs/archive/2026-11-09/README.md).

## Current and next task

Active goal: implement the whole selected observability/performance/model
roadmap. Order remains Gemma 4 E2B, Qwen3.5-4B, Qwen3.5-9B, then the selected
under-4B track; [model/metric plan](docs/model-observability-plan.md).

Current/next task: **score the complete original LFM3 loss and task suite
against the stock and modified references**. The authenticated CPU graph,
explicit CLI route and 1,025-position numerical comparison are reviewed.
Preserve the 6 GiB ceiling, original numerical and quality gates, and prior
adoption decisions. Engine and CLI work leads. Interactive screen capture
and iPhone runtime remain deferred.

- [ ] S1.L original quality: score all 10,000 loss targets and 100 tasks for
  both stock-to-modified and modified-to-native comparisons. Original loss
  inputs match, with a trailing newline difference in 40 stock code prompts.
  The full 10,000-target stock-to-modified loss comparison passes with
  a mean NLL increase of 0.00781 nats per token. Task collection is active.
  The first 1,024 native targets match the modified greedy IDs with
  3.55e-15 maximum absolute NLL delta.
  The remaining nine native blocks are queued after the CLI exits.
  Paired isolated grading and the final quality decision remain open.
  [Campaign](docs/lfm3-quality.md).

- [x] S1.L external comparison preflight: authenticated metadata-only views
  preserve numerical configuration and source bytes. Installed AutoConfig and
  MLX-VLM built-in config admission pass; the publisher module's incompatible
  interface is recorded. Eight portable controls and the 512 MiB guarded probe
  pass after correcting a synthetic fixture byte count and bounding actual reads
  against concurrent growth. Astra approves with no findings.
  Model startup, matched inference and performance remain queued after quality.
  [Scope](docs/lfm3-external-config.md).

- [x] S1.L isolated grading window: qualified Podman timezone, independent
  guest shutdown check and the original CPU budget with its qualified
  ambiguous-137 witness. Five portable controls and the real stopped-guest
  supervisor smoke pass. Astra approves with no findings.
  [Checks and review](artifacts/lfm3-context1025-2026-11-11/quality-inputs/grading-window-publication-checks.json).

- [x] S1.L original quality tooling: guarded native/source collection,
  trajectory and console replay, literal grading, and complete-only paired
  decisions. Independent replay matches all 16 source pilot decisions.
  Fmt, both strict Clippy, 733 CPU, 879 Metal, 11 examples and six
  portable tests pass. Astra approves after binding identical modified
  reference inputs across pairs.
  [Checks and review](artifacts/lfm3-context1025-2026-11-11/quality-inputs/quality-publication-checks.json).

- [x] S1.L 1,025-position comparison: all 854 stages and 68,054,016 BF16
  words match. Both late retries and 30 wrapped views pass. The phased
  reference corrects two 512 MiB guard failures and matches the prior
  prefix and full-head capture. Ordinary one-token median is 1.285 s.
  Fmt, both strict Clippy, 733 CPU, 879 Metal and nine examples pass.
  Native81 inputs and both C smokes match. Astra approves with no
  findings. [Scope](docs/lfm3-context1025.md),
  [review/commit](artifacts/lfm3-context1025-2026-11-11/publication-checks.json).

- [x] S1.L real 513-position rollout: both frozen schedules match 1,342
  observations and 70,418,432 BF16 words, including all nine shared tied-logit
  endpoints. Forty split old-cache views and both late-abort retries pass.
  Guarded source/native combined RSS stays under 512 MiB/6 GiB respectively.
  Isolated ordinary one-token median is 1.272 s near the wrapped prefix.
  Observer overhead remains unqualified. Fmt, both strict Clippy, 733 CPU/
  879 Metal/nine targeted example checks, native81 three slices and two C
  smokes pass. Astra approves with no findings. Full quality and adoption remain
  open. [Scope](docs/lfm3-context513.md),
  [review/commit](artifacts/lfm3-context513-2026-11-11/publication-checks.json).

- [x] S1.L generation, codec and CLI: explicit `lfm3-bf16` profile with
  audited assets, pending/committed history, exact prefix reuse, replay,
  cancellation and request metrics. Five chat and three raw source cases
  pass, with 284 MiB standalone codec peak RSS. Real CLI checks cover
  14 state/output cases, four rejections and a hidden pending stop ID.
  Both release sweeps pass, at 3.026–3.045 s after reset and 34.9–35.5 µs
  fully cached for four tokens and zero output. Fmt, strict CPU/Metal Clippy,
  733 CPU/879 Metal/nine example tests, explicit codec, native80 and both C
  smokes pass. Initial padding compile and RSS query errors corrected. Astra
  approves with no findings. Full quality and managed adoption remain open.
  [Scope](docs/lfm3-cli.md),
  [review/commit](artifacts/lfm3-cli-2026-11-11/publication-checks.json).

- [x] S1.L real BF16 graph and audited loader: all 24 layers, tied head and
  owner proposals match 366 source stages and 2,476,032 BF16 words on three
  short schedules. Late abort/retry and zero-allocation prepared forward pass.
  Real verification peaks at 3.27 GiB aggregate RSS. Fmt, both strict Clippy,
  731 CPU/877 Metal/nine targeted example tests, three native79 slices and two
  C smokes pass. Both release sweeps pass, with 1.252–1.260 s single-token
  continuation and 32.04–32.20 s loading. Initial header inventory count
  compile error corrected. Astra approves with no findings. Generation,
  codec, CLI, deeper parity and full quality/adoption remain open.
  [Scope](docs/lfm3-bf16-graph.md),
  [review/commit](artifacts/lfm3-bf16-graph-2026-11-11/publication-checks.json).

- [x] S1.L BF16 CPU causal/window attention: 25 pinned-source cases and
  17,742,592 exact BF16 stage words, 8,128 FP64 diagnostic samples per stage,
  wrapped views, and zero-allocation debug/release checks pass. Fmt, both
  strict Clippy configurations, 728 CPU/874 Metal/nine targeted example checks,
  five optimized source tests, and native78 three 46-export slices and CPU/Metal
  C smokes pass. All 24 release component cohorts are retained. Initial test
  borrow and six softmax rounding differences were corrected. The extra
  all-examples build reproduces a parent Gemma fixture error. Initial runtime
  provenance was incomplete. Guarded replays verify 28,982 runtime files at
  both boundaries and reproduce the Torch and MLX outputs exactly. Strict MLX
  component correspondence still rejects. Astra approves with no unresolved
  findings. Full graph, CLI and original qualification remain open.
  [Scope](docs/lfm3-attention-bf16.md),
  [review/commit](artifacts/lfm3-attention-bf16-2026-11-11/publication-checks.json).

- [x] S1.L BF16 CPU projections: exact source matches for 94,248 learned
  outputs across four selected rows of all 154 rank-two tensors, 65,280 finite
  controls and 739 finite corner words/three rejection markers. Original
  pre-store FP32 diagnostic passes (maximum 8.19e-6). Fresh fmt/both strict
  Clippy/723 CPU/869 Metal/nine example, eight release source tests, native77
  three 46-export slices/strict links/C smokes pass. Debug/release projections
  allocate nothing; four portable tests in both modes, six source and ten
  compiled admission controls pass. All 42 release component cohorts pass;
  representative 128-token MLP calls take 802–810 ms. No model speedup claim.
  Actual MLX CPU/Metal exact component correspondence rejects. Preparation
  misses/capability-query failure are retained; original gates remain fixed.
  Astra approves scoped retention, no findings; attention/graph/model/session/
  CLI, full quality/fit/adoption remain open.
  [Scope/evidence](docs/lfm3-projection-bf16.md),
  [measurements](benchmarks/results/2026-11-11-lfm3-projection-bf16-v1.json),
  [review](artifacts/lfm3-projection-bf16-2026-11-11/astra-review-result-1.json).

- [x] S1.L BF16 state ownership: qualify six-row convolution history and
  256-token/full KV views, bit-preserving storage, abort/retry and all-owner
  publication before graph composition. Implementation and independent fixtures
  are present; actual declared-parent and MLX CPU/Metal logical-state comparisons
  pass. The isolated authenticated Transformers 5.20.2 reference preserves the
  installed environment; initial TLS/runtime preparation failures are retained.
  Fresh fmt/both strict Clippy/714 CPU/860 Metal/nine example checks and native76
  three 46-export slices/strict links and CPU/Metal C execution smokes pass.
  Eight release state tests and debug/release zero-allocation checks pass.
  Four portable tests in both modes and 34 compiled CLI negatives pass. All
  24 original FP32 5% compatibility pairs and 18 BF16 release cohorts pass;
  state medians 0.338–0.365 / 7.119–7.254 / 27.706–28.644 ms at 1/32/128 tokens.
  State owns 28,108,984 B at capacity 2,048; no model throughput/fit claim.
  Astra approves scoped retention, no findings; full graph/CLI/quality/model
  fit/adoption and original gates remain open/unchanged.
  [Scope/evidence](docs/lfm3-bf16-state.md),
  [measurements](benchmarks/results/2026-11-11-lfm3-bf16-state-v1.json),
  [review](artifacts/lfm3-bf16-state-2026-11-11/astra-review-result-1.json).

- [x] S1.L BF16 binary/convolution boundaries: 13 learned banks/104 cases/
  8,146,944 convolution words, 34 binary grids and head broadcasting match
  the canonical source exactly; original FP32 diagnostic and chunk splits pass.
  Fresh fmt/both strict Clippy/705 CPU/851 Metal/nine example, five release
  source tests and native75 three 46-export slices/C smokes pass. All four
  operators have zero Rust allocations. Six release component cohorts pass:
  1.150–1.168 / 33.547–33.923 / 135.099–135.660 ms at 1/32/128 tokens;
  operator baselines only. Initial collector/cache-path failures are retained
  and corrected; portable/admission controls pass. Actual child CPU/Metal exact
  correspondence rejects. Astra approves scoped retention, no findings. BF16
  history/window/graph, model fit/quality/adoption and all original gates remain
  open/unchanged.
  [Scope/evidence](docs/lfm3-binary-conv-source.md),
  [review/commit](artifacts/lfm3-conv-bf16-2026-11-11/publication-checks.json).

- [x] S1.L CPU BF16 norms/unary gates: all 119 authenticated gains,
  238 cases/4,082,688 outputs, 5,706 FP32 statistic rows, 16 epsilon/zero
  controls and both 65,280-word finite unary grids match exactly. Original
  mathematical gate passes; source reduction/cast order stays explicit.
  Actual child CPU/Metal component correspondence rejects and remains separate
  from the canonical target. Fresh fmt/both strict Clippy/699 CPU/845 Metal/
  nine example, five release source tests and native73 three 46-export slices/
  CPU+Metal C smokes pass. Both widths have zero Rust allocations. Six release
  component cohorts pass: combined medians 1.285–1.315 / 39.048–39.207 /
  152.448–154.071 ms for 1/32/128 tokens; operator baselines only. Four portable
  tests in both modes, six source and ten console admission controls pass.
  Astra approves scoped retention, no findings; 6 GiB and all quality/caps/
  default gates remain fixed.
  Full LFM3 execution/fit/quality/adoption and remaining BF16 products/conv/
  window composition stay open.
  [Scope/evidence](docs/lfm3-bf16-source.md),
  [review/commit](artifacts/lfm3-bf16-2026-11-10/publication-checks.json).

- [x] Fifth user-requested disk recovery: removed 520 inactive Cargo test
  executables, confirmed by Cargo test fingerprints and Rust harness symbols.
  Recovered 14.4 GiB net; 87.2 GiB free at completion. All 44,970 protected
  hashes and 494,201 retained-cache metadata checks pass. Models, raw evidence,
  production binaries, benchmark examples, current caches and user work remain
  intact. Engine implementation and qualification are unchanged.
  [Checks/review/commit](artifacts/lfm3-rotary-2026-11-10/disk-cleanup-5/publication-checks.json),
  [deletion receipt](artifacts/lfm3-rotary-2026-11-10/disk-cleanup-5/result.json).

- [x] S1.L source-profile BF16 rotary: exact frequencies/525,312 trig words
  and 460,800 source outputs pass; source-limit sparse rows and chunk schedules
  agree. Both profiles have zero steady-state allocations; two shared owners
  total 1,049,152 B. Fresh fmt/both strict Clippy/693 CPU/839 Metal/nine example,
  five release source tests and native72 three 46-export slices/C smokes pass.
  Eighteen frozen M1 Pro release cohorts pass: eleven-layer Q/K medians span
  0.0665–0.0703 / 2.112–2.226 / 8.437–8.695 ms for chunks 1/32/128; operator
  baselines only. Four portable tests in both modes, six source and ten console
  controls pass. Source/runtime gates and 6 GiB remain fixed. Astra approves
  scoped retention, no findings; whole LFM3 graph/quality/fit/adoption remain open.
  [Scope/evidence](docs/lfm3-rotary-source.md),
  [review/commit](artifacts/lfm3-rotary-2026-11-10/publication-checks.json).

- [x] S1.L seven-tap FP32 history: shared static three/seven-tap state and
  concrete compatibility APIs reuse the unchanged scalar leaf. Five independent
  fixtures/132 tensor outputs match bitwise; 288 Rust schedules and both zero-
  allocation profiles pass. Fresh fmt/both strict Clippy/688 CPU/834 Metal/nine
  example checks, two explicit Conv3 GPU regressions and native71 three 46-export
  slices/CPU+Metal C smokes pass. The first generic prepare boundary fails its
  three-tap timing screen and is archived/rejected. Concrete engine-crate
  entrypoints pass all six original 5% comparisons (−2.33% to +1.26%); new seven-
  tap medians are 0.357/9.2–9.4/36.7–38.1 ms for 13 owners at width 2,048 and
  chunks 1/32/128. Source math, original gates/caps and 6 GiB remain unchanged.
  Initial compile/harness/lock/sandbox failures and their corrections are retained.
  Astra approves scoped retention with no findings; full LFM3/BF16 activation/
  quality/fit/adoption remain open.
  [Scope/evidence](docs/lfm3-seven-tap-history.md),
  [review/commit](artifacts/lfm3-seven-tap-2026-11-10/publication-checks.json).

- [x] S1.L LFM3 BF16 source audit: complete child/parent hashes and all 286
  tensors/1,728,801,536 parameters match under 13 singleton convolution-axis
  transforms, including 119 learned norms. Exact declared config/tokenizer links
  pass. Parent bytes are discarded; the child is independently rehashed across
  1,014 blocks. Nineteen portable tests and 28 actual offline CLI rejections pass,
  including a changed byte in a temporary APFS clone, then removed with its
  reconstruction recipe retained. Both owned campaigns close within a separate
  512 MiB aggregate guard (247.6 MB maximum conservative final RSS sum); the
  engine remains at 6 GiB. Fresh fmt and identical 4,034 Cargo/421 native/ten
  package input/output hashes support required-check reuse. Astra approves
  scoped retention after its RSS-consistency finding was fixed: final per-PID peaks now
  cover that PID's sampled RSS, with a complete replay and coherent negatives.
  The initial closure schema preflight failed before writes and was corrected.
  This authenticates repackaging only; no graph, codec, numerical,
  memory-fit, quality, performance or adoption qualification follows.
  [Scope/next work](docs/lfm3-source-qualification.md),
  [review/commit](artifacts/lfm3-source-2026-11-10/publication-checks.json),
  [parent audit](artifacts/lfm3-source-2026-11-10/parent-checks-1.json),
  [offline replay](artifacts/lfm3-source-2026-11-10/offline-checks-2.json),
  [negative controls](artifacts/lfm3-source-2026-11-10/negative-checks-2.json).

- [x] S1.B native precision audit: all 442,863,360 paired CPU coordinates admit a
  finite original-domain output; zero discrete/continuous exclusions. A fresh
  complete replay agrees, and a different midpoint/neighbor/source algorithm
  verifies concrete native witnesses for every coordinate. No graph qualifies
  from this feasibility result; the prior 82 native failures remain binding.
  Preserve the current managed-adoption rejection and advance to LFM3 source
  qualification. Eight new/25 total portable tests, all 63,488 finite Half inputs
  with adjacent peers/271 scalar controls, 11 CLI negatives and one actual source-
  byte rejection pass. Early output-path rejection adds no numerical change,
  proven by AST; historical source snapshots and a corrected negative-runner
  error-text expectation are retained. Integration preflight caught an unrelated
  history-formatting overlap; original user-patch context was restored before
  changing root work. Three closed CPU-only campaigns stay
  below 512 MiB (maximum 172,408,832 B final RSS); 6 GiB engine/codec/quality policies
  stay fixed. Required Rust/native inputs are unchanged; Astra approves scoped
  diagnostic retention, no findings.
  [Scope/evidence](docs/minicpm5-precision-audit.md),
  [review/commit](artifacts/minicpm5-representability-2026-11-10/publication-checks.json).

- [x] Fourth user-requested disk recovery: removed 7,919 inactive Cargo test
  executables, recovering 22.5 GiB net and leaving 80.0 GiB free at completion.
  All 22,464 protected hashes and 494,721 retained-cache metadata checks pass.
  Benchmark examples, engine outputs, current caches, models, raw evidence and
  user work are preserved. This cleanup changes no engine implementation;
  LFM3 qualification remains the current task.
  [Receipt](artifacts/lfm3-source-2026-11-10/disk-cleanup-4/result.json).

- [x] Third user-requested disk recovery: removed 23,292 inactive Cargo
  support/intermediate files, recovering 5.3 GiB net; 65.8 GiB free at completion.
  All 4,054 protected hashes and 502,723 retained-cache metadata
  checks pass. Models, evidence, engine binaries/libraries, current caches and
  user work remain intact. Process-inspection/schema preflight failures were
  corrected before deletion; receipts retain both. Subsequent research decisions are
  recorded below; the failed candidate implementation stays excluded.
  [Receipt](artifacts/minicpm5-projection-isolation-2026-11-10/disk-cleanup-3/result.json).

- [x] S1.B numerical isolation and rejected rotary experiment: eight sampled
  projection coordinates reproduce exact rational dots of their own inputs.
  Five of seven traced token rows first diverge from FP64 at layer0 rotary;
  two first differ at input norm. A model-owned coefficient table matches all
  7,084,800 same-input primitive oracle values, but the complete original short
  suite rejects six of 52,896,768 comparisons. Preserve that rejection; partial
  deeper capture was deliberately stopped, with no complete 1,025 qualification.
  Candidate fmt/both strict Clippy/683 CPU/831 Metal/9 example, two explicit GPU
  tests, 36 portable, native69 three 46-export slices and both C smokes pass.
  Six raw witnesses independently replay; 12 resealed owner/metadata copies reject.
  Candidate engine/CLI/reader code is archived; only research/decision documents
  were retained. Required clocks/resource receipts and preparation
  failures remain linked; no performance/quality/adoption grant. Astra approves
  documentation-only retention, with no findings; candidate code stays excluded.
  [Scope/evidence](docs/minicpm5-rotary-rejection.md),
  [review/commit](artifacts/minicpm5-projection-isolation-2026-11-10/publication-checks.json).

- [x] S1.B deeper numerical diagnostic: frozen 128/512/1,025-token prefixes from
  the complete quality payload, both independent CPU profiles and native chunks
  1/32. All 51 original six-case error gates stay unchanged; no recalibration
  from deeper outcomes. Keep the original CPU implementation and ordinary
  512/chunk32 route intact. Separate diagnostic domain only; preserve 6 GiB,
  full quality workload/caps, 2 GiB disk floor and deferred UI/device scope.
  CPU capture passes: 306 arrays/2.641 GB, 2.029 GB final RSS, six portable tests;
  only the original bound changes, proven by AST identity. Fresh fmt/both strict
  Clippy/681 CPU/827 Metal tests and nine example tests pass. Release CLI and
  native68 three 46-export slices plus CPU/Metal C smokes pass.
  First native attempt rejects at the existing ordinary512 geometry guard before
  GPU loading. Its erroneous constant capture-completed flag is retained and
  explicitly corrected by interpretation; passed=false is authoritative. The
  separate fixed-geometry affine8/BF16 diagnostic loader admits through1,025,
  without changing ordinary admission or gates. Full capture/state controls pass:
  87,624 arrays/1,718 chunks, 2.291 GB final RSS and 2.827 GB sampled lifetime
  physical peak. Both complete cold comparisons reproduce 82 failures in 1.773B
  checks; 442.9M CPU profile coordinates have no disjoint bounds. Preserve this
  numerical rejection; isolate a projection mismatch next. Seventeen portable,
  15 CLI and 51 saved-metadata/sample negatives pass; 48 raw witnesses replay
  independently. Astra's two clock/ledger findings are fixed in the reader; 13
  resealed datasets and all 1,718 zero-ledger/1,717 repeated-window controls reject.
  Both corrected full replays preserve the numerical result. Runner errors and disk cleanup are retained
  in the scope report. Astra approves scoped retention, both findings resolved;
  quality/adoption remain open.
  [Failed attempt](artifacts/minicpm5-context1025-2026-11-10/native-attempt-1-interpretation.json).
  [Frozen contract](artifacts/minicpm5-context1025-2026-11-10/prospective-contract-1.json).
  [Results and next work](docs/minicpm5-context1025.md).
  [Review/commit](artifacts/minicpm5-context1025-2026-11-10/publication-checks.json).

- [x] S1.B full quality inputs: freeze all original 10,000 loss targets and
  40 code/30 prose/30 continuation cases with unchanged output caps and gates.
  Official stock metadata is pinned at `42433e90`; no stock weights downloaded.
  Wrapper/padding and template tool branches differ, but complete corpus IDs and
  all 100 task renders/IDs match exactly. All 10,000 targets/100 tasks prepare
  under a 3,643-file admitted parser runtime; 546 MB final RSS, no model execution.
  Full tasks need up to 559 positions and loss blocks 1,025; current capacity
  stays 512. Seven portable tests, complete cold source replay and 28 altered-
  input rejections pass, including eight coherent source/self-hash mutations.
  Manifest/token bytes are versioned; raw tasks/prefix remain in artifacts.
  Fresh fmt and exact 4,031 Cargo/421 native input/10 package hash reuse pass.
  Astra independently replays all payload hashes and approves scoped retention,
  no findings; full numerical/quality/adoption remain open.
  [Scope/evidence](docs/minicpm5-quality-inputs.md),
  [review/commit](artifacts/minicpm5-quality-inputs-2026-11-10/publication-checks.json).

- [x] S1.B MiniCPM5 matrix reproduction: all 296 matrices/888 affine arrays and
  85 copied norms match exactly across 4,128 bounded tiles. Both complete BF16
  objects authenticate (5,033,557,128 bytes); zero mismatches and no retained
  parent payload. Frozen GPU/BF16 MLX 0.32.3 recipe, 12 literal controls and
  three prior range controls pass; publisher's exact binary remains unknown.
  Separate 512 MiB audit allowance passes: 322,764,800-byte conservative final
  RSS sum, 4,035 observations; final physical peak unavailable. Engine/codec/
  disk limits unchanged. Byte-exact fresh witness seals, 35 portable tests,
  34 actual-output rejections and two explicit structural acceptances pass.
  Fresh fmt and unchanged 4,031 Cargo/421 native inputs/10 package outputs pass.
  Astra independently replays the seal and approves scoped retention, with no
  findings. Quality, deeper numerical, default/adoption and overhead gates stay
  open. Post-integration patch-simulation check corrected; all protected user
  work is verified unchanged. [Scope/evidence](docs/minicpm5-matrix-lineage.md),
  [review/commit](artifacts/minicpm5-matrix-lineage-2026-11-10/publication-checks.json).

- [x] S1.B MiniCPM5 bounded text codec and diagnostic CLI: exact original BOS,
  assistant LF/empty-think/padding semantics and stop union;98independent source
  records/66stream schedules pass. Preserve512MiBcodec/4MiBstream/6GiBengine and
  512/32 bounds; no screen capture or adoption claim. Source assumptions and
  ASCII decode-bound cache-stress failure are archived/fixed without reducing
  original witnesses,660-phase stress or text limits. Corrected-source fresh
  fmt/both strict Clippy/679CPU/821Metal and native67three46-export slices/C
  smokes pass;36portable tests pass after the Astra causal-window fix. Final98-record fidelity,
  both13-construction campaigns and unchanged660-phase/22k-word cache stress pass
  (maximum215MB RSS). Both14/38-token text histories pass4,177,920complete-logit
  native comparisons with exact tokens/text/state. Eight cohorts/80measured+
  24warm requests clear the5%screen:18.85%/18.32% and12.29%/11.79%request gains;
  83actual-report negatives and offline replay pass. Astra's sole first-text
  enclosure finding is fixed in the reader;6old-accepted causal copies now reject.
  Producer/native/raw evidence is unchanged. Root protection passes after a
  retained stash-format inspection typo. Astra approves scoped retention;
  quality/default/adoption/1%gates stay open.
  [Review/commit](artifacts/minicpm5-text-codec-2026-11-10/publication-checks.json).
  [Scope/current evidence](docs/minicpm5-text-codec.md),
  [contract](artifacts/minicpm5-text-codec-2026-11-10/prospective-contract.json).

- [x] Second user-requested disk recovery: removed 1,809,290 nonexecutable
  compiler/intermediate files from 190 inactive Cargo caches; recovered about
  51.1 GiB net, leaving 68.0 GiB available after compacting the cleanup manifests.
  All 4,054 protected hashes, 521,828 preserved-cache metadata records, root work,
  refs, stashes and app link pass. Models, raw evidence, binaries/libraries,
  current caches and the running numerical capture are preserved. A slow cleanup
  controller was interrupted, then its exact remaining manifest resumed; receipts
  retain that interruption. Documentation-only checks apply; code review remains
  recorded for the numerical task above.
  [Receipt](artifacts/minicpm5-context1025-2026-11-10/disk-cleanup-2/result.json),
  [compressed manifests](artifacts/minicpm5-context1025-2026-11-10/disk-cleanup-2/manifest-storage.json).

- [x] Earlier user-requested disk recovery: removed323,538nonexecutable intermediate/
  incremental files from7completed-build caches; free space2.0→29.1GiB.
  Kept current build caches, executables, native libraries, models and evidence.
  All4,031task inputs/28protected hashes pass; root user work/app link preserved.
  [Receipt](artifacts/minicpm5-text-codec-2026-11-10/disk-cleanup-2026-11-10/result.json).

- [x] S1.B parent identity: bounded whole-object authentication of both pinned
  BF16 shards, per-tensor/block hashes and exact unquantized norm parity. Stream
  payloads without retaining a5GB parent file; preserve2GiB disk/6GiB engine limits.
  Both full hashes/5,033,557,128bytes,381tensors and85copied norms pass;1,349block
  digests retained, no payload. Fifteen portable tests,27structural negatives and
  valid-block mutation/seal detection pass; fresh seal reproduces exactly. All4023
  required/419native inputs are unchanged;670CPU/811Metal/native65 reused by hashes.
  Source-audit resource guard passes; finalphysicalpeak unavailable. Astra approves.
  Quantized conversion lineage, codec/task quality and adoption stay open.
  [Scope](docs/minicpm5-parent-auth.md),
  [review/commit](artifacts/minicpm5-parent-auth-2026-11-10/publication-checks.json).

- [x] Half RMS parallel stores: explicit bounded MiniCPM5 Mac path, exact scalar
  reduction and 2,614 bitwise-equal original arrays; 52.9M dual-profile comparisons pass.
  Fresh fmt/both strict Clippy, 670 CPU/811 Metal tests, two explicit GPU parity/
  state/dispatch tests, native65 three46-export slices/C smokes, 34 portable tests
  and 49 altered-report negatives pass. Six eligible cohorts/60 measured requests
  reduce warm request time21.51%/21.50%, TTFT14% and raise decode35%; fallback
  within0.14%. Extra64MiB allowance explicit; observed memory stays below6GiB
  in named scopes, final physical peak unavailable. Disk stops/cache recovery,
  closure compile error and portable-controller typo are retained/fixed.
  Astra approves explicit retention; defaults/adoption gates stay fixed.
  [Scope](docs/minicpm5-rms-store.md),
  [review/commit](artifacts/minicpm5-rms-store-2026-11-10/publication-checks.json).

- [x] MiniCPM5 CLI GPU attribution: expose existing bounded counters with exact
  ordinary/profiled logit and state parity, two controlled release cohorts and
  strict offline clock/coverage/memory reporting. Preserve the ordinary baseline
  and all model/adoption gates; no screen sharing. New checkout briefly crossed
  the disk floor; byte-identical APFS clones and verified inactive compiler-cache
  recovery restored headroom. Fresh fmt/both strict Clippy, five example tests,
  eleven portable tests and 58 resealed negatives pass; unchanged core/native
  checks are reused by input hashes. Two cohorts pass 104 pairs/74,672 intervals/
  13.58M logit comparisons. Profiler inflates phase clocks 11.5–19.7%; no ordinary
  overhead or optimization claim. Prefill projections and decode normalization
  guide the next exact-reduction/parallel-store RMS hypothesis. Astra counter
  enclosure finding is fixed; saved results revalidate. Astra approves scoped
  retention. [Review/commit](artifacts/minicpm5-profile-2026-11-10/publication-checks.json).
  [Scope/results](docs/minicpm5-profile.md).

- [x] S1.B short console vertical: authenticated 2.674 GB Heretic MLX8 object;
  parent headers match 381 BF16 tensors, whole-parent/lineage still open. The
  698 CPU arrays and 52,896,768 dual-profile native comparisons pass with zero
  failures/max gate fraction 0.97221; exact schedules, causal prefix and 512-token
  functional state pass. Fresh fmt/both strict Clippy, 670 CPU/810 Metal tests,
  native64 three slices/46 exports each and CPU/Metal C smokes pass. Portable
  7 numerical/10 capture-reader/11 request-reader tests and 32/41/39 resealed
  negative controls pass. Forty measured requests plus 12 warmups close four
  eligible cohorts: token-only readback is
  0.64–0.70% slower; reject a performance/default change. Memory observations
  stay below 6 GiB in named scopes; final physical peaks remain unavailable.
  Failed invocations, toolchain mismatch, disk admission/floor observation and
  verified cache recoveries are retained. Astra timing-enclosure and balanced
  owner-ledger findings are fixed in consumers; Astra approves scoped retention.
  [Review/commit](artifacts/minicpm5-heretic8-payload-2026-11-10/publication-checks.json).
  Next: attribute the dominant GPU work before one bounded CLI experiment.
  Whole-parent/lineage and original codec/quality/deeper-numerical gates stay open.
  No model adoption, visible-text timing or 1% overhead qualification.
  [Scope and all evidence](docs/minicpm5-console.md).

- [x] S1.A: explicit default-RoPE declarations for the existing Llama graph;
  equivalent aliases preserve kernels/state/budgets and cold CLI reports.
  Required fmt/both strict Clippy, 670 CPU/806 Metal tests, five targeted GPU
  tests, three positive/four negative CLI controls and three native slices with
  46 exports each plus CPU/Metal C smokes pass. Six metadata cohorts retain
  60 measured/18 warm batches, after medians 2.107–2.145 ms/open; no repeated
  ≥5% legacy regression, but unchanged-before-binary drift was 36.6%.
  Astra approves after the timing-envelope fix: 10 portable tests and all six
  unchanged cohorts revalidate. No model execution/codec/quality/fit/adoption
  or optimization qualification. Original source-access/probe failures preserved.
  [Scope](docs/minicpm5-default-rope.md),
  [checks/review/commit](artifacts/minicpm5-default-rope-2026-11-10/publication-checks.json).

- [x] Q2.G9B128/256 evaluation retained as a rejection in `8215d96a`:
  both79.35M-value captures reject45,332 values;14,847 saved coordinates have
  disjoint original profile intervals. Functional/recovery, required667CPU/
  803Metal,190Python and two performance runs pass in their scoped senses.
  First256 cohort is noisy; no optimization/adoption claim. Astra approves
  diagnostic retention; feature-integrated, no push/main change, rootWIP kept.
  [Full scope/failed attempts](docs/qwen35-9b-context256.md),
  [checks/review/commit](artifacts/qwen35-9b-context256-2026-11-09/publication-checks.json).

- [x] Refresh completed working-document history from reviewed `88950b11`;
  exact inverse-link reconstruction, local link/anchor and whitespace checks
  pass. Routine TODO/improvement bytes fall about 56%; live PLAN, 15 reference
  producers, 3,967 Cargo/native inputs and root user work remain unchanged.
  [Archive](docs/archive/2026-11-09/README.md),
  [validation](artifacts/qwen35-9b-context256-2026-11-09/working-history-checks.json).

## Completed work index

Full check counts, failed attempts, limitations, reviews and commit references
are preserved in the [reviewed history](docs/archive/2026-11-09/TODO.md). Use the
linked scope reports for individual results; a passed narrow gate never expands
model, context, device or adoption qualification.

| Completed scope | Current decision and evidence |
| --- | --- |
| O1, O2.A, O2.B1, O2.B2.A | Bounded summaries, native phases, repeated numerical workloads and controller publication delivered; summary mode stays opt-in because the 1% overhead bound is unqualified. [O1](docs/o1-gemma-request-summaries.md), [phases](docs/o2-gemma-cost-report.md), [workloads](docs/o2-gemma-workloads.md), [publication](docs/o2-ui-publication.md). |
| O3 and R1 | Scoped runtime explanation/profile and comparable offline regression screening delivered. [O3](docs/o3-gemma-runtime.md), [R1](docs/gemma-regression-job.md). |
| Console engines | Ollama/vLLM installed and 78 requests collected; unmatched weights/precision/cache prohibit engine ratios. [Comparison](docs/console-engine-comparison.md), `34db5778`. |
| Gemma guard and preferred MLX artifact | Exact guard optimization retained. Heretic affine4 candidate rejected on task quality; official BF16 remains selected. [Guard](docs/gemma-guard-validation.md), [artifact](docs/gemma-heretic-mlx.md). |
| Q1.A–I: Qwen3.5-4B mechanisms | Pinned source, bounded owners, scalar/hybrid kernels, codec and generation bridge delivered only in their recorded scopes. [Model plan](docs/model-observability-plan.md). |
| Q1.J: 4B 128/256 | Rejected: 13,505 failures and 3,132 disjoint dual-profile intervals; matching greedy IDs do not qualify adoption. [Decision](artifacts/qwen35-context256-2026-11-09/decision.json), `9d270f3a`. |
| Q2.A–D: separate 9B support | Pinned recipe/store, distinct head and independent one-input oracle delivered without loosening 4B APIs or gates. [Recipe](docs/qwen35-9b-cold.md), [store](docs/qwen35-9b-storage.md), [head](docs/qwen35-untied-head.md), [oracle](docs/qwen35-9b-oracle.md). |
| Q2.E: 9B hybrid Metal | Both one-input captures pass; 64.72×/64.42× applies only to the raw one-input baseline. [Scope](docs/qwen35-9b-metal.md), `3ccce99f`. |
| Q2.F: 9B context17 | Both 33.85M-value gates and prefix/repair/zero-allocation witnesses pass; descriptive release medians 4.093/4.121s. [Scope](docs/qwen35-9b-context17.md), `41287c6c`. |
| Q2.G0: exact reference helper | All 67.17M short-context profile values bitwise match scalar references. Helper gains are separate from engine speed or deeper-context qualification. [Scope](docs/qwen35-reference-vector.md), `88950b11`. |
| Five under-4B candidates | Metadata research complete; payload, new graphs, numerical/quality/fit/request and adoption gates remain open. [Selected order and evidence](docs/model-observability-plan.md#five-candidates-strictly-below-4b-total-parameters). |

## Remaining observability gates

- [ ] Display-frame qualification **deferred by user, engine/CLI priority**.
  Ready9 reuses checked43source/native46 inputs and completes13exact trials;
  fresh viewport/crop visually inspected, new receipt bound to its own UUID.
  Coverage rejects433/1721future-display callbacks, zero occlusion;32numeric
  samples retained. The owned session is closed; no further sharing requests.
  Earlier7/8failures and isolated UI WIP remain preserved/unreviewed/uncommitted.
  [Deferred status](artifacts/o2-ui-frames-2026-11-07/status-audit-1/current-state.json).
- [ ] Keep first-visible/final-answer and ordinary summary-overhead qualification
  explicit. Controller publication is not rendering; fixed-answer frame matches
  provide upper bounds only. iPhone work remains deferred.

## Working method

- Update current/next work, checkboxes, blockers and check/review outcomes after
  every iteration, including research, failed checks and interrupted work.
- Keep entries concise; link full logs in artifacts and evidence/commit records.
- Code completion requires applicable checks and one combined Astra review of
  diff, invariants and results before commit. Keep [improvements](docs/improvements.md)
  current. Documentation-only work needs link/content and whitespace checks.
- Read [PLAN.md](PLAN.md) for binding criteria, the
  [design rationale](apple_llm_performance_plan.md) when relevant, and targeted
  source before assuming a feature exists. Read archives only on demand.
- Preserve historical failures and scope limits. Mac observations do not qualify
  iPhone; planning/admission estimates do not establish allocation or execution.

## Current delivery and retained decisions

| Area | Current scope and evidence |
| --- | --- |
| Foundation M0–M7 | Complete in recorded scopes; [plan](PLAN.md), [M7 closure](docs/experiments/2026-10-21-m7-release.md). Later iPhone work remains deferred. |
| M8 shared blocks, model breadth and capacity controls | Selected Mac engine/app scopes delivered; [coverage index](docs/model-coverage.md), [shared-block contract](docs/shared-block-model-plan.md), [historical checklist](docs/archive/2026-11-07/TODO.md#m8--architecture-breadth-and-usable-context-controls). No blanket family/full-window qualification. |
| Managed Mac journeys | Six earlier profile journeys reviewed; [report](docs/mac-hub-journeys.md). Qwen1.5B Chat/native41 complete; [evidence](benchmarks/results/2026-11-06-qwen15-mac-app-v1.json). |
| Official Gemma BF16 Mac | Full 10.28 GB public snapshot, six-file SHA, managed publication, Chat/Stop and network-denied reopen pass; [journey](benchmarks/results/2026-11-07-gemma-public-resume-mac-v1.json). Native45 Settings/context/KV checks pass; [evidence](benchmarks/results/2026-11-06-gemma-bf16-live-window-v1.json). |
| G2 prefill | Retained on official BF16 Mac route; both primary runs reduce prefill about 61% and request time about 28%; [evidence](benchmarks/results/2026-11-06-gemma-bf16-g2-retention-v1.json). |
| M9 affine tile | First delivery complete as pinned Mac opt-in, with full quality, state and repeated timing. Managed adoption remains separate; [decision](docs/m9-mac-affine-decision.md). |
| Memory reporting | Owner snapshots, physical KV ceiling in Preview/Apply and conditional load-process peaks complete; [owner](benchmarks/results/2026-11-02-session-owner-memory-v1.json), [KV](benchmarks/results/2026-11-03-kv-preview-apply-v1.json), [loading](benchmarks/results/2026-11-03-load-memory-checkpoints-v1.json). Planned/admitted, owned and measured memory remain distinct. |

## Open or deferred scope

- [ ] Record separate iPhone Settings runtime evidence.
- [ ] Independent physical iPhone14Pro journeys/Settings; latest connection query times out.

The user deferred this device work on 5 November. The paired iPhone 14 Pro is
unavailable; a different Pro Max is not a substitute.
[Availability evidence](artifacts/m8-phone-availability-2026-11-04/availability.json).

Full-Qwen3-window expansion and unselected research techniques remain deferred.
The [detailed plans](docs/todo-implementation-plans.md) preserve task-specific
acceptance; older queue statements and historical pending notes do not override
later accepted decisions. Use the [archived M8 closure record](docs/archive/2026-11-07/TODO.md#m8-closure)
for individual model/adoption limitations.

Preserve these rejected or restricted results:

- Gemma3 coupled-intermediate rejections remain failed under the recorded
  [conformance policy](docs/gemma3-native-conformance.md); full adoption is distinct.
- Qwen3 historical fidelity rejections remain separate from scoped native
  qualification; [native profile](docs/qwen3-native-profile.md).
- Llama Online and split-KV candidates remain rejected/deferred under unchanged
  gates; [Llama decision](docs/todo-implementation-plans.md#llama-online-correctness),
  [split-KV evidence](benchmarks/results/2026-11-05-m9-split-kv-session-v1.json).
- Gemma QAT full task quality fails the two-point loss limit; passing loss/wrap
  fidelity does not override it; [evidence](benchmarks/results/2026-11-06-gemma-qat-full-quality-v1.json).
- G3 assistant remains test-only after the longer guard regressed about 21%;
  [decision](benchmarks/results/2026-11-06-gemma-assistant-mtp-retention-v1.json).
- Prompt lookup remains an explicit nonstreaming API; default adoption rejected.
  [Plan and evidence](docs/todo-implementation-plans.md#p5-strict-greedy-verifier-and-prompt-lookup).

## Latest iteration

| Date | Result and checks | Evidence / next action |
| --- | --- | --- |
| 2026-11-08 | Exact guard reduction retained:25.0–26.6% primary/26.3% sustained request reduction;512 deeper bitwise pairs and626 sustained requests pass. Required CPU/Metal/GPU/native checks and Astra approval pass;6GiB preserved. | [Scope](docs/gemma-guard-validation.md), [checks/review/commit](artifacts/gemma-guard-validation-2026-11-08/publication-checks.json); next preferred Gemma MLX format/quality evaluation. |
| 2026-11-07 | O2.B2.A bounded UI publication complete: six real Gemma controls, 104 paired requests across UI/headless tracks, fresh strict Swift/app checks and 29 portable tests pass. Astra approves state/count join fix; timing captures recompute unchanged. Both 1% overhead bounds remain unqualified/default-disabled. | [Scope](docs/o2-ui-publication.md), [result](benchmarks/results/2026-11-07-o2-ui-publication-v1.json), [checks/commit](artifacts/o2-ui-publication-2026-11-07/checks.json); visible-frame/O3 remain open. |
| 2026-11-07 | O2.B1 numerical coverage complete: strict Swift/new Python and reused production gates pass; two batches plus 601 seconds sustained, max lifetime peak 4.924 GB. Astra approves. Earlier counter-assumption/interference run is superseded by fresh captures; no quality gate changed. | [Scope](docs/o2-gemma-workloads.md), [result](benchmarks/results/2026-11-07-o2b-gemma-workloads-v1.json), [checks/commit](artifacts/o2b-gemma-workloads-2026-11-07/checks.json); next UI/O3. |
| 2026-11-07 | R1 offline regression screening complete: comparable frozen identity/work, independent cohorts, explicit noisy/failure receipts and named portable CI. Local17 tests/Astra pass; no runtime or optimization qualification. O2.B1 corrected suite running. | [Job](docs/gemma-regression-job.md), [checks/commit](artifacts/r1-gemma-regression-2026-11-07/checks.json); next O2.B1 completion, then UI/O3. |
| 2026-11-07 | O2.A native46 API/schema2 phases and strict matched warm reports complete. Required checks and Astra approval pass after parser clock/enclosure fixes. Decode dominates this workload; overhead remains unqualified/default-disabled. | [Report](docs/o2-gemma-cost-report.md), [receipt](benchmarks/results/2026-11-07-o2a-gemma-phase-report-v1.json), [checks/commit](artifacts/o2-gemma-metrics-2026-11-07/checks.json); next O2.B. |
| 2026-11-07 | O1 Gemma opt-in summaries complete; Cargo, strict Swift/Xcode, real output/state/failure/admission controls pass. Corrected full-boundary overhead remains unqualified; Astra approves after three fixes. Earlier archive mismatch, compiler sandbox/type failures and superseded timings are retained in check notes/logs. | [Pilot](docs/o1-gemma-request-summaries.md), [result](benchmarks/results/2026-11-07-o1-gemma-summary-v1.json), [checks/commit](artifacts/o1-gemma-summary-2026-11-07/checks.json); next O2 within full-roadmap goal. |
| 2026-11-07 | Observability-first plan complete; five exact under-4B counts verified. Preferred uncensored MLX formats/lineage and conversion/memory gaps recorded. Initial metadata DNS/Python certificate failures resolved with system curl; receipts retained. Documentation checks pass; binding policy unchanged. No runtime checks/Astra review claimed. | [Plan](docs/model-observability-plan.md), [checks/commit](artifacts/model-observability-plan-2026-11-07/checks.json); next implementation O1 remains stopped. |
| 2026-11-07 | Researched ten current under-30B candidates from primary sources; capability/local-efficiency shortlist, not common-harness qualification. O1 stopped; unfinished code uncommitted and unchecked. Documentation checks pass. | [Research](docs/models-under-30b-2026-11-07.md), [checks](artifacts/models-under-30b-2026-11-07/checks.json); model selection next. |
| 2026-11-07 | Archived full TODO/PLAN/improvement snapshots at `338c4e5b`; routine document bytes reduced about 97%. All snapshot text, 32 milestone criteria, four binding policy sections and both open items preserved; link/content/whitespace checks pass. Documentation-only. | [Archive manifest](docs/archive/2026-11-07/manifest.json), [checks](artifacts/context-archive-2026-11-07/checks.json); next proposed O1 pilot. |

Older iterations and all completed checklists remain in the
[archived TODO](docs/archive/2026-11-07/TODO.md). Add only a concise recent record
here; archive completed history periodically rather than loading it by default.
