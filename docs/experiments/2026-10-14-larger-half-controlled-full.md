# Full larger-model compensated Half workload matrix

All eight frozen controlled workloads complete on the preselected Qwen2.5-1.5B
checkpoint with three warmups and ten measured requests each. The unchanged
qualified worker uses HalfCompensated projections, Original normalization,
FullLogits readback, capacity 2,048 and chunk 128. Exactly 128 teacher-forced IDs
are emitted, with EOS ignored. No numerical code, model, gate or default changes.

| Workload | Median TTFT, s | Median request, s | Request MAD, s | Median decode steps/s |
| --- | ---: | ---: | ---: | ---: |
| short | 2.037335896 | 28.398590 | 0.011764 | 4.817349 |
| medium | 9.233068459 | 75.446195 | 0.037574 | 1.917664 |
| long | 21.623262209 | 141.158441 | 0.062146 | 1.062459 |
| boundary | 51.331213646 | 264.684586 | 0.038155 | 0.595285 |
| append | 1.821654770 | 74.724694 | 0.016843 | 1.741866 |
| cached | 0.000000750 | 66.278119 | 0.007274 | 1.916168 |
| pending | 0.470977312 | 66.832540 | 0.018127 | 1.913733 |
| changed | 9.275988521 | 75.584861 | 0.003513 | 1.915350 |

Timers cover completed GPU evaluation and full last-logits readback. Token events
precede staging the next fixed ID; decode rates count 127 intervals. Sampling,
tokenization, text and first-visible-text latency are outside this track. Prefix
preparation is untimed; changed-prefix reset/replay is timed. Cached prefill
throughput is unset and its near-floor TTFT supports no percentage ranking.

There are 187 raw events, 104 completed requests, 24 warmups and 80 measured
samples. Complete histories, committed/pending counts, phase coverage, numerical
timing boundaries and persistent inventories replay exactly. The boundary worker
commits all 2,048 positions and rejects the next token without changing that prefix.
All eight workers exit 0 with empty cleanup errors and complete transports; the
collector exits 0. Initial/endpoint identities match. Current observations verify
the recorded collector and eight worker PIDs absent, all eight worker groups empty,
no controlled commands active and the machine stopped/not-starting. The recorded
live snapshots bind each successive worker; the driver group is not claimed as an
independently owned worker group.

Seven rows exceed ten minutes of measured request time in their individual repeated
request sequences. Their totals are medium 754.185183 s, long 1,411.587778 s,
boundary 2,646.948165 s, append 747.214432 s, cached 662.712967 s, pending
668.301719 s and changed 755.834220 s. Short totals 283.565480 s and does not
independently qualify that measured-duration threshold. There are no manual holds,
builds, tests, other owned models/VM execution or large hashes during numerical
requests. Lightweight read-only process observations are recorded during collection.

Start/end observations show AC attached, AC low-power mode zero and no reported
thermal/performance warning. These qualify duration and endpoint scope, not
continuous temperature/clocks, energy or full system memory pressure. Source
hashing warms storage; setup is not cold-storage qualification.

Maximum complete admission is 3,396,602,720 bytes; maximum sampled
worker-plus-driver native lifetime charge is 3,301,821,896 bytes. Both fit 6 GiB.
Weights remain 3,087,433,340 bytes, counted once. Complete KV/scratch/history/logits/
metadata/framework/application inventories remain fixed. Native counters are
charges, not complete shared physical allocation totals or an external memory rank.

The [full collection](../../benchmarks/results/2026-10-14-larger-half-controlled-full-v1.json),
[raw events](../../benchmarks/results/2026-10-14-larger-half-controlled-full-v1.raw.jsonl),
[stderr](../../benchmarks/results/2026-10-14-larger-half-controlled-full-v1.stderr),
[independent analysis](../../benchmarks/results/2026-10-14-larger-half-controlled-full-analysis-v1.json)
and [exact qualified analyzer](../../benchmarks/results/2026-10-14-half-controlled-full-analysis-v1.py.txt)
retain the completed data and byte-identical published copies. Replay checks the
exact bounded parsed bytes, fixed executable/source qualification and inventories,
every request/raw phase, warmup/sample label, state and memory observation, and
independently derives medians, MAD and rates. Six meaningful provenance/command/
setup-error/memory/peak/repetition mutations reject. No engine test/build rerun is
needed for this unchanged-source data task. Content/link and whitespace checks pass.

Dispatch parent is `f778bb4`, clean at launch. The accepted immutable worker remains
SHA-256 `31434813900abe5f2fd5cd9ed80d593bde95185cea06c2a3fddabe67f1278396`;
compiled snapshot/build identity is separate from dispatch. Hardware is Apple M1
Pro / MacBookPro18,3 / 16 GiB, macOS 26.6.2 (25G83), Rust 1.90.0. The fixed model
revision, source/tokenizer/contract hashes, precision, cache and flags are recorded.
The [qualification receipt](../../benchmarks/results/2026-10-14-larger-half-controlled-full-qualification-v1.json)
binds data, exact identities, idle evidence and protected CPU binaries. Astra accepts
the final evidence after byte-identical archived replay, all338bindings, arithmetic,
memory and saved process/machine observations, with no findings.

Together with the [small matrix](2026-10-14-small-half-controlled-full.md), this
completes both models' full controlled workload collections and stated sustained
duration scopes. Full application traces, confirming comparisons and the final
release decision remain open. Frozen Half full loss/task gates separately pass
both checkpoints. Both Q4 v1 full-loss failures still block the explicit M5 exit.
No external performance lead, default adoption or completed M5 claim follows.

Sequential reproduction requires native Metal access and fresh output:

```sh
.venv/bin/python tools/experiments/metal_controlled.py --binary artifacts/m5-controlled-native-binary-v1/metal_bench --model larger --output artifacts/half-larger-full-rerun.json
.venv/bin/python benchmarks/results/2026-10-14-half-controlled-full-analysis-v1.py.txt artifacts/half-larger-full-rerun.json > artifacts/half-larger-full-rerun-analysis.json
```
