# Independent compensated Half primary repeats

Both fixed checkpoints repeat the frozen short128-input/128-teacher-output workload
with three warmups and ten measured requests. The explicit caller delegates to the
unchanged qualified controlled worker; this is a second independent candidate run,
not an alternating CPU/external comparison or a repeat of all eight matrix rows.
No numerical code, precision, source model, token arrays, cache semantics or default
changes. The primary metric remains warm TTFT with completed GPU work and full
last-logits readback; sampling/tokenization/text remain outside timing.

| Checkpoint | Initial / confirming median TTFT, s | TTFT change | Initial / confirming request, s | Request change |
| --- | ---: | ---: | ---: | ---: |
| Small |0.484317792 /0.4843344165|+0.003433%|12.192119521 /12.2027512915|+0.087202%|
| Larger |2.037335896 /2.0339035205|−0.168474%|28.3985897495 /28.4000374375|+0.005098%|

These shifts are descriptive repeatability, not new optimization gains. Larger TTFT
ranges1.856106–2.040837s initially and1.850053–2.039807s on confirmation; MADs
0.002139/0.005185s do not replace the full ranges. No cause for lower samples is
assigned. Ten requests do not establish tail percentiles or formal confidence.

All26completed requests/30raw events pass actual HalfCompensated/Original/
FullLogits mode, geometry, capacity2048/chunk128, complete historySHA and
255committed/onepending state, timing and persistent inventory replay. Each new
worker rejects the same1921+128admission before evaluation, exits0, finishes its
transport and has empty cleanup errors. Both collectors exit0; all input/source/
compiled-binary/caller/accepted-baseline endpoint identities match without errors.
Fresh current observations find no controlled worker/caller commands and a stopped/
not-starting machine. Historical worker/collector IDs were not retained; individual
PID/group absence is not reconstructed from command absence.

Complete admissions remain1,262,602,960/3,396,602,720bytes. Maximum sampled
worker-plus-driver lifetime charges are1,176,504,144/3,311,848,736bytes, below6GiB.
The caller retains the accepted full baseline JSON, adding driver state; native
charge differences from the first driver do not establish a kernel allocation
regression. Charges do not measure all shared physical allocations. Start/end
power/thermal observations are recorded; continuous sensors, energy and system
pressure remain unavailable. No tests/builds/other owned numerical/VM work or
large hashes overlap timed requests. Source hashing warms storage.

The exact caller bounds/hashes the same accepted baseline bytes it parses, checks
the full accepted inventory before launch, reuses fixed array/source admission,
records its own source identity and guards final observations. Astra clears that
source. Five meaningful mocked checks cover baseline replacement, changed initial
source and existing evidence rejection, and isolated endpoint failure after successful
or failed owned workers with unchanged identities. These are caller failure checks,
not substitute native numerical results. The separate actual native replay checks
literal request/response coverage,3+10labels, every memory observation/sample peak,
raw equality, qualified result schema and independently derives medians/MAD/changes.
No new Cargo check is needed for this artifact caller using unchanged engine code.

Dispatch parent`1ebf2c9`, clean at both launches. Immutable worker SHA-256 remains
`31434813900abe5f2fd5cd9ed80d593bde95185cea06c2a3fddabe67f1278396`;
compiled snapshot identity is separately qualified. Device is Apple M1 Pro /
MacBookPro18,3 /16GiB/macOS26.6.2(25G83)/Rust1.90.0. Exact checkpoint/tokenizer/
contract/compiler/build/precision/cache metadata is retained.

[Small confirmation](../../benchmarks/results/2026-10-14-small-half-primary-confirm-v1.json),
[small raw](../../benchmarks/results/2026-10-14-small-half-primary-confirm-v1.raw.jsonl),
[larger confirmation](../../benchmarks/results/2026-10-14-larger-half-primary-confirm-v1.json),
[larger raw](../../benchmarks/results/2026-10-14-larger-half-primary-confirm-v1.raw.jsonl),
[analysis](../../benchmarks/results/2026-10-14-half-primary-confirm-analysis-v1.json),
[exact caller](../../benchmarks/results/2026-10-14-half-primary-confirm-caller-v1.py.txt)
and [qualification receipt](../../benchmarks/results/2026-10-14-half-primary-confirm-qualification-v1.json)
bind the completed evidence and unchanged protected CPU binaries. Content/link/
whitespace checks pass. Astra accepts after independently verifying all87bindings,
raw/state/memory/schema replay, exact copies and statistics, with no findings.

This confirms the primary candidate row on both models. Existing full matrices and
sustained-duration rows remain separate. Application confirmation, final comparisons/
release decision and the unchanged Q4-blocked M5 exit remain open. No new paired
performance adoption, external lead, default change or quality policy follows.

Sequential native reproduction with fresh outputs:

```sh
.venv/bin/python benchmarks/results/2026-10-14-half-primary-confirm-caller-v1.py.txt small artifacts/half-small-primary-rerun.json
.venv/bin/python benchmarks/results/2026-10-14-half-primary-confirm-caller-v1.py.txt larger artifacts/half-larger-primary-rerun.json
```
