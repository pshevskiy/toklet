# Larger Q4 v2 full matrix and short confirmation, 15 October 2026

The frozen larger group8 v2 configuration completes all eight original workloads and a separate short confirmation. Each row has three warmups and ten measured fixed-length128-output requests; the matrix has104requests, confirmation13. All workers exit0 with empty cleanup/endpoint errors, unchanged exact source/packed/input/helper/executable identities and clean collection revision `4ad5452836e6de052d5835d2b8315a5b0a18799e`.

AppleM1Pro/MacBookPro18,3/16GiB/macOS26.6.2(25G83), Rust1.90.0, optimized Metal build. Actual selection is Q4CompensatedTiled/Compensated/FullLogits, FP16 KV, capacity2048/chunk128. Four-bit codes plus group metadata use8effectivebits/weight before padding; exclusions stay Half. WorkerSHA is `2549a3381cb13ece96e3d7b7407079de9e30c55fd8ad41164ef4b2cfe30d7dfe`. The [worker qualification](2026-10-14-q4-v2-bench.md) and [geometry](2026-10-14-q4-v2-real-geometry.md) establish its source/artifact linkage. Loading/conversion are recorded outside warm request timing; storage has been warmed by identity checks, so no cold-storage claim follows.

| Workload | Median TTFT, s | Median request, s | Decode steps/s | Ten measured requests, s |
| --- | ---: | ---: | ---: | ---: |
| short | 2.269295229 | 31.277405917 | 4.377839 | 312.765312 |
| medium | 9.973801687 | 78.491203458 | 1.853052 | 785.324895 |
| long | 23.182646625 | 144.950974729 | 1.042985 | 1459.621069 |
| boundary | 55.029469896 | 278.982118250 | 0.567019 | 2766.920597 |
| append | 1.951125750 | 77.110844729 | 1.689537 | 770.872562 |
| cached | 0.000000667 | 68.515083896 | 1.853606 | 685.074986 |
| pending | 0.488773813 | 69.106323375 | 1.850821 | 691.051404 |
| changed | 9.980149562 | 78.514079583 | 1.853095 | 785.069095 |

Raw samples and TTFT ranges are retained; short-confirmation MAD is reported below. Other dispersion remains derivable from the raw samples. Ten observations do not qualify tail percentiles. Timing ends after completed GPU work/full last-logit readback. Controlled events omit tokenization, sampling and text. Decode rate counts127post-first-event intervals. Cached TTFT uses a retained distribution near timer floor; cached prefill throughput remains null and has no percentage ranking.

Cache preparation is outside request timing. Append adds64tokens after512committed; pending evaluates one sampled token not yet in KV; changed times explicit reset/replay. Exact fixed histories, reused/evaluated/pending counts and memory inventories are replayed. Boundary commits all2048positions and rejects another token without changing the valid prefix. The two native streams have202events, matched exactly to readiness and ordered request/response phases.

| Short confirmation metric | Initial median, s | Confirm median, s | Change | Initial MAD, s | Confirm MAD, s |
| --- | ---: | ---: | ---: | ---: | ---: |
| ttft_seconds | 2.269295229 | 2.193284896 | -3.349513% | 0.009228084 | 0.021867604 |
| total_seconds | 31.277405917 | 30.819728313 | -1.463285% | 0.081985292 | 0.024462896 |

This is repeatability of128/128requests, separate from the accepted paired512/1gain. Confirmation explicitly selects only `short` and cannot claim a second full matrix.

Complete admission is2,086,416,808bytes, including1,777,247,428resident weight bytes, KV, scratch, history/logits/native metadata and application allowance. Maximum sampled joint worker/driver native lifetime charge is2,010,909,336bytes; short confirmation reaches2,001,881,488bytes. Both are below6GiB. Native charge counters do not establish total system/shared physical allocation or system pressure.

The canonical repeated-request sequence totals10,742.769647seconds of completed requests including24warmups, or8,256.699921seconds measured-only. This exceeds the600second sustained requirement. Loading, cache preparation, hashes and checks are excluded from those sums; collection wall time11,656.629486seconds is distinct. Cache reset/prepare order is archived. Power endpoints report AC/lowpowermode0; battery100%→80% for the matrix and80%→80% for confirmation, without causal attribution. Endpoint thermal/performance warnings are absent; continuous temperatures/clocks/pressure and energy remain unmeasured.

The [fresh primary pairs](2026-10-14-q4-v2-primary.md) accept only the named tile-prefill refinement. The [affected-shape checks](2026-10-14-q4-v2-shapes.md) measure128reset andactual64cachedappend; all longer fixed prompts consist of128-row chunks. Forn<8 both modes use the same compensated vector kernel, grid, arguments and resources; attention/cache/norm/readback code is unchanged. This supports scoped regression analysis without treating historical Half timings as a matched vector baseline. Application coverage and final release synthesis are separate; no CPU/default replacement or external lead follows.

Reproduction: run `tools/experiments/metal_controlled_v2.py` with the immutable worker and a fresh output path, then repeat with `--workload short`. The [qualification receipt](../../benchmarks/results/2026-10-15-q4-v2-controlled-final-qualification-v1.json) binds complete inventories, original/retained compiled snapshots, pinned inputs/commands, same-byte exclusive publications, all raw events, result-schema arithmetic, native phase observations and links. No build/test/model/reference/hash job overlapped retained request timing.
