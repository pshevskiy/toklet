# Native eight-bit MLX app performance on Mac and iPhone 14 Pro

22 October 2026; numerical/app base revision `783550a`. This collection measures the selected loss/task-qualified `mlx-community/Qwen2.5-0.5B-Instruct-8bit` snapshot at immutable revision `6474d4c1a05bd7bd634a9c993560e4e7272eda3d`. Execution is native Rust/Metal affine8/group64 with Half exclusions, FP16 activations/KV, compensated normalization, Online attention and FullLogits readback. No upstream MLX runtime executes these application trials. The [full loss/task gates](mlx8-full-task-quality.md) remain unchanged; the public four-bit rejection remains retained.

The optimized Mac app runner uses M1 Pro / MacBookPro18,3 / 16 GiB / macOS 26.6.2 (25G83), on AC. The signed Release iPhone app uses iPhone 14 Pro / iPhone15,2 / iOS 27.0.1 over the wired connection. Low Power Mode is off. Numerical/Swift/Cargo/package/signed-app inputs and protected CPU binaries remain unchanged and hash-bound.

## Workload and completed timings

Each primary and confirmation sequence uses the original frozen 128-, 512- or 1024-token prompt and exactly 128 generated raw IDs, capacity 1,152 and chunk 128. One loaded model/session remains resident per sequence; KV resets before each request. Three warmups precede ten measured requests. Artifact verification warms storage; load/session creation is separate and is not a cold-cache experiment. Across all sequences, observed load/session times range 1.4774–1.6118 seconds on Mac and 1.3545–1.4627 seconds on phone.

Native TTFT ends at the first sampled-ID event. Decode rate counts the 127 intervals between first and last native events. Caller completion waits for the entire native/coordinator request. Caller TTFT, all samples, min/max, median and linearly interpolated quartiles/IQR are retained in the qualification. Small sample distributions do not establish tail percentiles. Tokenization, chat formatting and visible text are outside these raw-ID measurements.

| Device | Run | Prompt IDs | Native TTFT median, s | Decode median, IDs/s | Caller completion median, s |
| --- | --- | ---: | ---: | ---: | ---: |
| M1 Pro | primary | 128 | 0.7467 | 31.8938 | 4.7286 |
| M1 Pro | primary | 512 | 3.1133 | 22.1772 | 8.8395 |
| M1 Pro | primary | 1024 | 6.5717 | 15.6567 | 14.6829 |
| M1 Pro | confirmation | 128 | 0.7465 | 31.9456 | 4.7219 |
| M1 Pro | confirmation | 512 | 3.1134 | 22.2205 | 8.8282 |
| M1 Pro | confirmation | 1024 | 6.5696 | 15.7181 | 14.6477 |
| iPhone 14 Pro | primary | 128 | 2.2126 | 20.4250 | 8.4308 |
| iPhone 14 Pro | primary | 512 | 13.3074 | 15.2318 | 21.6504 |
| iPhone 14 Pro | primary | 1024 | 30.3424 | 11.7477 | 41.1546 |
| iPhone 14 Pro | confirmation | 128 | 2.9915 | 17.7772 | 10.1453 |
| iPhone 14 Pro | confirmation | 512 | 15.1906 | 13.6681 | 24.4832 |
| iPhone 14 Pro | confirmation | 1024 | 38.6397 | 9.6400 | 51.8211 |

All twelve short sequences pass. Every repeated 128-ID response agrees within each run, with its confirmation and with the corresponding other-device response. Mac primary/confirmation median differences are at most 0.033% for native TTFT and 0.393% for decode. Phone median differences reach 35.203% for TTFT and 21.864% for decode while the sequence moves through nominal/fair/serious thermal states. There is no imposed cooldown or controlled thermal-cause experiment. These are descriptive timings, with no comparative speedup or constant-throughput acceptance.

## Sustained observations and memory

Dedicated 512/128 sequences stop at the first completed request satisfying both ten measured repetitions and 600 native seconds, excluding all three warmups. The ordered binary64 duration sum is replayed exactly. Mac completes 68 measured requests over 600.6852207929999 seconds; phone completes 26 over 628.915998376 seconds. Together with the short matrix, this is 256 completed requests/32,768native token events; all IDs, sampled/committed/pending counts and completed timing brackets replay.

| Device | Native TTFT median, s | Decode median, IDs/s | Caller completion median, s | First-five → last-five decode, IDs/s | Thermal states |
| --- | ---: | ---: | ---: | --- | --- |
| M1 Pro | 3.1136 | 22.1931 | 8.8362 | 22.3728 → 22.1792 (−0.866%) | nominal |
| iPhone 14 Pro | 14.7307 | 14.1524 | 23.7051 | 15.2324 → 11.8319 (−22.324%) | nominal, fair, serious |

The phone slowdown remains material; duration coverage does not establish a steady thermal rate. Maximum observed whole-process lifetime peaks across this matrix are 639,157,232 bytes on Mac and 651,888,720 bytes on phone. These include the native app process, are distinct from logical numerical admission and are not whole-system RAM or permanent iOS memory guarantees. No energy instrument or attributable joules/watts result is available.

## Evidence and retain decision

The [qualification](../benchmarks/results/2026-10-22-native-mlx8-performance-v2.json) and [raw inventory](../benchmarks/results/2026-10-22-native-mlx8-performance-raw-v2.json) bind 1,306 files, 427 raw files and 114 current source copies. The publisher independently regenerates all frozen recipes, binds exact native/prepared/ready/tiny-proof lineage, replays original copy/storage freshness even for reused no-copy launches, validates fourteen complete streams and recomputes every statistic. Strict raw process/signal replay checks request and response device/PID/executable/typed signal identity; final intervention cannot count as natural completion.

All 53 current portable methods pass (17 authored saved-data/mutation methods, 8 Mac measurement, 19 physical measurement and 9 job methods), alongside the already reviewed 2-method orchestration receipt: 55 methods/five current owned check commands. Mutation cases cover self-consistent workload mislabelling, missing/replaced proof/ready/native bindings, storage clocks/inventory, response identity, wrong signals, final interventions and outer-command order. These checks run only after all collector-owned command groups have exited.

All 68 outer hardware/Mac/condition commands and 154 phone transport commands pass: 138 devicectl operations plus 16 host boot queries. Raw replay proves 29 natural absence observations and 3 exact-owned preflight resets, with no final reset, unresolved cleanup or endpoint failure. Device identifiers, installed paths and original private command/failure ledgers stay local and hash-bound. The direct cooperative collector has no finite parent deadline that could kill remote cleanup.

The first publication is explicitly superseded and unaccepted: its live publisher stdout was inventoried empty before its final print changed that file. Its exact original sources, checks, qualification, inventory and final log are retained. The fresh v2 publisher captures output through the tool outside inventoried directories, binds and discloses that original mismatch, and repeats all 53 checks plus the full replay. No measurement or gate changed.

Retain this selected-checkpoint performance observation and strict saved-data replay mechanism. Physical MLX state/lifecycle qualification, broader native architecture families, chat-template/tokenizer compatibility, device-aware discovery/downloads and wider M8 acceptance remain open. Earlier Half/Q4 lifecycle evidence does not qualify the new MLX representation.

Astra accepted v2 and the documentation with no remaining blockers after independently verifying all 1,306 pins, 427 raw files, 114 source copies, 51 unchanged numerical/Swift/Cargo sources, fourteen complete sequences, all launch/storage/process/signal replays, 222 clean collection commands and all 55 current check methods. Statistics, privacy, 261 local links and whitespace pass. Acceptance covers selected-checkpoint repeated/sustained raw-ID observations only.
