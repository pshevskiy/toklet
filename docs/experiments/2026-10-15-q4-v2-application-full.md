# Larger Q4 v2 full application and confirmation, 15 October 2026

Both frozen three-turn application collections complete three warmups and ten measured scenarios at output cap128:78turns total,60measured. Exact IDs, finalized text, stop reasons and committed/pending state match across all13scenarios in each collection and between collections. Natural exit0, post-exit independent decoding, complete source/executable/packed/tokenizer/transcript/parser endpoints and owned cleanup pass.

Use the pinned larger Qwen2.5-1.5B-Instruct/group8v2 artifact and immutable CLI SHA `52eb300beb625084971803c6875ddbda684a682f491037af3ac044a1d5c12d13`. Actual Q4CompensatedTiled/Compensated/FullLogits, FP16KV, capacity2048/chunk128 on AppleM1Pro/MacBookPro18,3/16GiB/macOS26.6.2(25G83), Rust1.90.0 release. Collection revision is `4ad5452836e6de052d5835d2b8315a5b0a18799e`. Larger [full loss](2026-10-14-q4-v2-full-loss.md) and [task quality](2026-10-14-q4-v2-task-quality.md) are pinned prerequisites; this application record does not rerun those gates. The [adapter qualification](2026-10-14-q4-v2-application.md) preserves legacy transport semantics.

| Run | Scenario median, s | MAD, s | Range, s | Ten measured scenarios, s | Maximum joint native charge, bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initial | 41.267054271 | 0.022353291 | 41.239048125–41.321157209 | 412.726713 | 2162575952 |
| Confirmation | 41.301725271 | 0.018905104 | 41.201197709–41.328575542 | 412.866698 | 2159086208 |

Median scenario time changes+0.084016%. These are same-configuration repeatability observations. Output counts/IDs can differ from Half or FP32: elapsed differences across representations do not isolate backend speed.

Each scripted turn starts empty with36/76/117prompt tokens; assistant/tool history is fixed authored input. Greedy generation stops on EOS or cap128; hidden stop tokens count as sampled, not visible. Reset flushes final text. Initial measured-turn medians:

| Turn | Prompt IDs | Visible IDs | Sampled IDs | Stop | First token receipt, s | First visible text, s | Finalized/reset receipt, s |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 1 | 36 | 86 | 87 | EOS151645 | 0.701828479 | 0.701828479 | 12.207133333 |
| 2 | 76 | 128 | 128 | Output limit | 1.361852188 | 1.361852188 | 24.658755896 |
| 3 | 117 | 14 | 15 | EOS151645 | 2.031827729 | 2.031827729 | 4.424563813 |

Prompt/output/state signatures are retained in the raw records, including sampled counts87/128/15, committed122/203/131 and pending1perturn. Driver timers start at submission and end at token, nonempty text, done and reset receipt. Native tokenization/formatting/numerical/sampling/detokenization/publication components overlap these boundaries and are not summed. Scenario time also includes driver validation and memory instrumentation. Full event-receipt vectors are unavailable; exact token clocks and separate-text phase bounds are verified, without claiming exact reconstruction of every receipt clock. Null absent token/text times remain valid.

Admission is2,267,080,568bytes, including1,777,247,428resident weight bytes, allocatedKV/scratch/history/logits/native metadata and tokenizer/application allowance. Both joint native lifetime charges in the table are below6GiB. Shared physical totals and system pressure are not inferred from charge counters. No redundant full CPU/GPU model is retained. Neither ten-measured-scenario sum reaches600seconds; sustained qualification comes from the [full controlled sequence](2026-10-15-q4-v2-controlled.md), not these application collections.

Start power is AC/lowpowermode0/battery80%; thermal endpoints have no recorded warnings. End-power readings, continuous thermals/clocks, energy and full system pressure are unavailable. Setup/hash/oracle decode are outside warm request timing; storage is already warmed. Both runs execute serially without builds/tests/other model/reference/hash work during retained measurements.

Reproduce with `tools/experiments/metal_application_v2.py --output FRESH_PATH` twice after the pinned prerequisites. The [qualification receipt](../../benchmarks/results/2026-10-15-q4-v2-application-final-qualification-v1.json) binds exact producer/compiled/source inventories and quality inputs, original requests, raw response/reset streams, independent post-exit decoding, native observations, receipt-time bounds and same-byte exclusive publications. Final M5 retain/default/baseline decisions are recorded separately.
