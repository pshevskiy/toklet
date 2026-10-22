# Shared MLX jobs and real Mac app-runner checks

[The data-only stager](../tools/experiments/m8_mlx_jobs.py) produces explicit version-two native MLX jobs for the Mac transport and the existing iOS app. [The Mac entry point](../apple/tests/swift/MLXRealMain.swift) authenticates the exact job SHA, rejects legacy/backend ambiguity, and delegates to the same measurement, quality, task and state coordinators used by the app. There is no tokenizer, model download, inference or conversion in the stager, and no new numerical code in this task.

The selected public snapshot and tiny mixed-precision fixture retain their [frozen reference identities](mlx-real-oracle.md). Model weights are hash-checked before immutable same-filesystem hard-link staging and at completion; jobs/payloads use exclusive output paths and exact inventories. A failed post-link endpoint mutation cannot publish a successful staged receipt. Prepared recipes include the unchanged 128/512/1024-token prompts, 3 warmups/10 repetitions, a 600-second sustained option, the complete 10,000-target corpus/block policy and all 100 original task IDs/caps/stops. Having a recipe does not qualify those full workloads.

Eight actual Mac jobs pass: tiny/public × short measurement, three-target scoring, task response collection and state component checks. Each measurement job completes 13 reset requests with four exact expected IDs; the public IDs are `[264,5458,315,17407]`. Quality rows preserve the original targets, committed positions and stable aggregation. Tiny tasks produce two complete expected responses; the public short case preserves its original ID and collects up to eight samples. All 26 state rows pass the existing complete prefix/scalar/rejection/reset/closure validator. This Mac state route explicitly requires no physical partial-prefill or OS-event witness.

The task stream validator now accepts an explicit Mac proof scope while retaining the default physical installation/foreground/battery requirements. Both scopes validate every chunk, callback clock, sampled/visible count, hidden EOS, case reset and complete response. Version-two MLX backend fields are explicit; no physical bundle proof or UIKit observation is invented for the Mac. Seventeen task methods, including 13 existing methods and four new Mac/MLX methods, pass. Eight staging methods and 15 existing state-validator methods also pass: 40 methods total.

Whole-process native peak physical charges in these fresh Mac processes are:

| Public-checkpoint job | Peak physical bytes |
| --- | ---: |
| Short measurement | 295,387,952 |
| Three-target scoring | 293,356,336 |
| First task, short cap | 384,779,104 |
| State component | 295,011,096 |

These are workload-specific Mach physical-footprint observations, including the lifetime peak, under the explicit 2.5 GiB process trial cap. Clean/shared pages and RSS have different accounting. Tiny peaks are approximately 7.9 MB. The short measurement validates routing and retains raw clocks; it does not establish representative performance, a speedup, or full held-out quality. Numerical admission is separate: tiny 256 MiB/public 1.5 GiB. Phone memory qualification cannot be inferred from Mac charge.

Strict Mac compilation, iOS source typechecking, four expected CLI rejections and all eight jobs pass in 19 owned commands: 15 successful commands and four exact expected exit-one/zero-output rejections. Native receipt parsing uses one bounded SHA-verified buffer; its full source/package closure, headers and archive remain guarded throughout. Bounded child logs are captured/pinned before result handling and the same consumed buffers are analyzed. Supplemental portable/Mac/iOS source checks complete in five earlier owned commands. All numerical/Cargo inputs and protected historical CPU binaries are unchanged.

Separately, phone preparation initially refused an existing unowned Harness process before build/install/launch. The unchanged reviewed M7 reset procedure verified its exact previous signed installation and fresh device/process identity, stopped only that test app and confirmed absence. A new 13-command preparation built, signed, verified and installed the current MLX-enabled Harness on iPhone 14 Pro/iOS27.0.1 using the existing signing setup. It launched no inference. Private signing/device logs remain local, digest-bound by redacted receipts. Actual transfer must still use fresh device-local storage admission and the app must execute before any phone MLX claim.

Receipts: [qualification](../benchmarks/results/2026-10-22-native-mlx-mac-jobs-v1.json), [raw inventory](../benchmarks/results/2026-10-22-native-mlx-mac-jobs-raw-v1.json). Retain the shared transport and selected short app checks. Full quality, representative and sustained performance, phone MLX execution/memory/state/lifecycle, chat-template compatibility, architecture expansion and discovery/download product work remain open.

Astra accepted the source and completed evidence with no blockers after independently replaying all eight saved streams and checking 430 provenance/output/operation pins, all 104 raw files, exact source correspondence, 40 portable methods, expected command outcomes, privacy and the stated qualification limits.
