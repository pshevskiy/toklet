# LFM2 bounded native Mac qualification

The pinned `mlx-community/LFM2-350M-4bit` revision
`18dc72abf3b2337f9123cfd6eeeb58dfa7947066` passes the prospectively selected
native conformance checks on Apple M1 Pro. Astra independently approves the
saved capture and complete reader. This closes B4's selected Mac engine-adapter
scope: 64 trajectory positions with separately admitted capacity 512. The
128,000-position model declaration, ordinary app integration, original-model
quality and iPhone qualification remain separate requirements.

The 16-layer model composes six full-attention layers and ten causal
short-convolution layers using shared packed projection, normalization,
attention/cache and convolution operations. Original weights remain Q4,
group 64 with BF16 metadata; direct gains/taps are widened for their concrete
resident operations. Immutable GPU weights survive session recreation; original
CPU payload/preparation storage is released before session use.

All 6,440 actual-input groups / 11,971,552 values and 303 strict coupled groups /
1,965,024 values pass with zero numerical or semantic failures. The former
checks operators against their observed native inputs; the latter checks
attention scores/context, final and teacher outputs against the frozen source.
This split was approved before collecting native outcomes. All 6,743 original
trajectory groups / 13,936,576 values per profile remain reported: three MAIN
and eight WIDE coordinates, each in a separate group, fail their diagnostic
gates. Those failures remain failures. A separate stdlib-only audit independently
replays all strict source comparisons and reproduces both diagnostic totals.

The reader reports 40,112 projection stripes, 5,134,331 equal-center
certificates and five exact-bracket checks. These are aggregate counters;
independent retained certificates for each fallback are absent. The qualified
binary64 affine-backend premise and conservative per-value softmax enclosure
remain explicit. Source-state diagnostics cover all 168 identities and 4,212
groups / 21,657,600 values per profile, with zero failures.

The real GPU capture completes 104 attempts / 367 rows, including alternate
chunk schedules, teacher prefixes, two raw greedy repeats, three late faults,
reset/replay and ordinary last-row operation. Its 14,672 arrays contain
154,629,840 bytes. All 335 model allocation hashes/sizes match before and after;
complete model buffer copies and per-boundary model hashing are absent. The
reader authenticates 299 row hashes, 126 guard records / 719,376 bytes,
18 history stacks, 77 matching native prefixes, 72 physical prefixes and two
ordinary calls with zero Rust allocations. Ordinary checks compare last-row
heads and published state; ordinary per-layer producers were not captured.

Omitted history sets status 1 without poisoning; a nonfinite head sets status 4
without poisoning. Partial copy changes exactly 128 head-0 K words, leaves
status 0 and poisons the session until reset/replay. Published-prefix, bank-role,
untouched-owner, dirty-tail and recovery checks pass. Exact and one-byte-under
budgets pass for ordinary last-row, all-logits and diagnostic modes.

On MacBookPro18,3 / Apple M1 Pro / 16 GiB / macOS 26.6.2, 431 sparse native
observations record maximum resident size 533,544,960 bytes and maximum
physical/lifetime footprint 562,005,048 bytes. The diagnostic ledger admits
973,679,662 bytes including weights, physical KV, history, scratch, diagnostics,
readback/logits, metadata, framework and external allowances. Ordinary last-row
and all-logits ledgers are 814,794,458 and 831,047,386 bytes. Allowances are not
measured allocations or whole-app peaks.

The separate complete reader exits zero with clean endpoints and a closed
process group. Its 4,394 direct samples and OS maximum RSS record 472,580,096
bytes maximum resident size. Elapsed wall time was not retained; the owner
enforces a 900-second / 6-GiB limit. Neither the capture nor reader establishes
generation throughput or latency adoption.

All ten codec fixtures and both raw repeats pass with tokenizers 0.22.2.
Both repeats sample `[561, 30219, 1127, 9319]`, decode to
`Titre et description`, and leave 9319 pending after the final sample.
Neither repeat reaches stop 7; BOS 1, stop-before-feed semantics, non-stop ID 2
and unmapped head slots 64400–65535 have separate fixture/control evidence.
Raw traces are not production Chat emission or natural-language quality tests.

The initial readers stopped on source KV axes, flat attention layout and a
Boolean interpretation of the fault status. The separate malformed 0/0 control
also remains failed. Narrow verifier corrections changed no engine arithmetic,
source values, gates or native data. The corrected reader passes 33 controls,
formatting, strict default/Metal Clippy, default 457 / Metal 564 tests
(20 / 183 ignored; suites overlap) and the release test build. A diagnostic state
run explicitly stubbed arithmetic and retained false qualification fields;
only the final complete, unstubbed run plus Astra's outer audit closes this task.

[Exact result and retained evidence](../benchmarks/results/2026-10-30-lfm2-real-native-v1.json)
pin the model, executable, all 15 reader inputs, complete outputs, original
three consumer replacements and final audit. Original-unquantized adoption
quality, app discovery/load/Chat, configurable context/KV settings, iPhone 14 Pro,
full declared context and performance adoption remain in B5/M8/M9. This does not
qualify arbitrary LFM2 exports or arbitrary Hugging Face models.
