# Full tiled application trace, 12 October 2026

The frozen three-turn coding/tool-result transcript runs three warmups and ten
measured scenarios through the qualified isolated tiled CLI. Median scenario
latency is 129.141379 s, range 128.901821–129.971217 s. All thirty
measured turns match the accepted scalar token IDs, visible text, stop
reasons and untimed generation state. This completes the candidate application
measurement; M3.1 remains open for an explicit retention decision.

| Authored turn | Prompt tokens | Emitted tokens | Stop | Median first token / visible text (s) | Median finalized response (s) | Committed / pending |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| 1 | 36 | 128 | Output limit | 2.694224 / 2.694224 | 55.218936 | 163 / 1 |
| 2 | 76 | 18 | EOS 151645 | 5.366895 / 5.366895 | 12.820273 | 94 / 1 |
| 3 | 117 | 128 | Output limit | 8.225117 / 8.225117 | 61.036084 | 244 / 1 |

The hidden EOS is sampled and remains pending, so turn 2 has nineteen sampled
and eighteen emitted tokens. Each authored turn starts with an empty cache and
ends with reset; weights and session buffers remain resident. Scripted assistant
messages replace generated replies in later prompts. This is latency evidence,
not task-quality scoring, live tool execution or an external speed ranking.
The frozen loss/task comparisons are separately accepted without gate changes.

The [result](../../benchmarks/results/2026-10-12-application-tiled-v3.json),
[raw checkpoints](../../benchmarks/results/2026-10-12-application-tiled-v3.json.samples.jsonl),
[analysis](../../benchmarks/results/2026-10-12-application-tiled-v3-analysis.json)
and [exact analyzer](../../benchmarks/results/2026-10-12-application-tiled-v3-analysis.py.txt)
retain ten exact pairs, all thirty responses, ordered receipt times, text
hashes, independent summaries, memory accounting and provenance. The analyzer
launches no model and reads no model/binary. The
[scalar application](2026-10-11-application-scalar.md) median is 263.230241 s;
the descriptive scenario reduction is 50.9398%. These are separate
historical runs, not an alternating application experiment. The
[paired primary comparison](2026-10-10-cpu-primary-comparison.md) supplies the
confirming primary-metric acceptance evidence.

Driver timestamps cover submission through complete JSON-line receipt,
including transport/dispatch. First visible text is the first nonempty payload.
Finalization includes reset and any trailing Unicode text. Native numerical,
sampling, formatting/tokenization, detokenization and publication components
overlap native totals; they must not be added to them. Summed turn medians are
not the scenario median. Detailed warmup states and complete per-token/text
protocol lines are not retained by this driver; measured records retain token
receipt times, final text, native statistics and first-visible-text times.

Submission-to-ready is 1.967360 s. Native loading is 1.838237 s,
including nested conversion 1.174314 s; tokenizer loading is
0.118316 s. Source hashing warms files; startup is not cold-storage
qualification. Warm timers exclude setup. The ten measured scenarios total
1292.114342 s; recorded power and endpoint thermal observations apply only
to this uninterrupted sequence, not continuous sensor or energy qualification.

Maximum own-child physical charge is 2,123,745,392 bytes; conservative admission
is 2,353,012,560 bytes, below 6 GiB. Logical accounting retains
1,976,131,072-byte weights, 50,331,648-byte KV and 10,436,608-byte scratch,
plus history/logits, metadata and tokenizer/application allowances. The complete
inventory sums to admission and matches scalar. Complete system pressure and
energy are unavailable; physical scope is the CLI child.

Hardware is Apple M1 Pro / MacBookPro18,3 / 16 GiB / macOS 26.6.2 (25G83),
Rust 1.90.0 / Python 3.12.8. Dispatch revision is
`226e71db247522f6a1b34810b1529a19abbc3f2a`, with a clean tree. The
[CLI qualification](../../benchmarks/results/2026-10-10-cpu-tiled-cli-qualification-v1.json)
binds numerical sources/builds and isolated CLI SHA-256
`8e1be695229fb4a46e49725f06050b3296f6fb398e718fca8567bf0c6b400b43`.
The [functional adapter qualification](../../benchmarks/results/2026-10-10-cpu-tiled-application-qualification-v1.json)
is separately admitted. The [exact driver](../../benchmarks/results/2026-10-12-application-tiled-v3-driver.py.txt)
remains `9bcee63ef402a416d5a8cf743a81abfd7061add578544a383b633762418b57da`.
Dispatch identifies measurement, not a newly compiled executable. Exact model,
tokenizer, transcript, compiler/build and dependency bindings are retained.
Startup observations show AC attached, battery 80%, not charging, and AC low-power mode zero. Start/end thermal queries recorded no thermal or performance warning. End power source, continuous temperature/clocks, complete system pressure and energy are unavailable. Endpoint warnings do not establish controlled thermal conditions.

The [launch](../../benchmarks/results/2026-10-12-application-tiled-v3-launch.json),
[owned observation](../../benchmarks/results/2026-10-12-application-tiled-v3-owned.json),
[actual tool completion](../../benchmarks/results/2026-10-12-application-tiled-v3-session-completion.json)
and [owned idle proof](../../benchmarks/results/2026-10-12-application-tiled-v3-idle.json)
bind launcher/driver/CLI PIDs 51172/51212/51226 and PG 51172. The
[accepted changed-prefix predecessor](2026-10-12-cpu-tiled-changed.md) reaches
verified idle before dispatch. The
[paused cached attempt](2026-10-12-cpu-tiled-cached-excluded.md) remains excluded,
with unknown exit status/time/cause. Its committed recovery proves current
original-orchestration absence; no obsolete held-parent helper is used.
Pinned .venv dependencies and an exclusive fresh claim precede heavy hashes.
No other owned model/VM/test/build or large-hash work overlaps requests;
bounded observations and inactive source preparation/review continue.
Actual successful tool completion was observed at 2026-10-12T09:39:49.747035+00:00; owned idle was verified at 2026-10-12T09:39:52.344860+00:00. These are observation times, not a retained exact exit timestamp. All three owned PIDs/group are absent; Podman remained stopped, and root binaries, all eight assets, isolated CLI and compiled sources retain their admitted hashes.

Saved-data, content/link and whitespace checks apply; no kernel, numerical
helper, qualified artifact or quality gate changes and no new test suite are
needed for this evidence task. Astra accepted the complete evidence task without findings after byte-identical independent replay, all thirty scalar-identical responses/states, receipt timing, summaries, memory accounting, source/recovery bindings, four raw copies and all fifteen report links.
