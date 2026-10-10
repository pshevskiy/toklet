# Opt-in CPU projection selection in the CLI, 10 October 2026

The [CLI](../../src/bin/cli/mod.rs) accepts `--cpu-projection scalar|tiled`.
Default and explicit scalar execution use the existing reference path; tiled
opts into the numerically qualified ordered FP64 projection candidate.
Selection consumes the loaded model before session creation, changing only
its cold enum. Model weights and session buffers are not copied, and numerical
code, layouts, admission, accumulation and session commit behavior are unchanged.
The model's actual selected kernel supplies `cpu_projection` in JSON readiness
and every request's statistics, including one-shot generation. Selection stays
fixed for the process, so no new choice occurs within a numerical loop.

Unknown, noncanonical, missing and duplicate projection values fail during
argument parsing before model/config/tokenizer reads. The backend remains `cpu`.
The [CLI session check](../../tests/cli.rs) compares default scalar, explicit
scalar and tiled tokens, indices, stop/history/KV state, recoverable rejection,
reset and identical admitted/logical memory. Timing fields are excluded from
semantic comparison. Existing frozen token/EOS and numerical/session tests
remain independent references.

The [functional record](../../benchmarks/results/2026-10-10-cpu-tiled-cli-qualification-v1.json)
retains four separate real Qwen processes: frozen scalar CLI, new default,
explicit scalar and tiled. Each generates one token and then continues its
pending token for each of the three frozen offset-zero cases, resetting between
cases. All 24 generation requests agree on token/index/text events and complete non-timing
generation state. First IDs match the independent Qwen references; continuation
IDs match the frozen CLI. The three case sequences are `[1410, 1492]`,
`[4460, 311]` and `[271, 40]`. Every limited request ends with one pending token;
continuation evaluates that token and reuses the exact preceding prefix.
The selected mode is acknowledged at readiness and every completed request.

All four logical ownership accounts and admissions are identical. Maximum
own-child physical peak is `2,125,400,200` bytes and admission is `2,353,012,560`
bytes, both below 6 GiB. Tokenizer and application allowances are included.
Full role JSONL and stderr are retained alongside the result and hash-bound
before compaction. The [executed producer](../../benchmarks/results/2026-10-10-cpu-tiled-cli-qualification-v1.py.txt)
binds compiled source inventory, checks, model/tokenizer identity, fixture,
existing numerical qualification, helpers and all binary bytes. It refuses
overwrite and saves command output and partial child evidence on failure.
Its build-check watchdog kills and reaps only its own process group.

Formatting, Clippy for all targets with warnings denied, all 76 ordinary Rust
tests and an isolated locked offline release CLI build pass. Existing qualified
numerical sources are unchanged. Checks ran from 04:38:51 to 04:39:43 UTC after
the scalar boundary worker finished normally and committed all 2,048 positions.
The [parent hold](../../benchmarks/results/2026-10-10-cpu-cli-check-window-hold.json)
and [idle observation](../../benchmarks/results/2026-10-10-cpu-cli-check-window-idle.json)
show that only the matrix orchestrator was stopped. Its transport and numerical
worker completed the remaining measured requests and final capacity check.
The parent remains held for the separately reviewed application/Metal checks;
no timed numerical worker is stopped. Frozen matrix/application binaries remain
unchanged before and after qualification.

| Artifact | SHA-256 |
|---|---|
| New isolated CLI | `8e1be695229fb4a46e49725f06050b3296f6fb398e718fca8567bf0c6b400b43` |
| Qualification record | `8d2a633ed95c488f3bbba5de186aa3e3418fa75df51a170af383e31d98ae2cc8` |
| Executed producer | `db50c557c776b7ea397d86bbb5819a570a44ac5d0d502e2510521bdbff1b12a5` |
| Existing numerical receipt | `2ecfb27ef923973d327fa1be3c328d6acc85e10ab33ae55d6ca05a4c4d4bfbdc` |

Astra accepted the completed task without findings after independently
reconstructing all raw role events, source/fixture/receipt/binary identities,
ownership sums, check outcomes and window evidence. Application wiring and
Metal primitives remain separate tasks.

This task prepares application comparison; it adds no default adoption,
performance acceptance or M2.5/M3 completion claim. Controlled application
measurement requires a separately checked adapter and complete repetitions.
