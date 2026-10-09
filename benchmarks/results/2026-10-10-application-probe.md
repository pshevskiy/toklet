# CLI application timing functional probe, 10 October 2026

The [record](2026-10-10-application-probe.json) checks one three-turn scenario
using the frozen authored coding/tool-result transcript and local pinned Qwen
checkpoint on Apple M1 Pro, macOS 26.6.2 (25G83). Each turn has a two-token output
limit and uses an empty cache with the same resident model. This is a functional
check with no warmups, rather than the required application benchmark.

All three turns emit two ordered greedy token events and nonempty visible text:
`To validate`, `I will`, and `The operation`. Chat inputs contain 36, 76 and 117
tokens. Explicit resets finalize output before later authored turns; all native
records leave one sampled token pending before reset. Generated replies do not
replace the frozen assistant messages in the later scripted context.

New-process submission-to-ready is 2.309204 s. Native model load is 1.445585 s,
including 1.128968 s of payload conversion; tokenizer load is 0.123247 s. Those
timers overlap startup and are not added to it. Source hashing warms storage.
Native per-turn records separate chat formatting, tokenization, prefill, decode,
sampling, detokenization and publication. Driver timestamps separately cover
submission through complete JSON-line receipt, first visible text and explicit
Unicode finalization. Component scopes and overlap are explained in the record
and [benchmark instructions](../README.md).

Logical allocations, including tokenizer/application allowances, sum exactly to
2,353,012,560 admitted bytes. Physical child-process peak is 2,126,055,536 bytes,
below 6 GiB. Pressure and energy are unavailable. The driver, helper, release
binary, transcript, model and tokenizer identities are recorded. The base engine
revision is `b23795b` plus this CLI timing change. Numerical kernels are unchanged.

```sh
cargo build --locked --offline --release
.venv/bin/python tools/bench/application.py --probe --output benchmarks/raw/application-probe-repeat.json
```

Seventy ordinary Rust tests, formatting, Clippy with warnings denied, sixteen
offline Python tests, release build and whitespace checks pass. The full
application run still requires three warmups and ten EOS-or-128 scenarios;
the full scalar workload matrix is also pending. M2.5 remains open.
