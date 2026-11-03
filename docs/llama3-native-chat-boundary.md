# Llama3 codec and native session boundary

This explicit ignored Mac test composes the pinned plain Rust codec with the
existing generic native MLX loader and generation/session APIs. It does not
change native chat factories, ordinary app discovery or the selected Apple
framework. The six cached files are checked against the existing Llama3 pins
before loading and after releasing the session, cancellation flag and model.

The test finds a real rendered 511-token user prompt. It checks exact
prompt-plus-one-reply admission at 512, then rejects a second requested reply
token without emitting callbacks or changing sampled/committed state. A separate
short prompt cancels after the first visible token, retains that pending token,
and successfully resets and replays. Every request receives all three hidden
stop IDs. Observed stops are reported separately from fixture coverage; absent
stops are not claimed as model-generated cases.

The non-unwinding C callback records cancellation errors and lets the test
assert after returning. Successful requests check emitted counts and stop
sentinels/membership. Ordinary CPU tests remain independent of downloads and GPU.

This qualifies only the pinned codec/native-session composition on Mac at 512.
An audited assembled-owner receipt, complete native chat pair, managed app route,
original full-adoption quality, iPhone and performance remain separate gates.

All cold checks pass: fmt, strict Clippy in both configurations,544default and
652Metal tests. The explicit cached Mac check passes and reauthenticates all
six files after release. Recovery emits IDs8586/596. Neither successful request
selects a stop; all three remain independently checked by codec fixtures.
[Evidence](../benchmarks/results/2026-11-03-llama3-native-chat-v1.json).
