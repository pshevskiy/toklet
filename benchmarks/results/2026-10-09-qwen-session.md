# M2.3 CPU KV sessions, 9 October 2026

Cached execution exactly matches full-prefix FP32 logits on all five tiny cases
and all four pinned Qwen numerical cases, using one-shot, chunked, and single-token
appends. The Qwen suite makes 47 complete-distribution comparisons, including
absolute position offset five, pending sampled tokens, fully cached prompts, and
reset/replay. Greedy IDs agree. The existing M2.2 full-array Transformers suite
still passes and reproduces its committed numerical report exactly; no numerical
gate or arithmetic precision changed.

[Session ownership and API](../../src/session.rs) borrow the immutable model for
the session lifetime. Each layer owns separate FP32 K/V
`[kv_heads, capacity, head_dim]`, strides `[capacity * head_dim, head_dim, 1]`.
Only new K/V is copied from token-major projection scratch; no old cache prefix
is moved and no KV heads are expanded. Attention reads adjacent head features,
preserves causal key boundaries and GQA mapping, and reuses one `[capacity]`
score/probability row. Activation/MLP scratch is bounded by the admitted chunk
size rather than total context. Dots/context remain FP32, softmax and projection
accumulation retain the qualified FP64 behavior.

History includes unevaluated input or sampled tokens; `evaluated_len` alone marks
valid KV. Each chunk writes beyond that length and stages its final distribution.
The length and logits advance together only after all layers and the output
projection succeed. A failed chunk leaves the earlier prefix valid, with its
pending history available for retry or discard. Tests fail a later layer and the
tied output projection after earlier KV writes, then verify recovery. Reset and
discard need no cache clearing because stale tails are outside the valid prefix.

`append` stages validated IDs; `evaluate` completes pending chunks;
`evaluate_next_chunk` supplies a synchronous cancellation boundary between
chunks. It does not interrupt an in-flight chunk. `logits` is available only for
a nonempty fully evaluated history. `set_tokens` reuses an exact extension of
the complete current history, including pending tokens. Editing or shortening
resets and replays; longest-prefix truncation remains M5.5. `discard_pending`
restores access to the committed prefix's distribution. Rejected requests leave
history, committed length and distribution unchanged.

Admission checks token/rotary limits, multiplication/addition overflow, and the
single-model/single-session budget before allocating. It counts numerical weight
capacities once, allocated KV, chunk scratch, history, two logits buffers,
allocated model/cache layer metadata tables, and a 64 MiB metadata allowance.
Actual allocated capacities are checked again after construction. Tokenizer,
application buffers and other live sessions need separate accounting. This is
an admission estimate, not an equality with physical process footprint.

[Detailed results](2026-10-09-qwen-session.json) retain checkpoint identity,
cases, comparison counts, numerical allocation accounting, and the own-process
physical peak. The default real-model session admits 2,048 KV tokens and a
128-token chunk, with 50,331,648 KV bytes (48 MiB). The full 2,048-token numerical
boundary is checked with a small two-layer CPU model against full-prefix
execution, plus rotary positions 2,046–2,047 and rejection of token 2,049. Real
Qwen numerical execution here covers short probes; its 1,920-input/128-output
workload, longer-context quality and performance remain for M2.5.

Ordinary tests require neither model files nor Python. They cover frozen GQA
values, ignored nonfinite cache tails, invalid valid-prefix data, shape/admission
errors, cached/changed/pending prompts, greedy trace, transactional failures,
capacity and zero allocator-counted allocations during admitted prefill, decode,
cached reuse, reset and replay. The test-only allocator delegates all memory
contracts to `System` and counts only the measured thread. This is allocation
evidence, not a throughput claim.

Reproduce on the explicitly prepared pinned model:

```sh
cargo fmt --check
cargo clippy --locked --offline --all-targets -- -D warnings
cargo test --locked --offline
cargo test --locked --offline --release --test qwen_forward pinned_qwen_fp32_matches_full_transformers_forward -- --ignored --nocapture
cargo test --locked --offline --release --test qwen_session pinned_qwen_cached_sessions_match_full_prefix -- --ignored --nocapture
```

The full forward suite also needs the M2.2 local Transformers arrays. Mac
physical footprint inspection uses `/usr/bin/footprint` on the test's own PID;
other platforms record no physical peak. Qualification ran on Apple M1 Pro,
MacBookPro18,3 with 16 GiB, macOS 26.6.2 (25G83), Rust/Cargo 1.90.0, pre-task
revision `66630c5` plus this task's changes. Source hashing warms the file cache.
The physical peak includes hash/test/diagnostic setup and the live default
session. No latency, held-out quality, Metal, or real-model long-context result
is claimed.
