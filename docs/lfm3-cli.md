# LFM3 BF16 CPU console route

Select the authenticated checkpoint explicitly with `--model-profile lfm3-bf16`.

```sh
toklet --model-profile lfm3-bf16 --model /path/to/lfm3 --prompt "Hi" --max-new-tokens 8 --json
toklet --model-profile lfm3-bf16 --model /path/to/lfm3 --interactive
```

The profile uses the pinned BF16 CPU graph and a default 16-token prefill
chunk. The existing raw ID, plain-text, JSON and interactive request formats
apply. Plain chat renders system, user and assistant string messages using the
source defaults for thinking. Structured content and tool messages are outside
this route. Greedy sampling is the shared console policy. The pinned EOS and
caller stop IDs are hidden when sampled.

The text route verifies the exact config, generation config, tokenizer config,
template and tokenizer before parsing. The source tokenizer's batch padding is
disabled for single-sequence execution. Codec and application admission reserve
512 MiB and 32 MiB respectively when both are in use. The generation adapter
reserves pending history and one FP32 distribution before checkpoint loading.
It widens the published BF16 distribution after a complete numerical step.

The controller retains an exact committed prefix. A changed or shortened
prompt resets and replays, and a fully cached prompt reuses its distribution.
Sampled IDs remain pending until a subsequent complete evaluation. The CLI
reports load phase timings, BF16 execution and memory, phase counts and times,
decode evaluations, token-event latency, and committed and pending counts.
Interactive cancellation is cooperative at chunk and token boundaries.

Five plain chats and three raw strings compare rendered bytes, token IDs and
streaming decode with the pinned Transformers and Tokenizers reference.
The reference verifies all 28,982 runtime files before and after and remains
under 512 MiB. The standalone codec test uses the same pinned assets.
[Fixture](../tests/fixtures/lfm3-cli-text-v1/fixture.json),
[reference guard](../artifacts/lfm3-cli-2026-11-11/attested-text-checks.json).

This scope uses a short CPU console workload. Deeper numerical parity,
the original full quality campaign, sustained performance and managed
adoption remain open.

## Console checks and timing

The standalone codec fixture passed at 297,861,120 bytes peak RSS. The real
interactive run passed 14 state/output cases and four request rejections.
It checked exact text/ID prefix reuse, changed and shortened replay, append,
continuation, reset, two-token decode, and SIGINT cancellation and resume.
The separate raw-ID run matched the reference's greedy ID 207 and retained it
as a hidden pending stop. Both runs and the timing process stayed below 6 GiB.
The timing process peaked at 3,603,202,048 bytes combined RSS upper bounds.
[Interactive checks](../artifacts/lfm3-cli-2026-11-11/verify-checks.json),
[stop check](../artifacts/lfm3-cli-2026-11-11/stop-checks.json).

Two reverse-order sweeps on M1 Pro measured four input tokens with a zero
output budget. Each cohort retains its first pass, three warmups and ten
samples. Median pipe round trip was 3.026–3.045 seconds after reset and
34.9–35.5 microseconds with the same fully cached prompt. The difference
between wall time and generation time includes host work and transport.
The samples leave telemetry-only overhead and application TTFT unqualified.
[Release timings](../benchmarks/results/2026-11-11-lfm3-cli-v1.json).

Fmt, both strict Clippy configurations, 733 CPU, 879 Metal and nine targeted
example tests pass. The explicit codec check and three native arm64 build/link
slices with 46 exports each pass, as do Mac CPU and Metal C smokes. An initial
padding enum comparison compile error was corrected. The first standalone
RSS command lacked access to macOS resource counters and the repeat passed.
[Checks](../artifacts/lfm3-cli-2026-11-11/checks.json).

Astra approved the console route after verifying source, protocol, timing and
native evidence.
[Astra review](../artifacts/lfm3-cli-2026-11-11/astra-review.json),
[Commit](../artifacts/lfm3-cli-2026-11-11/publication-checks.json).
