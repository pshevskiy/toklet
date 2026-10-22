# Real Llama-family app execution on Mac and iPhone

Toklet's native Rust/Metal engine executes the pinned
[SmolLM2-135M eight-bit LlamaForCausalLM checkpoint](mlx-smollm-checkpoint.md)
through the Swift app coordinator on Apple M1 Pro and the signed physical
**iPhone 14 Pro, iOS27.0.1**. This is SmolLM2's Llama-family graph; Meta-branded
Llama checkpoints and rotary variants are separate compatibility work.
The [native numerical qualification](mlx-smollm-metal.md) and its independently
frozen gates remain unchanged. This app task changes no Rust numerical source,
shader, C ABI selector, model bytes, tolerance or protected CPU binary.

## Implementation and checks

Explicit SmolLM recipes use its captured tokenizer IDs and EOS2, vocabulary49152,
Reference attention and the exact six numerical/tokenizer file pins. The original
Qwen recipes remain intact. Three scoring targets use `Hello, world!`'s four IDs;
two authored generation probes have caps4/2. These do not use Qwen's held-out
corpus or task baseline. Numerical/process trial budgets are512MiB/1GiB, separate
from measured process charge. Generation/tasks/state use capacity32/chunk3;
scoring uses capacity4/chunk1.

The physical transport accepts one complete explicit native-package/prepared-app
context, replays same-app storage admission, authenticates transfers and supports
a distinct `--state-component` route. Legacy `--state` still requires its original
prefill witness. UI labels now describe supported MLX-format Qwen2/Llama-family
artifacts. A separately reviewed preflight reset validates the freshly installed
development bundle, device, executable, PID and signal response immediately
before termination. It preserves original command/capture/cleanup errors and
never claims ownership of a process that the new trial did not launch.

Fresh Rust1.90 XCFramework slices build and link for Mac, device and simulator;
the Mac coordinator compiles with strict concurrency/warnings and its iOS
sources type-check. The actual iPhone Release app is signed, profile/signature/
seven-file inventory verified and installed. All232 portable methods, formatting
and strict all-target Metal Clippy pass. Earlier142 CPU/158 Metal-feature ordinary
passes and14 explicit GPU cases are inherited from the unchanged numerical task.
Simulator linking/type-checking is distinct from physical execution.

Both devices pass thirteen generation requests in each of two runs (three
warmups/ten measured), all returning `[339,3543,719,1891]` with history8,
committed7 and pending1. Three complete scalar score records match exactly across
devices and satisfy intervals derived from the frozen independent full-logit
arrays. Both authored task responses match, and all26 state/scalar rows agree
exactly: cached reuse, pending append, changed-prefix replay, full32 commitment,
overflow preservation, reset and release. This component state route makes no
asynchronous partial-prefill cancellation or OS interruption claim.

## Short workload observations

Four input IDs/four generated IDs, capacity32, chunk3, resident eight-bit affine
weights with Half activations/KV and Reference attention. Each primary and
confirmation has three warmups and ten measured requests. Values below are
primary/confirmation medians in milliseconds. Decode is the per-request average
of the three intervals after the first token. Timed native work completes before
its event; loading/session creation, transfer, artifact hashing, tokenization and
visible UI/text latency are excluded.

| Device | Native first token, ms | Native decode interval, ms | Largest observed process peak across component jobs, decimal MB |
| --- | ---: | ---: | ---: |
| Apple M1 Pro | 39.471 / 39.950 | 18.986 / 19.201 | 156.730 |
| iPhone 14 Pro | 63.240 / 63.359 | 27.102 / 27.615 | 169.855 |

These are short functional observations, with raw samples retained. They do not
establish representative long-generation throughput, a comparative optimization
speedup, sustained behavior or a universal mobile memory limit. The native engine
keeps142,955,990 allocated weight bytes; logical session admission and whole-
process lifetime peak remain different quantities.

## Evidence and remaining scope

The [publication receipt](../benchmarks/results/2026-10-23-smollm-app-device-v1.json)
binds4,790 files and298 current source copies, including source/package/signed/
fixture identities, raw logs, independent score replay and storage/launch/process
observations. Fourteen saved-data mutation methods also pass. Thirteen phone
stages complete130 owned commands, including117 device commands and13 host boot
queries, with26 natural absence observations and no trial cleanup signals.
Two explicit initial resets are recorded separately.

The initial generated-library-link refusal, a later existing-process refusal and
missing reuse-source snapshots remain retained failed attempts. Only the verified
generated link was changed; exact receipt-bound controller copies were added
before reuse. Failed attempts receive no inference credit.

Retain selected real-model app execution on both devices. Metal admission is
bounded to512 positions, with a separate actual512-capacity witness on Mac;
this phone collection exercises32 positions. Online Llama attention, independent
long-context numerical parity, full held-out/task quality, downloaded chat
formatting, actual interruption/jetsam, longer/sustained performance and broader
checkpoint/architecture coverage remain open. M8.3's other dense/MoE/recurrent/
multimodal families and M8.4 discovery/downloads are not closed by these probes.
