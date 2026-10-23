# Selected Mistral raw-ID observations on Mac

Two fresh optimized native processes execute the pinned
[Mistral 7B four-bit/group64 checkpoint](mlx-mistral-checkpoint.md) on **Apple M1 Pro**.
The native package receipt is
`37f255ca04d4c839f51ece740bda1d8d2fa0bd026b505a1a551cf50975e3eb04`.
The unchanged measurement coordinator and Mac validator use three warmups and at
least ten measured requests, stopping at the first eligible request after
600 measured native seconds. Both runs complete **82 measured requests**.

The fixed raw prompt is `[23325,29493,2294,29576]`, output limit 32, capacity 128
and prefill chunk 32. Every request resets KV while retaining weights/scratch.
The first four IDs match the independently frozen FP64 reference
`[781,781,29505,29510]`. Full 32-ID sequences are stable within and across both
processes. This is a numerical probe, not an instruction-following quality test.
No numerical, clock or memory gate changes are made. The numerical cap is
4 GiB minus one byte; the whole-process observation cap is 4 GiB.

| Observation | First process | Second process |
|---|---:|---:|
| Measured native duration | 602.477 s | 602.576 s |
| Native first-token median | 536.028 ms | 535.985 ms |
| Decode median | 4.551 IDs/s | 4.549 IDs/s |
| Completed caller median | 7.3472 s | 7.3503 s |
| Load and session creation | 3.6019 s | 3.1186 s |
| Charged lifetime process peak | 4,119,186,864 B | 4,119,039,408 B |

Native time measures completed work. Decode is the 31 intervals after the first
output divided into their elapsed native time. Caller duration also includes
queue/callback handling. Loading is separate, and artifact verification warms
storage. Tokenization, Chat UI and network transfer are outside these timings.
This shorter KV capacity and absent text codec differ from the
[512-capacity functional app run](mlx-mistral-app.md).

Every sampled thermal state is nominal (0) and low-power mode is false. One
**pre-run** query reports AC power; it does not prove unchanged power throughout
both runs. A fresh Hugging Face range download runs concurrently and can affect
storage/network/thermal contention. These are confounded selected-workload
observations, not a performance qualification, external speed comparison or
representative context-depth matrix. Charged physical counters can omit clean
mapped pages and do not replace logical memory accounting or mobile limits.

Both command processes close their owners, retain valid state/clock brackets,
recheck complete input hashes and pass the unchanged strict validator. The
[receipt](../benchmarks/results/2026-10-23-mistral-mac-observations-v1.json) binds
sources admitted before import, exact workload/package/model identities, raw
samples, diagnostics and protected historical binaries. Phone, held-out quality,
other shapes and text/UI latency remain outside this observation.
