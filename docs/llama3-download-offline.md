# Fresh Llama download and offline Mac Chat

The pinned Llama3.2 1B MLX8 profile now passes the production Hub journey in an
isolated empty cache on Apple M1 Pro: remote search returns the exact repository,
the Hub exposes its supported row, device admission passes, and Download and Use
retrieves and verifies all six files. The total is1,322,325,118 bytes. Its
1,313,157,436-byte weight exceeds the1GiB transfer threshold, so production uses
the ranged assembler; there is no whole-file fallback. This proves successful
assembly and final SHA verification, without claiming a retry count.

The verified model opens at512 positions and replies “Hello.” A separate process
reopens the same cache with all networking denied by a process-only sandbox.
Local writing succeeds; independent numeric loopback and pinned metadata probes
fail with EPERM. Offline cache verification and Chat succeed. Both phases record
the same snapshot SHA-256:
`4bb21b9e6b94ff2911fa9a4ced41ef3c50ee617eabdf04c6aed04484c809553e`.

A private cache copy then loses a required file and receives a same-size weight
mutation. Both are rejected before the open callback; the accepted model owner,
session, saved token prefix and conversation remain intact. The original cache
re-verifies and another Chat request succeeds. Both workers exit successfully.
No user cache or conversation is modified.

The existing Smol journey helper now accepts an explicit Llama test selector.
Review corrected two assertions before transfer: saved-profile inventory does
not itself perform device admission, and search results can contain local seeded
rows. The helper now refreshes admission and independently checks remote rows.
Strict compilation and existing Hub controls pass; the initial nested-sandbox
macro compiler failure is retained. No numerical implementation changes.

This completes fresh ranged-transfer/controller/cache evidence for the pinned
Mac profile. Ordinary catalog promotion is a separate change. iPhone14Pro,
rendered app UI and performance qualification remain separate.
[Evidence](../benchmarks/results/2026-11-03-llama3-download-offline-v1.json).
