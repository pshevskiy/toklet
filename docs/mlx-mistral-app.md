# Selected Mistral Models and Chat on Mac

Toklet Harness adds the immutable
`mlx-community/Mistral-7B-Instruct-v0.3-4bit` revision
`a4b8f870474b0eb527f466a03fbc187830d271f5` to the existing SmolLM2 and Qwen catalog.
Its six required files total **4,079,633,318 bytes** and retain exact size/SHA256
identities from the [verified checkpoint](mlx-mistral-checkpoint.md).
Search selects this pinned profile; other search results remain unsupported.
Saved receipt fields and both smaller profiles are unchanged.

Mistral reserves **5 GiB numerical**, **64 MiB text**, and checks **5.5 GiB of
current process allowance** before download/open. On Mac that allowance is
`min(RAM/2, 6 GiB)` minus current `TASK_VM_INFO` charge, not free system RAM.
Opening verifies again and the chat controller takes another current admission
reading before submitting the load.
The friendly header displays the loaded profile's name. Expert mode lists the
implemented full-attention Mistral family alongside Qwen2 and Llama-family models.

## Checks and measured scope

The new native package has three SDK slices and 21 successful build/link/symbol
commands. Receipt SHA256 is
`37f255ca04d4c839f51ece740bda1d8d2fa0bd026b505a1a551cf50975e3eb04`.
Final shared-app SDK builds, strict Swift integration/core/profile tests, old
snapshot compatibility and Qwen-to-Mistral-to-Smol selection checks pass.
No physical phone was contacted. SDK compilation grants no phone qualification.

The real Mac test seeds a fresh snapshot from the independently acquired and
verified local checkpoint through the actual ModelHub publication/hash path.
It then performs public search, verifies/reuses that snapshot, and executes
native multi-turn Chat, context rejection with prefix preservation, cancellation,
suspend/replay and offline friendly controllers. Its explicit local-seed marker
prevents treating this as a fresh network weight download.

Mistral's first eight IDs match the committed native C text test and decode to
**“Hello! How can I help you today”**. A subsequent turn and a 32-token controller
reply pass; Python tokenizers independently decodes every reply with its exact
source/native extension admitted before import. Qwen retains its prior token-0
reply, and Smol's old saved snapshot passes native and offline controller checks
with the new package. This tests functional text plumbing, not answer quality.

The completed Mistral test reports a lifetime charged process peak of
**4,226,780,712 bytes** and a post-close reading of **3,564,080,296 bytes**.
Closed native handles do not imply immediately reclaimed GPU/OS memory charge.
Charged counters can omit clean mapped pages; the
[decoder's logical allocation accounting](mlx-mistral-metal.md) remains separate.
One completed functional run is not a sustained memory or performance trial.

Two earlier local runs rejected a cached pre-verification admission result.
The second observed a later 50,758,328-byte charge and 6,391,692,616-byte allowance.
Those later readings do not prove exactly why the earlier observation differed.
Refresh now samples admission after awaited verification, and an observation
error clears an older allowed state. Tests cover both availability transitions
and observation failure; budgets were not lowered. Both failed trials and exact
historical sources remain in the [app receipt](../benchmarks/results/2026-10-23-mistral-model-hub-app-v1.json).

## Remaining limits

The separate [fresh Mac transfer and reopening](mlx-mistral-hub-download.md)
now passes: all six files are downloaded into a new root, independently hashed,
and the actual downloaded copy passes native and offline-controller Chat.
The earlier full-body connection loss and first ranged timeout remain retained
failures. That first ranged diagnostic does not identify its failed file/range
or terminal byte count. The successful later retry does not prove their cause.

Physical iPhone work stays paused at the user's request. Qwen's pending phone
Chat and all Mistral phone checks remain open. Wider architectures/checkpoints,
sliding/Online Mistral attention, held-out quality, long-context numerical gates,
GUI taps, sustained resource limits and latency/throughput qualification are
outside this functional Mac task.
