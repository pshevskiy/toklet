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

The first actual fresh multi-GB range trial also failed, with timeout code
-1001 and an empty final model directory. Its saved diagnostic does not identify
the failed file/range or final byte count. The earlier full-body trial lost the
connection. Both safely discarded incomplete staging. The
[bounded range mechanism](model-hub.md#bounded-large-file-transport--23-october-2026)
has deterministic acceptance; a complete real transfer requires its own full
SHA and publication evidence. This app task makes no fresh Mistral network
weight-transfer claim.

Physical iPhone work stays paused at the user's request. Qwen's pending phone
Chat and all Mistral phone checks remain open. Wider architectures/checkpoints,
sliding/Online Mistral attention, held-out quality, long-context numerical gates,
GUI taps, sustained resource limits and latency/throughput qualification are
outside this functional Mac task.
