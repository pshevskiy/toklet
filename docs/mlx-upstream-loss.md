# Diagnostic upstream loss for the selected public MLX checkpoint

The [explicit reference runner](../tools/reference/mlx_qwen_upstream_loss.py)
loads the same immutable public Qwen snapshot with MLX 0.32.3 and MLX-LM 0.32.0.
It uses the installed static Qwen2 classes, strict local weight loading and
`trust_remote_code=False`. It does not tokenize, download files, import model
repository Python, or add an MLX dependency to Toklet's native product backend.

The workload is the same frozen 10,000 targets and 1,024-target block policy used
by the [native Mac and phone loss experiment](mlx-full-loss.md). Each reference
step evaluates its input token with a fresh per-block prompt cache, completes the
GPU work, widens the returned logits to FP32, and scores the next exact corpus ID
using independent stable FP64 log-sum-exp. The runner records actual output dtype.
This upstream graph can use different kernels, activation/output rounding and
reductions from Toklet. Per-target differences are descriptive, with no newly
invented native acceptance tolerance or performance equivalence claim.

The reference freezes all non-cache MLX package files, including its separate
Metal library and headers, plus selected MLX-LM/NumPy Python sources/native binaries,
package versions, producer helpers, all model inputs, the native raw target stream,
the baseline and the unchanged 0.05 loss contract before inference. That inventory
is a selected producer closure, not a complete inventory of Python or the OS.
Block endpoints guard the same bytes, capture native process charge and enforce
the 6 GiB Mac trial limit. Exclusive outputs retain failed attempts and bind the
completed or partial target journal through result publication. Seven new portable methods
cover stable loss and ties, invalid numerical/shape inputs, full block coverage
and early rejection of existing or dangling output paths, exact upstream shard
inventory, retained failure-journal bytes and separate Metal-library provenance;
seven existing real oracle methods also pass.

Astra's initial source review found that upstream globs every matching safetensors
file and that failed journals needed an independent retained-byte binding. Exact
inventory guards and synthetic zero-load/one-row-failure tests fix those before
the first trial. During that trial, the agent found that the producer inventory
omitted MLX's separate Metal library. It verified the owned driver/worker tree,
sent SIGTERM to the driver and confirmed both processes absent. The interrupted
7,449-target journal, original interruption, clean cleanup/endpoints and exact
historical sources remain retained without interpreting its incomplete results.
A fresh reviewed source/check/run includes the library and all other non-cache
MLX files. Numerical policies and native product sources remain unchanged.

The corrected run completes all 10,000 targets on the verified M1 Pro. Its mean
loss is **3.083056082158314**, an increase of **0.19933093222882992** over the
frozen FP32 baseline: it also **fails** the unchanged **0.05** limit. Toklet's
native mean is **3.0829338398049995**, lower by **0.0001222423533144834**.
The large application-family loss rejection is therefore reproduced by the
upstream runtime, rather than being unique to Toklet's observed execution.
This supports rejecting the selected artifact under the policy; it does not
establish its exact prequantization source or uniquely attribute the loss.

The runtimes do not return identical distributions: 44 of 10,000 greedy IDs
differ, maximum target-logit difference is 0.14873313903808594 and maximum
per-target loss difference is 0.09343106160399373. Upstream returns Half logits;
Toklet deliberately computes FP32 tied-output logits. Other kernel/rounding
differences also remain. These are descriptive observations, not a newly adopted
parity tolerance. The separately frozen native oracle and failed loss gates
remain unchanged.

Upstream lifetime peak physical footprint is **620,070,376 bytes**, below its
6 GiB reference-process trial cap. This Python/MLX diagnostic has different
application/runtime state from the native app and supplies no speed or memory
ranking. The final 10,000-row journal is complete and byte-bound. All 14 portable
methods pass in two commands; the final owned hardware query and reference
process also pass with clean closure. The stopped attempt is retained separately.

Receipts: [diagnostic qualification](../benchmarks/results/2026-10-22-native-mlx-upstream-loss-v1.json)
and [raw inventory](../benchmarks/results/2026-10-22-native-mlx-upstream-loss-raw-v1.json).
Retain the diagnostic and the checkpoint rejection. No runtime default,
numerical threshold, native Rust/shader or Swift app code changes in this task.

Astra accepted the completed diagnostic after independently checking 1,145
current pins, 105 raw files, 57 source copies, every target/comparison metric,
14 portable methods and the retained interruption/cleanup evidence. Privacy,
scope, all 222 local links and whitespace checks pass. The checkpoint remains
rejected under the original loss policy.

Successful collection is separate from the application loss-gate decision.
The original-family FP32 comparison still lacks byte proof of the public
checkpoint's precise prequantization source revision. Diagnostic agreement
cannot waive the native loss rejection, establish checkpoint adoption or close
the remaining M8 quality, performance, architecture or app discovery requirements.
