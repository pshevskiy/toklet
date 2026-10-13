# Resident root Metal backend

The root engine now supports opt-in `--features metal` / `--backend metal` on
Apple Silicon macOS 15+. CPU remains the default. The backend uses the pinned
public objc2/Metal/MPS bindings, precise MSL 2.3 arithmetic, tracked shared
resources and checked completed commands. Qualified geometry is the 32-wide
independent tiny fixture and Qwen2.5-0.5B (896 hidden, 4864 intermediate, 24 layers,
14 query / 2 KV heads, dimension 64). Theta is 1e6 and RMS epsilon is 1e-6.

## Mechanism and ownership

The loader validates the existing Qwen artifact boundary and streams F32/F16/BF16
source values into resident Half buffers through the existing 64 KiB source
stripe. It retains no full FP32 model. Source nonfinite values and finite values
which overflow Half are rejected. Optional tied output weights must match the
original embedding values, before Half rounding; signed zero is canonicalized
for this numeric equality check. All load/session allocation arithmetic is
checked and admitted against the initial 6 GiB ceiling.

Weights use row-major `[out,in]` MPS layouts with queried row bytes. Activations
are contiguous `[chunk,hidden]` Half; K/V remain separate
`[kv_heads,capacity,dimension]` Half buffers per layer. Scores/probabilities use
`[query_heads,chunk_capacity,cache_capacity]` FP32 storage. The reusable projection
result is FP32 with a queried physical stride, large enough for vocabulary output.
MPS uses Half inputs/weights, FP32 results, right transpose and resident bias
initialization. Each stored intermediate is checked for finite Half range.

Sessions own fixed KV, scratch, history and two CPU logits buffers. Exact row
views and multiplication kernels for admitted chunk sizes are prepared before
execution. Generation uses a coarse statically dispatched session interface.
A chunk encodes embedding, all layers, final normalization and tied logits in
one command with one completion wait. CPU greedy sampling remains the initial
M4 path and reads the final logits row into preallocated storage.

Only successful completion publishes logits and advances committed positions.
A finite-overflow chunk leaves the previous valid KV prefix; its pending input
can be discarded. A command failure requires explicit reset/replay. Prefix reuse
requires extension of the entire history, including pending sampled tokens.
Cancellation is observed between chunks/steps; output failure and EOS preserve
pending sampled-token semantics.

Optional diagnostics pre-admit snapshot and CPU readback capacity and encode
ordered GPU blits, with four-byte aligned logical copy sizes, in the same command.
They add no intermediate waits. Public capture guards produce an actual
`.gputrace` with `MTL_CAPTURE_ENABLED=1`, reject an existing capture/destination,
and retain the captured queue/model through completion.

## Qualification

The final explicit release GPU suite checks five tiny cases at full/3/1 chunks,
all selected intermediate rows against the original frozen FP16-versus-source
FP32 gates, four frozen greedy steps, CPU-compatible pending prefix rules,
generation cancellation/continuation/EOS/output failure, exact budget admission,
and zero Rust allocations over 20 warmed decode steps. A zero-weight structural
oracle checks all 2048 positions and a nonzero absolute-offset boundary exactly.
It is a capacity/state test, not a real-model long-context quality claim.

Full Qwen checks cover four frozen prompts and full/3/1 chunks, with embedding,
16 stages in each of layers 0/1/23, final normalization and logits: 51 arrays per
run, 612 comparisons total. All original source gates pass; final greedy IDs
1410/537/4460/271 match source references. No tolerances were changed. The initial
debug harness run stopped because the full FP32 reference metadata has no dtype
key; correcting the bounded, hash-checked reader resolved it.

A separate GPU unit check demonstrates real finite-overflow rejection/recovery
and injects a post-completion status error to exercise reset/replay publication.
It does not deliberately induce a hardware fault. Ordinary CPU tests skip GPU
and installed-model suites. Streaming dtype/tied-output unit tests call no GPU.

The resident collector runs three warmups and ten repeated requests, resets
history between them, retains raw events and SHA-bound source/build/model/
tokenizer provenance, checks identities again after child cleanup, bounds both
pipes and has startup/request/lifetime limits. Seven tiny-process failure tests
cover EOF, oversized lines, timeout, unexpected events, stderr flood, evidence
preservation and descendants holding pipes after leader exit. Public libproc v4
charge counters are read only while the owned model child is idle. `/usr/bin/footprint`
hung even for an own Python process; only that owned diagnostic was terminated.

The [final resident report](../../benchmarks/results/2026-10-13-metal-engine-resident-v3.json)
retains all source/model/tokenizer/binary endpoint identities and raw events.
Native peak charge after 3/13 requests is 1,134,658,952 / 1,134,675,336 bytes
(a 16 KiB increase), with final chat peak 1,135,674,760 bytes. Numerical plus
application admission is 1,329,601,312 bytes. These are native charge counters,
not a claim of complete shared/clean physical-page accounting. The numerical
working set is admitted from physical buffer lengths, counting tied weights once.
The [qualification record](../../benchmarks/results/2026-10-13-metal-engine-qualification-v1.json)
binds final GPU binaries, logs, trace-bundle files and unchanged protected CPU
release binaries. Default tests: 76; Metal-feature ordinary tests: 78; Python
experiment tests: 125; strict Clippy and formatting pass in both configurations.

The complete Qwen chat reaches EOS 151645: 40 prompt tokens, 54 sampled tokens,
53 emitted, 93 committed and one pending EOS. This establishes execution and
stop semantics; it is not an accuracy benchmark of the response's prose.

## Scope and remaining work

This closes the root integration evidence for M4.1–M4.3. An actual tiny GPU trace
was captured (62 MiB; one captured frame, no unused pipelines/resources in its
metadata). Whole-command GPU start/end, host encoding, wait and readback are
recorded. `xctrace` is unavailable in the installed Command Line Tools. Kernel/
idle timing attribution remains a separate M4.4 task; the capture alone does not
close that requirement. Astra accepted the code and evidence after independently checking 10 collector/native-ABI tests, all resident raw responses, source/log/bundle hashes, admission and links. No speed-parity, FP16 held-out quality or M5 claim is made. The plan’s 10,000-target NLL and 100-task lossy-quality gates remain outstanding before quality/performance replacement acceptance.
