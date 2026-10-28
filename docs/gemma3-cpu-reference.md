# Independent Gemma3 CPU fixtures

B3b1 supplies frozen expected values for the concrete CPU composition task.
This is source preparation, not a passing Rust graph or a real-model/device
qualification. The [selected real graph](gemma3-local-global.md) remains separate.

The tiny fixture has hidden64, MLP128, vocabulary64, four query heads, one KV
head and head width32, so concatenated query width128 differs from residual64.
Two layers are local/window4 and global, with ordinary theta10000/1000000.
Admitted fixture capacity is12. One original packed4/group64/BF16 table serves
embedding and output; there are15 matrices and13 raw BF16 gamma vectors.
The synthetic safetensors payload begins at unaligned byte5806.

The independent stdlib graph widens raw BF16 gamma and forms FP32 `1+gamma`
coefficients once. It uses FP64 mathematical reductions with FP32 operator
stores, scaled embedding, Q/K norm, distinct rotary branches, local/global GQA,
four hidden norms, tanh-GeGLU and postnorm residuals. It does not call a target or
old Gemma4 forward. This profile is not canonical BF16/MLX or future Half parity.

Four cases cover one token, a four-token prompt, twelve tokens crossing window4
and a changed final token. Each has48 named stage arrays. All192 arrays,
108,816 finite FP32 values and435,264 bytes have authenticated shapes, contiguous
extents and hashes. Six complete export/replay files are byte-identical; five
fixture files are promoted, leaving progress logs in the private evidence.
The CPU gate is frozen at `1e-5 + 1e-4*abs(reference)` before target outcomes.

Component witnesses distinguish unit-offset from direct norms and check BF16
encoding and the H640 embedding multiplier25.25. They do not by themselves
qualify an alternative BF16/Half coefficient-addition path. Actual head256,
window512 visibility at query512 excludes key0 locally while retaining key1 and
current key512; global attention retains key0. This is an analytical operator
witness, not a complete513-token model run or a resident GPU ring-cache check.

Source attention diagnostics are `[head,query,all-input-keys]`. A chunk/ring
consumer must map its actual absolute-key intervals into those reference
columns. Evicted or unavailable future score cells have no target capture.
Causal exactness concerns valid activations/logits and masked probabilities;
raw scores for changed future keys need not match.

Six portable packing/parser/operator controls pass. Guarded export/replay use
isolated Python3.12, fresh bytecode prefixes, authenticated source/runtime
identities and closed owned process groups. Export binds both executable
launchers in addition to the framework/stdlib; the earlier controls receipt is
preserved with its original pin coverage. No runtime or mathematical source
was changed after controls.

Astra reviews the source before export, then independently verifies all saved
arrays, source causal prefixes, original packing and analytical controls. It
reconstructs the original tied table and exactly rescores all1,856 source head
logits. This approves fixture retention; Rust numerical/state/allocation checks
are the next code task. No real model, GPU, performance, app, phone or adoption
quality claim follows from these fixtures.

See [compact evidence](../benchmarks/results/2026-10-28-gemma3-cpu-reference-v1.json),
[TODO](../TODO.md) and [improvement record](improvements.md).
