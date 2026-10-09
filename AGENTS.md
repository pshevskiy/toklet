# Project guidance

This guidance applies throughout the Toklet repository. Build a small, understandable Rust inference engine for Apple Silicon, with correctness, predictable memory use, and measured latency as primary goals.

For implementation work, follow the milestones and acceptance criteria in [PLAN.md](PLAN.md). Use [apple_llm_performance_plan.md](apple_llm_performance_plan.md) for design rationale. Inspect the current code before assuming a planned feature exists. Keep changes within the requested task; a roadmap is not a request to implement every milestone.

## Architecture and working method

- Start with one package and ordinary functions over slices. Add modules when their implementation exists. Introduce abstractions only when concrete use cases justify them.
- Preserve the readable scalar CPU implementation and independent numerical fixtures as correctness references while optimizing other paths.
- Separate immutable model weights from session-owned KV state and scratch. Keep tokenization, chat formatting, UI, downloads, and tool execution outside numerical kernels.
- Implement one mechanism at a time. Explain its shapes, layout, ownership, and numerical behavior, then validate it before composing larger changes.
- Optimize the measured bottleneck. Treat every performance claim as a hypothesis until a representative workload supports it.

## Static dispatch and Rust hot paths

- Avoid dynamic dispatch and indirect calls in numerical hot loops. Prefer concrete types, generics, and statically dispatched functions over `dyn Trait`, boxed trait objects, or function-pointer calls per element, head, or tile.
- Select the backend, dtype, quantization scheme, and kernel variant outside inner loops. A small enum match at a model, request, or operation boundary is usually sufficient; do not repeat the same selection per scalar.
- Dynamic dispatch is acceptable at coarse integration boundaries when it improves the design and its cost is negligible or measured. Do not replace simple high-level code with a large generic hierarchy solely to remove one infrequent call.
- Keep specialization bounded to useful shapes and device capabilities. Consider generated code size, instruction-cache pressure, and build time when adding generic variants or unrolling.
- Use borrowed slices and explicit ownership. Avoid unnecessary clones, reference-count updates, boxing, and intermediate collections in the inference path. Heap ownership for long-lived weights and buffers is expected.
- Use iterators or indexed loops according to clarity and measured generated code. Neither style is inherently the optimization. Prefer loop structures that make contiguous access and bounds evident to the compiler.
- Hoist invariant calculations and shape checks out of inner loops after validating operation inputs. Do not remove validation at public or artifact-loading boundaries.
- Do not apply `#[inline(always)]`, unchecked indexing, manual prefetching, or target-specific intrinsics without evidence. Inspect optimized code when it helps explain a bottleneck. Preserve a qualified fallback for unsupported CPU features.

## Mechanical sympathy

Design data movement and work scheduling around the hardware that executes them. For each important kernel, understand the working set, bytes read and written, arithmetic performed, access pattern, and synchronization cost.

- Prefer contiguous, predictable traversal. Arrange inner loops around the physically contiguous dimension; avoid pointer chasing and scattered reads in tensor storage.
- Use the documented layouts: row-major linear weights `[out_features, in_features]`, activations `[tokens, hidden]`, and separate per-layer K/V buffers `[kv_heads, capacity, head_dim]`. Document strides, alignment, padding, and any packed representation explicitly.
- Choose array-of-structs or struct-of-arrays from the fields consumed together. Keep frequently accessed numerical data separate from cold metadata. Resolve tensor names and configuration lookups during loading or planning, rather than through maps in every forward pass.
- Tile work to reuse weights and activations in caches or GPU on-chip storage. Measure tile sizes on real projection shapes. Account for register pressure and occupancy as well as reuse.
- Treat single-token decode and multi-token prefill separately. Decode often emphasizes weight bandwidth and dispatch cost; prefill can reuse weights across rows. Confirm the limiting resource for the actual model and context depth.
- Reduce total memory traffic before chasing instruction-level savings. Avoid redundant transposes, whole-tensor copies, expanded KV heads, and materialized intermediates that a qualified fused operation can eliminate.
- Preserve alignment guarantees when using vector loads or packed formats. Do not assume a `Vec<T>` provides alignment beyond that required by `T`; handle tails and unaligned accesses correctly.
- Partition parallel work into sufficiently large, independent chunks. Avoid false sharing between workers, oversubscribed thread pools, and per-element locks or atomics. Measure whether threading amortizes its scheduling cost for small shapes.
- Do not hardcode cache sizes, cache-line sizes, SIMD widths, or GPU execution widths from assumptions about one Apple device. Use documented capabilities where needed and qualify tuning on the target hardware.

## Allocation and memory ownership

- Load weights once. Reuse session KV buffers, scratch, pipelines, and other persistent resources. Aim for allocation-free steady-state numerical decode after capacity admission and warmup.
- Reserve bounded capacity or grow in coarse increments. Avoid `Vec` growth, repeated concatenation, and full-cache copying per token. Keep error formatting and verbose logging outside timed hot paths.
- Include weights, allocated KV capacity, scratch, tokenizer/application state, and temporary conversion buffers in memory accounting. Check allocation-size arithmetic for overflow before allocating.
- Avoid retaining redundant full CPU and GPU model copies in performance mode. Keep CPU reference code and small fixtures without requiring a second live FP32 model.
- Unified memory does not remove synchronization or lifetime requirements. Track CPU/GPU access and command completion explicitly, and count shared physical allocations once in accounting.
- Keep the initial Mac working-set budget from the plan. Qualify mobile memory limits independently; do not infer an iPhone budget from desktop RAM.

## Metal and quantized execution

- Keep GPU weights, KV, and scratch resident. Cache compiled pipelines and reuse command resources where supported and beneficial.
- Encode dependent operations without a host wait after each operator or layer once correctness is established. Preserve resource dependencies and buffer lifetimes until the GPU finishes.
- Minimize host readback. Ordinary generation should be able to return a token ID and small statistics; full logits and intermediate tensors belong in diagnostic paths.
- Design GPU access for adjacent lanes to access adjacent data where the kernel permits. Compare memory access, register use, occupancy, and dispatch overhead when tuning.
- Evaluate an available Apple matrix primitive before committing to a substantial custom GEMM. Keep API/device capability checks and compatible execution paths explicit.
- Fuse dequantization with matrix operations and, when implemented, cache dequantization with attention. Avoid expanding an entire quantized model or KV cache on every step.
- Treat fusion as an experiment: fewer dispatches or intermediate writes can be offset by register pressure or duplicated work. Retain it only after end-to-end measurement.
- A packed format needs an explicit version, bit order, group layout, scale convention, alignment, tensor exclusions, and reconstruction oracle. Do not assume two Q4 artifacts are compatible because they use the same container or bit width.

## Correctness before performance

- Preserve causal masking, grouped-query head mapping, rotary conventions and offsets, normalization epsilon, tied weights, and tokenizer/stop semantics.
- Track sampled token history separately from tokens committed to KV. Advance committed positions only after the full step succeeds. Cancellation and failure must leave a valid prefix or explicitly require replay.
- Reuse cache state only for an exact valid token prefix with matching model and position semantics. Handle fully cached prompts and pending sampled tokens explicitly.
- Preserve stable reductions and qualified accumulation precision. Reduced precision, reassociation, fast-math settings, weight quantization, and KV quantization require numerical and quality validation.
- Keep unsafe code small and localized. Document bounds, alignment, aliasing, lifetimes, initialization, synchronization, and device requirements that make it sound. Use safe Rust unless profiling justifies an unsafe optimization or FFI requires it.
- Never loosen tolerances or quality gates merely to make an optimization pass. Investigate discrepancies, including near-tied logits, and record any justified policy change.

## Measurement and verification

- Benchmark optimized release builds on named hardware and representative model shapes. Record the engine revision, build flags, model/tokenizer identity, precision, context depth, cache state, and workload.
- Separate loading/conversion/compilation, tokenization, prefill, decode, sampling, and user-visible latency. Time completed asynchronous GPU work rather than submission alone.
- Use the warmups, repetitions, raw samples, and acceptance policy in `PLAN.md`. Control power and thermal conditions, compare against the same workload, and confirm promising changes in a second run.
- Pair microbenchmarks with end-to-end measurements. Report memory and quality effects alongside speed; a faster isolated kernel may not improve generation.
- Record each optimization's hypothesis, implementation difference, correctness result, measurements, and retain/revert decision. Remove complexity that brings no useful measured benefit.
- Once the Cargo package exists, use `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`, and `cargo test` with the relevant supported feature configuration. Run targeted model/session/GPU checks for the behavior changed.
- Keep ordinary CPU tests independent of model downloads and Apple GPU availability. Run hardware and real-model suites explicitly. Report unavailable checks accurately; do not claim performance or device qualification from inspection alone.
- Documentation-only changes need link/content and whitespace checks, not a new test suite or unrelated engine benchmarks.
