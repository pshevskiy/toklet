# Qwen3.5 independent real-source fidelity

Q1.D validates the full coupled scalar graph at raw input ID1, position0.
The [Q1.C native captures](qwen35-hybrid.md) retain their original binary/source
identity. Both pass against an independent source-backed reference: each compares
32,134,656 values across2,102 arrays, with no failed comparisons or greedy
mismatches. Maximum gate ratios are0.164 for ideal FP64 and0.239 for staged
FP32. All native/reference greedy IDs are6503.

This is the first-input prerequisite. It exercises zero initial recurrent state
and zero rotary offset. Nonzero recurrence, longer contexts/offsets, generation,
codec/stops, quality, GPU execution and managed adoption remain separate gates.

The source remains the exact authenticated
`avan-ag/Qwen3.5-4B-Uncensored-MLX-4bit` revision
`fb98fbc59c71a567f3e23435386c9e93951261db`, SHA256
`0dfd6217f56cadcc373c90283d58d5e673833650fed1d6c299313fb536999043`.
All924 physical components bind426 logical owners:249 packed matrices and177
small BF16 vectors/four-tap tensors. Tied output uses the embedding owner.
Weights are already sanitized; no extra normalization offset or transpose is
applied. The odd119,837-byte payload origin is preserved.

The independent scalar math predates Rust composition. This change adds explicit
geometry, a streaming observation sink and a bounded projection adapter. The
original tiny manifest, weights and full reference regenerate byte-for-byte.
Real-source matrices use the existing affine row helper: little-endian nibbles,
group64, BF16 scales/biases, FP64 reconstruction rounded once to FP32. Projection
tiles accumulate in FP64 with declared output precision. Small BF16 values use
bounded byte reads without alignment assumptions. No full floating-point model,
native intermediate operand or target/MLX/Torch graph is imported.

The two profiles emit all1,051 complete stage/state/logit arrays directly to a
192,807,936-byte file, below256MiB. The frozen ledger stays below6GiB; measured
reference lifetime physical peak is539,182,160B. The owned process exits cleanly,
its group is absent, source authentication closes with a complete second hash,
and admitted numerical/native runtime origins are checked at exit. These are
reference resource observations, not engine performance or telemetry-overhead
qualification. Native captures retain their earlier2.638GB peaks.

The reader authenticates sealed files, frozen producer/contract identities,
runtime bindings, strict array types/geometry/uniqueness, complete body coverage,
finiteness and lowest-ID greedy selection. Every value uses the original
`1e-5 + 1e-4*abs(reference)` gate. It validates each native capture with the
existing source/accounting reader before comparing the full coupled arrays.

Nineteen Python controls and nine damaged-evidence controls pass. The latter
reject boolean counts/tokens, a live group, scope escalation, an unclosed owner,
a changed frozen source pin and a body change with updated outer hashes. All
original captures remain unchanged. Required Rust/native checks are reused with
all3,923 relevant source/test/fixture/Cargo/workload inputs unchanged; their
recorded631CPU/765Metal results and native53 package/C smokes keep their original
identities. This task changes Python tooling and documentation only.

One new failure-path test initially used an unresolved macOS temporary-path
alias. A freeze was also assembled before checking that failed control result.
It was invalidated before any inference; the corrected14-control result gates
the actual freeze. Failed records remain in the iteration evidence. Numerical
arithmetic, tiny fixture bytes and acceptance tolerances were not adjusted.

Astra approves the combined code, invariants, checks and unchanged-input reuse
with no findings. The next numerical step is a frozen
multi-token/offset replay that exercises recurrent and convolution history,
causal KV state, chunk equivalence and failure recovery on the actual source.

[Contract](../artifacts/qwen35-real-oracle-2026-11-08/acceptance-contract.json),
[freeze](../artifacts/qwen35-real-oracle-2026-11-08/adapter-freeze.json),
[verification](../artifacts/qwen35-real-oracle-2026-11-08/verification.json),
[review/commit](../artifacts/qwen35-real-oracle-2026-11-08/publication-checks.json).
