# Original-BF16 short source reproduction and profiling

The instrumented source path reproduces all 2,852 saved states and the complete
source-read chain exactly. This is the original/future four-token regression
under the existing mathematical and staged-FP32 profiles. It does not execute
upstream BF16 arithmetic or score model quality. [Prospective policy](gemma-original-quality-next.md).

Original IDs are `[2,100,257,100]`; the existing changed-future control uses
`[2,100,257,101]`. The authenticated 10,246,621,918-byte checkpoint is read through
its existing descriptor and BF16 row adapter. All 278 matrices and 262 direct
vectors retain their bindings. Every state hash, shape, dtype and logical offset
matches the saved capture; all greedy IDs/margins match. Three available next-
token NLLs per case/profile are recorded. The independent planned read chain
matches exactly: 38,182 reads / 18,236,693,062 bytes. No full floating-weight cache
is created.

The correctness worker takes 47.11 seconds, with peak RSS 372,899,840 bytes and
physical footprint 290,194,464 bytes under the 6 GiB policy. Authentication takes
5.76 seconds and binding 0.33 seconds, separately from the graph measurements.

| Case / profile | Graph seconds | Read/guard/hash | Decode/validate/guard | Projection + FP64 conversion/framework | Tied-head remainder | Capture |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original / mathematical |9.173 |2.588 |1.676 |2.704 |0.501 |0.153 |
| Original / staged |9.744 |2.819 |1.751 |2.906 |0.540 |0.133 |
| Future / mathematical |9.976 |2.913 |1.764 |2.972 |0.578 |0.149 |
| Future / staged |9.506 |2.867 |1.689 |2.803 |0.430 |0.128 |

The two projection columns are remainders after nested reads/decoding; they are
not pure GEMM timings. Other graph work accounts for roughly 1.55–1.60 seconds
per case. Inclusive head times are recorded separately in the evidence and
must not be added to these exclusive columns. This single instrumented probe
identifies source cost; it is not a decode-rate or optimization-retention claim.

The owned worker exits cleanly with runtime endpoint, complete source/dependency
hashes, finite outputs and resource checks passing. Uniform/shifted/nonfinite
NLL controls pass; the previously reviewed adapter's two cold equivalence controls
remain valid. No relevant Rust input changed after the latest required checks.
Final Astra review is tracked in the evidence.

Next is tiny upstream resident-versus-streamed validation under the pinned
Transformers/PyTorch BF16 operators, before the descriptive real upstream
comparison. Retain the eleven original-source cross-precision failures, earlier
27 packed mathematical failures and the 1,025-position four-logit/eight-KV
rejections. No tolerance or model/app eligibility changes follow.
[Evidence](../benchmarks/results/2026-11-04-gemma-bf16-source-experiment-v1.json).
