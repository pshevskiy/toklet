# Gemma final-logit softcap witness

The independent fixture freezes `cap*tanh(logit/cap)` before a native operator.
It runs the unchanged compiled `logit_softcap` from pinned MLX-LM revision
`0d4d3d04f8ab7ce4a51e7209e8615755a48ba94d`; `gemma4_text.py` hashes to
`73ffa75e9069ce7b2f864ccd8f0267bb80c5b5d4e2b73102b8a2a93fe03f936d`.
The original compile decorator remains active; only annotations are deferred.
The [receipt](../benchmarks/results/2026-10-25-gemma-softcap-reference-v1.json)
binds source/runtime, raw output, mathematical replay and check evidence.

## Contract and scope

Inputs are contiguous FP32 vectors; cap is finite, positive and exactly FP32.
The candidate's cap is30. Synthetic operator cases also use0.5 and1/1024, without
qualifying another checkpoint configuration. The FP64 ideal computes the whole
expression in FP64. The staged contract rounds the quotient to FP32, evaluates
tanh in FP64 and rounds it to FP32, then multiplies by cap with FP32 output.
For small caps and finite FP32 extrema, that quotient can become infinite while
the tanh and output remain finite. Four such staged divisions are counted
explicitly across two cases; nonfinite intermediates are not serialized or
silently replaced by ideal results. A future native FP64 implementation can
avoid these infinities while satisfying the same external numerical gate.

Six bounded cases cover an empty no-call, hand scaling/near-zero/zero values,
a129-point grid, saturated finite tails including FP32 extrema, and the two
alternate caps. Hand counterexamples distinguish omitted division, omitted final
multiplication, identity and clipping. Portable mathematical tests check odd
symmetry, monotonicity, magnitude bounds, signed-zero bits, independent chunks
and immutable inputs. Signed-zero bit qualification is mathematical/native-only;
the upstream comparison retains raw bytes but uses the stated numerical tolerance.
Cap admission rejects Boolean, complex, nonscalar, nonfinite, nonrepresentable,
nonpositive and out-of-reference-range values, including huge integers before
floating conversion can overflow.

The selected decorated helper executes on actual MLX CPU/GPU with output-only
observations. No helper-internal quotient/tanh capture or fusion qualification
is claimed. Both devices retain five raw outputs and pass ten comparisons
against ideal/staged contracts. Maximum error/policy ratios are0.000821852/0 CPU
and0.001249045/0.001176542 GPU under unchanged1e−5+1e−4×abs(expected).
Independent stdlib `tanh`/FP32 replay verifies18 arrays/585 values, exact staged
bits and zero signs; the largest ideal difference is3.552713678800501e−15,
within1e−13+1e−13×abs(expected).

Seven portable, three owner and three runtime-admission methods and all six
owned commands pass. The [producer](../tools/reference/gemma_softcap.py),
[portable tests](../tools/reference/test_gemma_softcap.py),
[controller](../tools/experiments/gemma_softcap_reference.py) and
[owner regressions](../tools/experiments/test_gemma_softcap_reference.py) retain
exclusive outputs, source/protected guards, earlier outputs after later failure,
independent failure-log capture and joined owned process cleanup.

Native implementation, disabled-cap model routing, constructor/config selection,
complete-vocabulary graph, BF16/quantized checkpoint, device fit, quality,
performance and long replies remain open. The eight
[upstream rotary discrepancies](gemma-rotary-precision.md) remain unwaived.
