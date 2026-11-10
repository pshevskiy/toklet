# MiniCPM5 native output precision audit

Every one of **442,863,360 paired original CPU coordinates** admits a finite
native-format value under both unchanged numerical bounds. A different
constructive method independently verifies such a value everywhere. There is
no output-grid obstruction in this captured domain. This does not qualify the
native graph: its prior 82 failures remain binding, and current managed native
adoption is rejected. Keep the existing research route and continue to LFM3
source/payload/graph qualification. Astra approves scoped diagnostic retention,
with no actionable findings.
The [CLI auditor](../tools/reference/check_minicpm5_precision.py) reports coverage,
output domains, source/runtime identity and explicit qualification limits.

The audit covers the original 128/512/1,025-token source prefixes, 51 stages,
both independent CPU profiles and every original coordinate. It does not replace
either original 10k-target/100-task quality comparison or change their caps.
Those comparisons remain unscored and mandatory if native numerical correction
allows reconsideration. The ordinary 512/chunk32 route and the separate deeper
diagnostic retain their existing scopes.

| Actual native snapshot domain | Stages | Paired coordinates |
| --- | ---: | ---: |
| Half | 50 | 225,480,960 |
| FP32 logits | 1 | 217,382,400 |
| Total | 51 | 442,863,360 |

The pinned Rust snapshot, execution and CLI mapping sources establish those
formats. Diagnostics convert Half to FP32 only after the completed chunk, so
that readback does not expand the set of admissible native values.

[Frozen audit contract](../artifacts/minicpm5-representability-2026-11-10/prospective-contract-2.json),
[original CPU reference](../artifacts/minicpm5-context1025-2026-11-10/reference-1/manifest.json).

## Exact predicate and independent verification

For each original CPU profile, acceptance remains the original binary64
predicate `abs(native - reference) <= atol + rtol * abs(reference)`. No gate
is enlarged, recomputed or fitted to these outcomes. Signed zeros have equal
numerical membership; NaNs and infinities are excluded.

The first method estimates the lowest candidate from an interval endpoint,
then verifies the exact discrete lower bracket with the actual predicate at
the candidate and its immediate native predecessor. The authoritative final
membership test calls the unchanged original reader. Failure to establish a
bracket rejects the audit itself. An independent stdlib ordered-bit binary
search checks scalar boundaries.

Continuous overlap alone would be insufficient: between `1` and
`1 + 3 * 2^-10`, equal bounds of `1.5 * 2^-10` meet at a value between adjacent
Half values. Synthetic Half/FP32 cases exercise this rejection. The actual
source domain has zero discrete and zero continuous exclusions.

The second method never calls the first method's search. It tries the rounded
midpoint, its two native neighbors and both original stored source values in a
fixed order. Every selected finite witness must independently pass both
original predicates. It succeeds at all coordinates. Per-group digests describe
the generated witness streams; no extra 1.32 GB witness file is retained. Missing
a witness with this method would be inconclusive, rather than an infeasibility
finding. Its contract openly records that primary feasibility was already known.

[Primary result](../artifacts/minicpm5-representability-2026-11-10/audit-1.json),
[fresh complete replay](../artifacts/minicpm5-representability-2026-11-10/audit-2.json),
[independent method contract](../artifacts/minicpm5-representability-2026-11-10/independent-witness-contract-1.json),
[constructive proof](../artifacts/minicpm5-representability-2026-11-10/witnesses-1.json),
[replay/source reconciliation](../artifacts/minicpm5-representability-2026-11-10/replay-and-source-checks-1.json).

## Checks, memory and CLI behavior

Eight new portable tests and 25 related portable tests in total pass. Scalar
search agrees with exhaustive finite Half membership for authored boundary
cases. The admitted numerical controls enumerate all 63,488 finite Half inputs
with adjacent peers, perform 271 independent scalar comparisons and include
FP32 extremes/subnormal/zero controls. The constructive method has 14 separate
literal cases.

Eleven actual CLI negatives reject bad bindings, guard flags and coherently
self-hashed runtime metadata. Existing files, directories and dangling output
links reject before environment/model-reference loading; exclusive creation
still protects the final write. One changed byte in an independent clone of the
actual CPU data is rejected before numerical execution, with the original
unchanged. A negative-runner error-text expectation was corrected; its initial
failure is retained and no invalid input was accepted.

[Portable log](../artifacts/minicpm5-representability-2026-11-10/portable-all-1.log),
[CLI controls](../artifacts/minicpm5-representability-2026-11-10/cli-negative-checks-2.json),
[actual byte control](../artifacts/minicpm5-representability-2026-11-10/raw-byte-negative-1.json),
[runner correction](../artifacts/minicpm5-representability-2026-11-10/cli-runner-attempt-1.json).

The complete campaigns have 172,359,680 / 172,408,832 / 164,937,728-byte final RSS
peaks under a separate 512 MiB audit guard, with closed owners and unchanged
inputs. Final physical peak is unavailable. The engine's 6 GiB ceiling, codec
allowance and 2 GiB disk floor are unchanged. NumPy 2.3.2, its actual native
implementation, Python/source files and fresh cold caches are externally
admitted. No MLX, Torch, model forward pass or GPU execution occurs.

[Primary owner](../artifacts/minicpm5-representability-2026-11-10/audit-checks-1.json),
[constructive owner](../artifacts/minicpm5-representability-2026-11-10/witness-checks-1.json),
[fresh replay owner](../artifacts/minicpm5-representability-2026-11-10/audit-checks-2.json).

The early output-path preflight was added after the first completed campaigns.
Its AST check proves every numerical/policy operation is unchanged. Exact
pre-change source copies reconstruct the historical pins; their receipts are
not rewritten. The fresh full replay preserves the complete semantic payload.

[Preflight-only AST check](../artifacts/minicpm5-representability-2026-11-10/preflight-only-ast-check-1.json).

The engine's Rust/native inputs remain unchanged, so their prior checks apply
only to those identical bytes. This source-domain tool is not a runtime speed,
telemetry overhead, graph fidelity, full quality, managed adoption, visible-frame
or iPhone qualification. Any future MiniCPM5 correction must still pass the
original numerical and complete quality gates.

[Review/commit and check reuse](../artifacts/minicpm5-representability-2026-11-10/publication-checks.json).

[Astra review](../artifacts/minicpm5-representability-2026-11-10/astra-review-result-1.json).
