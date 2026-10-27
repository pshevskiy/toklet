# Experimental C512 envelope: derivation checked, qualification rejected

[gates.json](gates.json) preserves the independently derived CPU calibration
envelope unchanged. [review.json](review.json) records Astra's separate decision:
**this envelope is rejected as a qualification contract; validation must remain
unopened.** The artifact's original pending-review status is historical; the
separate review decision governs its use. No production tolerance changes.

The [frozen calibration](../qwen3-context-calibration-v2/README.md) supplies all
inputs. The57 exact stage/layer keys pool only six CPU-FP32/FP64 comparisons and
six equivalent-prefix pairs under both arithmetic profiles. The predeclared
formula remains `max(1e-4, 2*cross_precision_max, 2*prefix_max)` with `rtol=1e-3`.
Historical C512 failures, native outputs and validation arrays are excluded.

Each derivation performs1026 bounded comparisons with36 parent admissions.
Full parent hashes/endpoints authenticate shorter prefixes, while full arrays
also match their frozen array hashes. Maximum coordinates and actual value pairs
are retained. Gate artifacts are returned only after all endpoint checks close.
Two cold derivations produce identical gate bytes. A separate admitted NumPy
replay uses a different block size and independently checks every comparison,
maximum witness, consumed hash, pooled maximum and final formula result.

The envelope is too permissive for qualification without an independent error/
quality budget. In the prose512 calibration case, position317 drives a24.5
layer27 output difference,88.125 versus112.625, and therefore blanket `atol=49`.
The same position's maximum logit difference is1.1971473693847656, yielding
`atol=2.3942947387695312` for all validation rows. Every calibrated prefix maximum
is zero. Reproducibility and an exact argmax check cannot establish distribution
or quality preservation under those blanket allowances. This is not evidence
that native execution is incorrect or model quality fails, and no replacement
ceiling is invented from these observations.

All18 cold tests pass, including five gate/formula/prefix/authentication controls.
A report-label collision was corrected by separating labels from span audits;
original reports remain in `artifacts/m8-qwen3-context-gates-checks-v1`.
Corrected cold reruns require the identical original gate SHA
`40cf0fd86ee1b67f3fe9719b23e2942c8a2b21fbec6d91b7cf99bfc547aef6d3`.
Six derivation/test groups and the independent publication group close.

On the same M1 Pro/16GiB Mac, corrected derivation physical/RSS peaks are
294,355,880/276,938,752 and297,436,144/292,339,712 bytes. Independent replay peaks
304,202,640/320,536,576 bytes; all fit512MiB. No latency or inference qualification
is inferred. See [public evidence](../../../benchmarks/results/2026-10-27-qwen3-context-gates-v2.json).

Next pin and execute the original quantized model's source implementation and
precision profile on calibration inputs. Keep the unquantized baseline and
complete10,000-target/100-task quality demand separate, without clipping to512.
The untouched validation split is preserved for a justified future contract.
Historical failures, full model-window and phone requirements remain open.
