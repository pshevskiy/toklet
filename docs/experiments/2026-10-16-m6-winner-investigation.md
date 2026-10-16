# M6 exact changed-winner investigation

The isolated diagnostic reproduces all 130 original larger-model 4K comparison
events exactly and captures the complete FP32 distributions for changed targets
90 and 127. All four 151,936-value distributions match the original SHA-256
identities. The diagnostic changes disk output only; its durations do not qualify
performance. The production engine and original paired worker stay unchanged.

## Capture and verification

The separately built Rust example uses the same shared immutable weights,
separate Reference/Online sessions, FP16 KV, chunk 128 and FullLogits execution.
After the existing completed readback, it writes selected borrowed distributions
to exclusive FP32 little-endian files. It adds no numerical operation or GPU
readback. At most eight selected pairs are admitted; this run selects two.

The collector requires the explicitly reviewed build SHA, original quality
record SHA, exact original requests/events, frozen token-prefix hashes, capture
inventory and full-distribution hashes. Raw capture events precede the unchanged
comparison events. The original worker ownership, 12-hour event timeout,
24-hour lifetime, EOF, natural exit, cleanup, identity and memory checks remain.
Independent full-vector work runs after the native worker exits. Final capture
inventory and hashes are checked again before publication.

The analyzer independently orders the full vocabulary with lowest-ID tie
breaking, verifies both winners and runners-up, exact FP32 candidate bits,
Reference top-two margins, maximum error, RMS and both target losses. Only the
independent FP64 sum/exp/sqrt replay uses a 1e-12 rounding comparison; stored
values, IDs, maxima, margins, hashes and all quality gates remain unchanged.
Signed-zero bit mismatches are rejected.

## Exact observed reversals

| Target / prefix length | Reference winner → Online winner | Reference margin | Online reversed margin | Maximum / RMS error | Target NLL change |
| --- | --- | ---: | ---: | ---: | ---: |
| 90 / 4,058 | 31,249 → 52,110 | 0.0023059844970703125 | 0.001552581787109375 | 0.019736289978027344 / 0.00418207346020154 | -0.00009231655115371495 |
| 127 / 4,095 | 14,856 → 1,075 | 0.0042743682861328125 | 0.003772735595703125 | 0.027275800704956055 / 0.005032867459099283 | -0.0026258361083160153 |

Both winners are unique. In each case the same top two IDs reverse order; the
remaining top-five order is unchanged. Differential candidate shifts are
0.0038585662841796875 and 0.008047103881835938 respectively, exactly accounting
for each Reference gap plus Online reversed gap. The gold IDs, 18,562 and 2,686,
are neither candidate. Their losses decrease in Online mode. These observations
do not establish task success or the cause of the differences inside attention.
The separately qualified full 10,000-target loss and 100-task gates remain the
model-quality evidence.

The qualification receipt binds the exact original record, each token prefix
and all four original full-distribution hashes. The accepted manual investigation
assessment applies only to those two observed cases, without a general margin
rule. Original automatic audits and capture analyses retain their unresolved
flags; manual review is recorded separately. Full eight-case long-quality
coverage still requires a successful fresh larger32K collection.

## Checks and measured memory

Fifteen portable Python tests, two portable Rust example tests, isolated
formatting and strict all-target Metal Clippy pass. Twelve negative replay
admission mutations reject incorrect reviewed build/anchor identities, role,
selection, capture directory and actual diagnostic format before projection.
Saved-data replay verifies 133 raw events, two complete distribution pairs and
6,843 bindings. The projection used for existing state/memory validation is
created only after verifying the original raw diagnostic coverage and exact
original comparison events; it is not relabeled collected evidence.

Complete paired admission is 2,445,934,228 bytes. Maximum sampled joint native
lifetime charge is 2,322,762,608 bytes, within 6 GiB. After natural worker exit 0,
the driver lifetime maximum sampled after full-vector analysis is 75,105,024
bytes; this is separate from the earlier joint worker measurement. Cleanup and
endpoint errors are empty, and all consumed identities/captures are unchanged.

The [qualification receipt](../../benchmarks/results/2026-10-16-m6-winner-investigation-qualification-v1.json)
and 16 exact published evidence copies preserve the source, build, checks,
raw capture events, four vectors and independent replay. Astra independently reproduced the saved replay and all vector analyses,
verified all six source pins, 16 evidence pins and 16 exact published copies,
and accepted the manual assessment of these two exact cases without findings.
All review execution and hashing finished before the next native worker.
Remaining context qualification and the final M6 decision stay open.
