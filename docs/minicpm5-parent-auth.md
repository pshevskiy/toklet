# MiniCPM5 bounded parent authentication

This source audit streams both pinned BF16 parent files without storing their
payloads or loading a numerical model. Both complete SHA-256 hashes match the
pinned LFS identities and both TLS transfers close successfully. This closes
the whole-parent identity gate in the recorded scope.
Quantized conversion lineage, codec/task quality and model adoption stay separate.

## Source and bounded stream

The selected MLX8 child remains
`bumblebuttpow/MiniCPM5-2B-heretic-abliterated-MLX-8bit` at `50ab1b22`.
The declared parent is `insraq/MiniCPM5-2B-heretic-abliterated` at
`8d83eaf42e30f4e4f164b8685be0ecdccd8ef0d8`. Cached pinned API/index/header evidence
binds the two parent object sizes and LFS hashes before any new payload result:

| Parent object | Bytes | SHA-256 |
| --- | ---: | --- |
| model-00001-of-00002.safetensors | 4,989,507,944 | `2a3ac0fb68423d0e359a0a8a937c8a295ddaea08c364efdeef7631b739d6425b` |
| model-00002-of-00002.safetensors | 44,049,184 | `9f35f94773989092e6d19a527c1e95a7be0252eae0fdcf50310eff4f6e26e11e` |

The [prospective contract](../artifacts/minicpm5-parent-auth-2026-11-10/prospective-contract.json)
freezes this scope before parent payload outcomes. The verifier authenticates the
complete child object before and after the audit, verifies the cached metadata
pins and all 381 BF16 logical tensor shapes, then consumes parent bytes in at
most 4 MiB blocks. It hashes the whole object, each tensor and each block during
the same pass. The 85 unquantized norm arrays must match child bytes exactly.
No matrix conversion or quantizer equivalence is inferred from norm equality.

Partial tensor journals remain explicitly unauthenticated until complete object
hashes and closed transfers pass. Payload blocks are discarded immediately.
Retained artifacts contain only headers, transport logs, tensor/block digests,
resource observations and manifests. The campaign monitors its owned process
group against 256 MiB and the unchanged 2 GiB disk floor. This smaller source
budget does not change the engine's 6 GiB or codec's 512 MiB policy.

## Commands and offline witness

```text
python3 tools/reference/minicpm5_parent.py PINNED_METADATA_DIRECTORY CHILD_SAFETENSORS FRESH_OUTPUT
python3 tools/reference/check_minicpm5_parent.py OUTPUT PINNED_METADATA_DIRECTORY CHILD_SAFETENSORS OWNED_RECEIPT SOURCE_FREEZE FRESH_SEAL
```

The recorded campaign controller adds owned-process memory/disk/deadline guards
around the first command. Every output destination is fresh. Curl ignores local
configuration, follows HTTPS only, requests identity encoding, enforces object
size/time bounds and captures its final transfer receipt. A cache-busting query
changes no pinned revision or byte/hash requirement. No remote model code runs.

The offline checker validates known object identities; tensor/block coverage,
absolute offsets, sizes, exact norm digests, journal agreement, raw transport
receipts, command bounds and sequential clocks enclosed by the owner. It seals
the retained evidence hashes. Re-running it to a fresh seal must reproduce the
saved seal exactly while source/evidence remain unchanged.

**Discarded payloads cannot be rehashed offline.** Block hashes are producer
witnesses tied to the checked complete stream and sealed evidence, not Merkle
membership proofs. A fresh whole-file authentication requires another stream;
structural validation alone does not repeat that authentication. Broader lineage
and quality approval cannot be manufactured by changing report flags.

## Checks and remaining work

Eight producer tests use independent literal BF16 bytes and short reads. Seven
consumer tests cover coverage, balanced extents, exact norm digests, pinned TLS
commands, clocks and resource/qualification flags. They require no network,
model payload or GPU. The initial test-refinement command used a repository path
from the reference directory; its edit failed, then the old suite passed. The
path and actual multi-block assertions were corrected before the fresh eight-test
receipt. That first attempt is not substituted for the corrected test result.
[Producer tests](../artifacts/minicpm5-parent-auth-2026-11-10/portable-tests-1.json),
[consumer tests](../artifacts/minicpm5-parent-auth-2026-11-10/consumer-tests-2.json).

All 4,023 required Rust/build/fixture inputs and 419 native inputs remain identical
to the reviewed RMS milestone. Its formatting, strict CPU/Metal Clippy,
670 CPU/811 Metal tests, native65 three-slice package and C smokes are reused by
exact hashes. New Python verifier checks are fresh; no engine benchmark or
numerical/device qualification is inferred from this source audit.
[Input reuse proof](../artifacts/minicpm5-parent-auth-2026-11-10/required-native-reuse-1.json).

The complete owned campaign authenticates **5,033,557,128 bytes**, 381 tensors
and 85 exact copied norms. The manifest retains **1,349 block digests** and zero
payload bytes. Its process group closes successfully. Exact final auditor RSS
is 194,953,216 bytes; curl peaks are 8,192,000 and 8,208,384. Their conservative
sum is 211,353,600 bytes, below the 256 MiB source budget. Maximum sampled
aggregate RSS is 202,997,760 bytes; sampled aggregate lifetime physical counters
reach 31,409,016. Final physical peaks remain unavailable. Minimum observed free
disk is 3,334,672,384 bytes. These are source-process observations, not engine
memory savings or default-allocator model fit.
[Owned campaign](../artifacts/minicpm5-parent-auth-2026-11-10/auth-checks-1.json),
[full source witness](../artifacts/minicpm5-parent-auth-2026-11-10/parent-auth-1/manifest.json),
[sealed offline report](../artifacts/minicpm5-parent-auth-2026-11-10/witness-seal-1.json).

Actual-output controls pass one copied-witness positive and 27 structural rejections. A
well-formed substituted multi-block hash remains structurally possible because
the payload was discarded; changing it changes the retained seal. That explicit
post-seal test passes as the second structural acceptance. No offline payload rehash or rejection of every possible
resealed false block is claimed. Original evidence and the child object remain
unchanged; there is no new network/model run for these controls.
[Controls](../artifacts/minicpm5-parent-auth-2026-11-10/witness-controls-1/checks.json).

The fresh seal reproduces byte-for-byte, and all 15 portable tests pass together.
[Final checks](../artifacts/minicpm5-parent-auth-2026-11-10/final-checks-1.json).
Astra approves scoped source-authentication retention with no actionable findings.
Only administrative documentation closure follows; reviewed code/contracts stay
frozen. [Review](../artifacts/minicpm5-parent-auth-2026-11-10/astra-review-result.json),
[commit/integration](../artifacts/minicpm5-parent-auth-2026-11-10/publication-checks.json).
The whole improvement goal continues with matrix lineage,
codec/template/stop handling, original 10k-target/100-task quality and broader
console workloads. Managed adoption and physical iPhone work remain open/deferred;
screen capture remains deferred by the engine/CLI priority.
