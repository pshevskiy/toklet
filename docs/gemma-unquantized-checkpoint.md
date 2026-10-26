# Official Gemma E2B checkpoint authentication

The complete official conversion-lineage checkpoint is now downloaded and
authenticated in the local Mac cache. Its size, full SHA-256 and original
safetensors header match the identities frozen by the
[header/layout task](gemma-unquantized-layout.md). This closes payload identity
for the [unquantized quality baseline](gemma-quality-contract.md), with no model
arithmetic, fit, quality, app or full-window qualification.

## Exact source and stored result

The source is [google/gemma-4-E2B-it at70af34e](https://huggingface.co/google/gemma-4-E2B-it/tree/70af34e20bd4b7a91f0de6b22675850c43922a03),
the revision named by the four-bit candidate's conversion README. The verified
`model.safetensors` has10246621918 bytes and SHA-256
`2db5482b20d746879bb3ef79b5203e9075a2e2b98f54ec7c2f281c1477ddc550`.
Its263952-byte header matches SHA-256
`12740d6fe7a66b316040fa4d77471a8e1809498a71992b3364a6d5417d10662e`.
The copied4954-byte official config matches the pinned source config exactly.

Cache: `models/Gemma4-E2B-it-BF16-70af34e`. Full weights remain ignored; Git
contains source, small metadata and qualification evidence. No model code from
Hugging Face executes. The cache is a reference input, not a newly eligible
Toklet Chat/Expert or phone profile.

## One-file acquisition and publication

The [acquirer](../tools/experiments/m8_gemma4_unquantized_checkpoint.py) reuses the
reviewed response validator, minimal environment, process-group cleanup and
exclusive same-volume publication helper. It writes into one exclusive retained
read/write descriptor. There is no second full chunk/assembly copy.

Curl uses HTTPS-only redirects, no ambient curl config, identity encoding, exact
maximum file size, a30-second connection timeout and bounded stalled-transfer
handling. A3600-second global deadline covers transfer, hashing and publication.
The child inherits an exact body-file size limit; the parent's original soft
limit is restored immediately after spawning. Three diagnostic pipes drain to
individually capped2MiB files. The bounded child PID/command journal is written
immediately after launch and retained independently of the final receipt.

After curl and its process group finish, the owner hashes the entire body through
that same descriptor using1MiB buffers, compares the exact length/full SHA and
header bytes, and checks original descriptor/path identity. Source inputs are
rechecked before publication. Same-filesystem exclusive rename refuses an
existing destination; the published file must retain the authenticated identity.
Partial files/logs survive failure. Publication and receipt failure are reported
separately so a later error cannot claim that nothing was published.

The outer controller allows3750 seconds with60 seconds of cleanup grace,
giving the acquirer time to stop curl's separate group. It audits recorded curl
PIDs/groups from both child journal and final receipt on every exit path.
It then independently streams the complete published file to the pinned SHA.
This second verification uses bounded reads and creates no additional file copy.

## Actual checks, 26 October 2026

Seventeen [offline acquisition tests](../tools/experiments/test_m8_gemma4_unquantized_checkpoint.py)
and eleven unchanged transport regressions pass. Real local-child controls cover
three diagnostic caps, inherited file limits and restoration, deadlines,
cancellation, PID journaling/write failure and reaping. Descriptor/path mutation,
hash cancellation, existing outputs, exclusive publication, successful inode/
body/config preservation, post-publication mutation and receipt failure are
checked. The resolved output path must remain inside the repository before any
input access or network activity.

The actual transfer, full-file hash, header comparison, exclusive publication and
independent cache rehash all pass. Source inputs and protected release binaries
remain unchanged. All14 recorded outer/download groups across development and
collection are absent; nested-child ownership is explicit. The source and outer
controller were revised to close the cleanup-grace/journaling gap before network
execution; no failed network attempt or gate waiver was needed.

On Apple M1 Pro/16GiB/macOS26.6.2 (25G83), the completed acquisition command
reports2582.81 seconds, physical peak35881752 bytes and RSS41680896, below its
256MiB diagnostic policy. These describe acquisition, hashing and instrumentation.
They do not measure model loading, inference latency, numerical working set or
whole-app/device fit. Network conditions were not a controlled benchmark.

The [public evidence](../benchmarks/results/2026-10-26-gemma-unquantized-checkpoint-v1.json)
binds all742 inputs and13 raw outputs, exact development/live source hashes,
process receipts and the complete cached weight identity. The full payload and
raw HTTP diagnostics stay in ignored artifacts/cache.

## Remaining reference and delivery work

The frozen quality manifest is unchanged: its false download/verification flags
describe the earlier preparation capture. This separate qualification supplies
subsequent payload evidence for the exact same baseline identity, without
changing any workload ID, threshold or held-out selection.

Next qualify positive official row/layout binding, the chosen unquantized
reference arithmetic and short independent graph agreement, then measure
whole-process fit. Selected BF16 text storage alone exceeds6GiB, so a fully
resident unquantized text model is not admitted by the Mac policy. Bounded file
reading is a candidate route, with conversion overlap, IO and caller-retained
buffers accounted separately. Full1025-position quality execution, historical
precision gates, resident scheduling, model-driven Chat/Expert context and
independent phone fit remain open.
