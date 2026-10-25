# Smol codec pairing with authenticated source content

`app_chat::Codec::load_smol_audited` validates the exact consumed model config,
tokenizer config and tokenizer JSON against the known Smol profile and supplied
[historical source evidence](smol-native-content.md). It publishes an immutable
`CodecContentIdentity` only after the authenticated tokenizer buffer parses and
passes every existing vocabulary, special-ID, truncation and padding check.
The [receipt](../benchmarks/results/2026-10-25-smol-codec-content-v1.json) records
source, checks and an explicit Mac diagnostic.

## Ownership and admission

The caller reserves a separate 64 MiB codec allowance before loading. Too-small
allowances or unsupported source fields reject before file opening or parsing.
The source must match the exact repository/revision, four authenticated hashes,
694 physical ranges, payload/container sizes and at least 694 verified reads.
Repeated successful reads remain valid. The final file components are opened
without following links, in nonblocking mode, and must be regular files. Parent
directories may be links; authority comes from content, never a path or geometry.
Reads are bounded to 16 KiB for each config and 4 MiB for tokenizer JSON.

Hashes cover the same owned buffers used for profile classification and tokenizer
construction. No file is reopened to infer which bytes were previously consumed.
The tokenizer parser and policy checks are shared with legacy loading; legacy
profile selection and asset opening retain their existing behavior. Legacy and
synthetic codecs carry absent identity. Encoding, decoding, templates, stop IDs
and numerical kernels are unchanged. No hashes enter token loops.

Identity copies the supplied opaque source value plus two consumed token-asset
digests. It proves that these assets matched that authenticated source content.
Source-only loading can supply the same evidence; copying it out of a resident
model does not prove ongoing ownership of that model. The value survives model
and codec drops and later disk changes. It is not a resident-byte digest, current
disk snapshot, native capability or proof of association with a live model.
An engine-owned model/codec boundary remains a separate task.

## Diagnostic and verification

Eight portable methods cover admission before missing-path access, every source
field/count mismatch, valid repeated counts, bounded regular reads, links/FIFOs,
saved-buffer authority after synthetic path changes, altered assets, parser/policy
failures and absent legacy evidence. Capture fault tests retain original codec
errors and a bounded active pair alongside independent capture errors; they
exercise first-side write failure, second-side retention, bounds/serialization
failure and later comparison failure with both complete records preserved.

The actual diagnostic reserves two codec allowances, two 16 MiB capture bounds
and 16 MiB scratch/overhead before model allocation. The remainder of a 1.5 GiB
Mac process policy is passed to the audited resident model loader. It obtains
source evidence from the successfully loaded model, creates both legacy and
audited codecs, then drops the model and checks the historical evidence persists.
No numerical forward or session is executed.

Every result is captured for both codecs before comparison. Complete prior frames
and any partial late frame survive errors; the owned diagnostic failure also
retains a bounded active pair before fallible serialization/admission/writes.
JSON frames are bounded to 1 MiB. Actual codec output bounds fit that reserve,
including debug escaping; arbitrary fault-test values report truncation explicitly.

The existing two-case independent chat fixture is authenticated against its
historically accepted literal before parsing. The diagnostic requires exact
repository/revision/tokenizer/config metadata and exactly two cases. It checks
all 49,152 singleton decodes, the independent encode/full-decode cases and each
chat prefix. Singleton/prefix agreement is a regression through the same tokenizer
implementation, not a new independent tokenizer or quality oracle.

Formatting, both strict checks, all 337 CPU/361 Metal-feature tests and both iOS
build targets pass. Eight release methods pass; the real diagnostic is separately
selected exactly once. Seventeen verifier methods pass in a fresh owned wrapper
with pre/post source/helper/interpreter/fixture pins and bounded raw logs. The
preliminary direct regression is retained without owned-execution credit.

The completed actual Mac run retains 49,258 frames per codec, including 49,152
singletons, two encode/full-decode cases and 102 prefix comparisons. Each capture
contains 4,962,863 bytes; complete files are byte-identical. Audited evidence
matches all source hashes and 694 verified reads/ranges; legacy evidence is absent.
The identity occupies 224 bytes, its option 232 and the complete codec 1,744 bytes
on this arm64 build, within the independent text allowance.

Hardware is MacBookPro18,3 / Apple M1 Pro / 16 GiB RAM / macOS 26.6.2. Completed
maximum RSS is 244,580,352 bytes and peak physical footprint 209,289,936 bytes,
both below the separate 1.5 GiB diagnostic ceiling. This is a compatibility/load
observation, not inference throughput, latency or phone memory qualification.

Three failed preliminary check attempts remain retained without acceptance
credit: the initial controller stopped on its own logging bug after cleaning;
a malformed-tokenizer test changed an added-token ID that tokenizers normalized
back to the actual vocabulary; its correction initially omitted a test import
qualification. The final fault mutates the real model vocabulary and verifies the
parsed effect before policy rejection. No production tolerance or gate changed.

## Scope

This closes one consumed-codec-content prerequisite. Native handles, live
model/codec association, capability/response production, app integration, device
admission and phone qualification remain separate. Existing 64 KiB text,
512-input-token and 256-visible-output-token bounds are unchanged. No additional
architecture or quantization format becomes supported. All earlier Gemma/source/
GPU/precision failures and full graph, quality/performance and 131072-output
requirements remain open.
