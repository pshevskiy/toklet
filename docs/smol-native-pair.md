# Live native Smol model and codec ownership

Two additive ABI-v1 factories now connect the accepted
[resident model evidence](smol-model-content.md) and
[consumed codec pairing](smol-codec-content.md) to one live native model owner:
`toklet_model_load_mlx_smol_audited` and `toklet_chat_create_smol_audited`.
The [receipt](../benchmarks/results/2026-10-25-smol-native-pair-v1.json) records
source, required checks, three-target C links and separate actual Mac probes.

## Publication and lifetime

The audited model factory shares the existing native MLX loader boundary,
selecting the audited loader once before loading. Existing options version/size,
20 KiB bridge accounting, local-path bounds, output clearing and creating-thread
rules remain unchanged. Unsupported artifacts or builds reject without fallback.
The legacy MLX factory still constructs models with absent source evidence.

The audited chat factory checks thread and the fixed independent text allowance
before touching config/tokenizer files. CPU and identity-less Metal models reject
before directory access. Evidence comes from the actual supplied loaded model;
its directory is only an asset locator. The accepted codec loader authenticates
and parses the exact consumed buffers before publication. No model provenance is
inferred from directory names, matching geometry or later hashes of weight files.

Only after every fallible validation succeeds does the new chat handle clone the
exact existing `Rc<NativeModel>`. The immutable codec and this owner publish
together. No full weights copy, unsafe borrow extension or per-token reference
count change is added. A live audited chat keeps its model allocation alive after
the external model handle is released. Releasing the final audited chat frees
that allocation when no sessions/other owners remain. Legacy/synthetic chat
handles retain no model and preserve their existing release independence.

Callers reserve at least 64 MiB text allowance before loading the model. Keep
numerical/model memory admitted while any session or audited chat retains it;
count shared physical weights once. The additional optional cold owner is covered
by the text allowance. The 6 GiB maximum individual allowance is a boundary limit,
not combined process or phone admission. Later capabilities/session association
must compare the exact retained allocation; equal historical identities alone
cannot prove a live association.

## Checks and diagnostics

Four portable factory methods cover null/invalid/version/size/budget inputs,
output clearing, unaudited CPU rejection before files, wrong-thread access and
unchanged strong-owner counts on rejected publication. Two additional portable
observation methods exercise wrong status/oversized counts and invalid UTF-8,
retaining bounded actual buffers before comparison or decoding. The reused
capture helper changes only test-build visibility; its accepted failure behavior
is unchanged.

The explicit real Rust probe first rejects a legacy Smol model, then loads an
audited one. Rejected allowances/asset locators retain no additional owner. It
checks exact allocation equality and two owners, rejects wrong-thread creation/
release without losing either owner, releases the external handle, and encodes,
decodes and queries stops through the surviving chat. A weak observer proves
final destruction after chat release. No session or extra strong observer is
created. Every successful factory result includes a nonnull observation before
any dereference; no pointer addresses are printed.

The unchanged independent two-case chat fixture is authenticated by its literal
SHA-256 before parsing. Actual IDs, raw UTF-8 bytes, counts/statuses and stop IDs
are captured before comparisons. Malformed reported counts remain visible while
reads clip to valid buffer bounds. The strict C smoke independently checks old
record sizes and both old/new symbols, null outputs on failure, query/short-buffer
transactions, exact frozen Hello IDs/bytes, stops and retained release behavior.
It flushes bounded raw outputs before assertions.

Formatting, both strict checks, all 343 CPU/367 Metal-feature tests, both iOS
build targets and the four factory/two observation release methods pass. Twenty
verifier fault methods pass in a fresh pinned owned run. No gate or warning is
suppressed. Failed preliminary check/build attempts and superseded regression
metadata remain retained without final acceptance credit.

Three fresh release static libraries export exactly the 26 current header
symbols, including the two additions. Strict C compilation/linking passes for
Mac arm64/minimum macOS 15.0, iPhone arm64/minimum iOS 18.0 and simulator arm64/
minimum iOS 18.0; platform/minimum records are checked from load commands. The
iOS binaries are linked only. No device or simulator execution is claimed.

The separate actual Rust process retains 39 complete frames per side, 10,028
bytes each, with byte-identical actual/reference files. Both independent chat
cases pass; exact owner counts progress from two to one to final destruction.
Rejected factories retain no owner, and wrong-thread calls preserve both handles.
The C process retains 15 status records, three ID buffers and two raw byte
buffers, matching all 32 Hello IDs and 98 decoded bytes. Its model-release,
query/short-buffer and stop behavior passes. Chat structure size is 1,760 bytes;
the retained-owner option occupies 8 bytes within text allowance.

Hardware is MacBookPro18,3 / Apple M1 Pro / 16 GiB RAM / macOS 26.6.2. Both actual
processes fit their separate 1 GiB diagnostic ceiling:

| Diagnostic | Maximum RSS, bytes | Peak physical footprint, bytes |
| --- | ---: | ---: |
| Rust ownership/text | 223,526,912 | 206,897,848 |
| C ABI/text | 225,968,128 | 207,504,032 |

These are load/text/lifetime observations, not inference throughput or phone
memory measurements. The first two source-check attempts stopped on test-only diagnostics for a
constant assertion and an invalid UTF-8 literal, corrected without suppression. The initial C package
attempt stopped when clang could not create a temporary file. The accepted retry
uses its own workspace temporary directory, preflights all outputs, exclusively
reserves link destinations and parses symbols from retained first-buffer bytes.
An inherited preliminary-regression metadata flag was corrected in a fresh owned
run; later package-path updates are covered by the final unchanged-verifier run.

## Scope

This closes exact live native model/codec pairing and retention for the explicit
frozen Smol path. It adds no capability or response endpoint, UI/app integration,
architecture/quantization support, numerical forward, independent quality oracle,
phone runtime qualification or representative inference performance result.
Existing 64 KiB text, 512 input-token, 256 visible-output-token and numerical
context limits remain unchanged. All wider Gemma/source/precision/GPU/quality and
131072-output-plus-prompt/history gates remain open.
