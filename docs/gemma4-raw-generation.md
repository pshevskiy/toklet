# Short raw Gemma4 E2B generation on Mac

The bounded [generation-state wrapper](gemma-generation-state.md) now has an
explicit real-checkpoint probe against the previously frozen
[independent greedy witness](gemma4-greedy-reference.md). The probe uses the same
borrowed scalar model, session-owned KV/scratch and existing greedy generator.
It changes no numerical kernel, generation policy, app profile or public limit.

## Scenarios and diagnostic ownership

Input IDs are `[2,100,257,100]`. Four schedules each sample four IDs:

- Uninterrupted output of four tokens.
- Two tokens followed by a continuation request for two more.
- Cancellation from the callback after two visible tokens, followed by two more.
- Caller-provided stop ID107, followed by continuation with an empty stop list.

The caller-stop scenario leaves107 sampled and pending but un-emitted. Its
continuation evaluates that pending ID once and emits the remaining three
tokens. This checks the explicit caller mechanism; it does not resolve model
EOS metadata. Every scenario starts with a reset and ends with eight history
IDs, seven committed to KV and one sampled ID pending. Pending state hides the
old distribution from ordinary callers.

An example-only observer statically delegates the existing session interface.
It captures each full `[262144]` FP32 vocabulary row inside the sampling method,
before the generator appends the selected ID. Callback-time access therefore
does not weaken the session's logits invariant. Sixteen rows occupy16MiB.
The observer streams each row through a64KiB writer without retaining full
vocabulary copies. Small bounded descriptors/prefixes/reports belong to this
diagnostic; they are outside numerical admission and inside measured process
memory. Each descriptor precedes its write, and raw bytes are flushed before
finite-value rejection. Ordinary failures preserve completed and incomplete
descriptors plus the capture error. The child has no graceful signal-finalizing
handler; its controller captures a partial raw file even if no final JSON exists.

## Verification

Three offline example methods check finite bit patterns, hashes and offsets;
invalid shape/prefix/global/per-scenario bounds before writes; nonfinite bytes
retained before rejection; and a real read-only-file write failure retaining an
incomplete descriptor and the original OS error. All302 CPU and324 Metal-feature
tests pass, with16/109 explicit hardware/model ignores. Formatting, both strict
lint configurations, both iOS targets, the release example build and five release
generation-state methods pass in an isolated target directory.

The real-model controller verifies every sampled prefix and selection, every
emitted sequence, all seven report counters and stop reasons, final pending
state and memory-category sum. It compares every vocabulary value against both
frozen precision contracts under unchanged `atol=1e-5, rtol=1e-4`. A final
collector independently replays these observations from the first captured raw
buffers and binds the exact executed release binary to its checked build.

## Observed Mac results

All four scenarios select `[107,100,45518,107]`. Each captured prefix is exactly
the corresponding frozen prefix. All16 full rows are bitwise identical across
the four schedules. All32 full-vector precision comparisons pass over8,388,608
paired values; the maximum error-to-bound fraction is0.5103206474158878 against
FP64 and0.0013656471300556837 against staged FP32. These selected rows do not
waive the earlier full-prompt27 native FP64 failures or five-array/57-value
mathematical cross-precision gap.

The numerical memory categories sum to2,765,262,700 bytes, including
2,671,706,564 bytes for the model and its retained payload/metadata. The session
portion is93,556,136 bytes. The MacBookPro18,3 / Apple M1 Pro /16GiB process on
macOS26.6.2 build25G83 reports2,143,485,952 bytes maximum resident memory and
2,710,440,456 bytes peak physical footprint, both below6GiB. Its138.13-second
combined load/four-scenario/output observation uses uncontrolled filesystem,
power and thermal conditions; it is not a latency or throughput benchmark.
See the [complete bound receipt](../benchmarks/results/2026-10-25-gemma4-raw-generation-v1.json).

## Remaining scope

This is short raw scalar continuation/state evidence. The public execution
limit remains32 positions;512 local physical slots do not grant512-position
execution. Full G3, original-BF16/source qualification, all earlier precision,
GPU-intermediate and long-rotary failures, tokenizer/chat/model stop semantics,
resident Metal, quality, representative performance, phone fit, app integration
and131072 output tokens remain open. No ABI, family, catalog or position-limit
change is made.
