# Bounded LFM2 evidence I/O

B4e3c2a adds a test-only writer and terminal borrowed-buffer observation. It
changes no production numerical kernel, transaction or model factory. The full
real-model workload and independent reader remain separate tasks.

The writer emits original stored Half/Float32 words in little-endian order,
including signed zero and diagnostic NaN payloads. It uses64KiB encoding scratch
and exclusive files, with256MiB raw/16MiB journal caps. Array descriptions are
at most384B; header262144B, boundary40000B and other events512B. Shape products,
scalar widths, offsets, record counts and output extents are checked. Short or
long iterators retain partial evidence and permanently fail the writer; errors
cannot turn into a later successful footer.

Footer hashes are noncircular. prefix() reports exact files before the end
event; finish() reports complete files including that event, for stdout and the
owned outer receipt. A later reader must check both scopes and reject missing
or trailing bytes. This module records observations; it does not interpret
numerical source gates or declare model eligibility.

The cache and snapshot visitors use the existing command-completion authority
before borrowing buffers. Callback errors propagate immediately. Cache status
metadata includes its actual address, length, device and words. Packed scales/
biases borrow one allocation at a time; the caller establishes terminal or
not-submitted state. Callbacks are statically dispatched, synchronous and outside
forward paths. Existing owned Tiny observation APIs remain unchanged.

Four portable controls cover raw words/ranges/SHA, complete versus prefix hashes,
exclusive creation, shape/overflow/width/order checks, caps, malformed iterator
lengths, retained partial bytes and sticky failure. The explicit ignored Tiny
GPU check compares borrowed observations to existing owned APIs for4 cache
buffers,57 snapshot allocations,36 packed metadata buffers and status. It also
checks callback failure and enqueued-command read refusal before completing a
real stamp-only command. All103 diagnostic records precede final assertions.

This hardware check uses one Tiny three-token forward, two phase commands and
one extra stamp command. It makes no new source-numerical, real-model, app,
cancellation, phone or performance claim. All four CPU writer controls pass in the Metal suite. Formatting, both strict
Clippy configurations, default457/Metal560 tests and the release Metal library
test build pass (20/182 ignored; counts overlap). Astra reviews the exact
optimized executable before the one Tiny GPU test and audits the saved results.

All103 records and98 unique observed addresses pass. The36 packed scale/bias
hashes reconstruct exactly from original BF16 fixture bytes, padded to4-byte
buffer length; these metadata buffers are not widened Float32. Cache/snapshot
parity has completed reviewed native whole-word assertions; saved logs contain
hashes/booleans rather than full raw words and do not establish a second full
reconstruction. Both callbacks stop after one injected error. All three pending
read attempts fail before callbacks; the actual stamp command then completes
successfully with offset3 and valid prefix retained before discard.

The raw log is25,779 bytes with empty stderr. Twenty-one direct samples observe
maximum resident17,809,408 bytes and lifetime physical-footprint6,619,736 bytes;
these sparse observations are not complete process/aggregate peaks. All5988
current package bindings and seven current owned groups pass endpoint checks.
The saved reader initially expected a named enum Debug string; the actual
wrapper prints Completed as numeric4. The pinned dependency confirms that
value; the corrected decoder and initial reader are retained, without changing
native data, code or criteria. [Exact evidence](../benchmarks/results/2026-10-29-lfm2-capture-io-v1.json)
pins source, receipts, executable and saved audit. Real-model capture, complete
reader/state/text/memory validation and application adoption remain pending.
