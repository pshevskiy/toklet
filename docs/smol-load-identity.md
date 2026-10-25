# SmolLM2 consumed-byte witnesses

The offline tools produce and independently audit raw tensor-range evidence for
`mlx-community/SmolLM2-135M-Instruct-8bit` at
`0f0d9b8218915bc34d401e1a340b8c049d300d5e`. This is a prerequisite for a future
qualified native loader. It adds no loader API, inference, model capability or
phone qualification. The [receipt](../benchmarks/results/2026-10-25-smol-load-identity-v1.json)
records the reviewed source, frozen data, executed commands and retained outputs.

## Bytes and ownership

[The producer](../tools/reference/smol_range_witness.py) authenticates six local
regular files before parsing their metadata. It retains the exact config/index,
eight-byte container prefix and raw header. A sequential pass updates the full
container SHA-256 and each physical tensor's SHA-256 from the same chunks, then
requires the literal full-container pin before publishing. Matching file hashes
before and after loading alone would miss a mutation restored between those
observations; the consumed-byte pass is the required evidence.

The 143,030,401-byte container contains 694 contiguous physical tensor ranges
covering 142,955,136 payload bytes: 211 U32 affine matrices with F16 scales/biases,
and 61 F16 norm vectors. There is no explicit output-head tensor. Six separate
embedding weight/scale/bias reads cover input and tied-output roles, each with
its own consumed-byte digest. The tools stream bounded chunks without making a
second checkpoint or tensor-data copy.

[The independent replay](../tools/experiments/smol_range_replay.py) imports no
producer code. It derives the expected inventory from the authenticated config,
reconstructs the whole container with 32 KiB chunks rather than the producer's
64 KiB, rereads all 694 ranges in reverse name order and repeats the six tied
reads. Both tools reject symlinks and nonregular files and bound metadata sizes.
The [fixture](../tests/fixtures/smol-load-identity-v1/manifest.json) preserves
exact source bytes; its config/index intentionally have no final newline.

## Validation

Twelve portable synthetic test methods cover read/restore mutation despite equal
endpoint hashes, repeated-read changes, range/schema/overflow/short-read errors
and regular-file requirements. Sixteen artifact-only runner methods cover the
owned process, interruption, capture and first-buffer admission mechanisms.
These runner helpers remain experiment infrastructure rather than product code.

After source relocation, a fresh reviewed controller ran all 28 methods, two
producer captures and two independent replays directly against the maintained
paths. All six owned children exited successfully, were reaped and had their
process groups drained. All five generated data files match each other and the
current fixture byte for byte. Nine source/fixture and six model identities
match before and after execution; all 50 output/log/resource records are pinned.
The historical artifact-source execution is retained separately and is not
substituted for this fresh run.

The completed directly timed Python children peaked at 32,849,920 bytes RSS,
below the 128 MiB experiment ceiling. Physical footprint is recorded separately;
the largest observed value was 21,938,800 bytes. Internal `ru_maxrss` fields in
the producer are taken before publication/return and are not treated as completed
child-lifetime measurements. Runtime evidence pins the interpreter, timing
executable and reviewed scripts; it does not cover the full standard-library or
native-image closure. These are offline correctness observations, not inference
latency or device-memory measurements.

The rejected first runner and failed preliminary newline check remain in the
artifact history without acceptance credit. The corrected runner preserves
attempts before launch, handles interruption, drains groups, bounds first log/
output captures and uses a controlled environment. Astra reviewed the code and
the fresh execution evidence before publication.

## Next boundary

A native loader must hash every raw chunk it actually converts or uploads,
including repeated tied-weight reads, and complete a checked range ledger
before publishing a private immutable identity. Metadata hashes must refer to
the same buffers that were parsed. Path names, matching geometry and reopening
files after construction cannot manufacture that identity. The admitted loaded
weights may retain their identity after the path changes.

That loader, trusted profile registry, tokenizer pairing, native capability/
response producer and app integration remain separate code tasks. This witness
adds no Gemma4, Metal, 131072-output, quality or phone grant. Earlier numerical
failures and all open [Gemma4 gates](gemma4-implementation-plan.md) remain open.
