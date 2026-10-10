# Bounded held-out loss runner, 10 October 2026

The standalone [Rust example](../../examples/quality_bench.rs) and
[Python adapter](../../tools/bench/quality.py) prepare the frozen 10,000-target
loss evaluation. Numerical kernel and session implementations are unchanged.
Source review is clear after correcting interrupted-check process cleanup.
Formatting, tests and native short-probe qualification pass. Astra accepted
the completed code and evidence without findings.

The existing [contract](../../benchmarks/contract-v1/manifest.json) already fixes
independent blocks of at most 1,024 targets with 1,025 token IDs and one-token
overlap. Each block resets KV and rotary positions to zero. The first ID seeds
the distribution; label `i+1` is scored from the committed prefix ending at `i`.
Every target is scored once. After scoring a label, the worker commits it before
the next step; complete blocks finish with no pending token. Ten blocks cover
all 10,000 targets; the final block has 784 targets and 785 committed tokens.
No sampled token, tokenizer, chat formatter or task execution enters this path.

The worker holds one immutable FP32 model, a capacity-1,025 session with chunk
one and one reusable little-endian FP32 logit serialization buffer. No full
prefix trace or all-position vocabulary matrix is retained. Tokens are bounded
to 2–10,001 UInt32 IDs before model loading, checked against vocabulary before
evaluation, and bound by the worker's corpus hash. Diagnostic storage is admitted
within a separate 64 MiB application allowance. Weights/KV/session scratch,
metadata, history and both session logit buffers retain explicit memory counts.

Loss uses max-subtracted FP64 exponentials and compensated summation over the
committed FP32 logits. `(maximum-target)+log(sum)` preserves equal-logit loss
even at the largest finite FP32 value. Empty/nonfinite distributions, invalid
labels and invalid reductions fail. Each target record includes global index,
block/local position, committed prefix length, label, loss, target-logit bits
and SHA-256 of the complete vocabulary logits. FP64 sums aggregate blocks and
the full corpus; these diagnostic calculations are outside inference kernels.

The adapter requires accepted CPU numerical evidence before model reads, checks
actual scalar/tiled admission, validates every label/position/block commit,
reconstructs aggregates and admits own-child physical peak under 6 GiB. Every
exact stdout line and stderr are retained, with source/build/binary, frozen
corpus/model and helper hashes. The completed child waits for its owner's exit
handshake while footprint is queried; bounded line reads and owned-process-group
watchdogs retain partial failures and reap children. Fresh evidence/build paths
protect existing artifacts; frozen root release binaries are not rebuilt.
Build/check interruptions kill and reap their owned process group, retain
stdout/stderr and exit/failure metadata, and re-raise the original exception.

`--probe` evaluates both scalar and tiled on the two frozen zero-offset Qwen
four-token cases. Complete independent Transformers FP32 logit arrays anchor
the six target losses. The existing short numerical NLL limit of 0.001 nats per
target remains unchanged, and scalar/tiled complete logit hashes and loss records
must also agree exactly. These probes establish scorer wiring, not held-out
quality. Without `--probe`, the exact contract's 10,001 IDs score all 10,000
targets. The corpus and numerical gates are never adjusted to accept a result.

The later lossy-quality decision compares mean NLL against the unquantized
reference under the frozen 0.05 nats/token increase limit and separately applies
the 100-task, two-percentage-point pass-rate gate. This runner produces loss
records; it does not yet execute or grade those tasks, compare a lossy model,
qualify performance, or close M2.5/M3. Logit hashing, output and handshakes are
diagnostic; no warmup/repetition latency acceptance is implied.

The [completed probe record](../../benchmarks/results/2026-10-10-held-out-loss-probe-v1.json)
and [exact native lines](../../benchmarks/results/2026-10-10-held-out-loss-probe-v1.json.native.jsonl)
retain four cleanly exited workers, twelve scores over six independent labels,
and all prefix/block/commit evidence. Scalar and tiled complete-logit hashes and
loss records agree exactly in both cases. Maximum independent source NLL
discrepancy is 0.000217481131459 nats, below the unchanged 0.001 short-probe
limit. This is short scorer qualification; no full held-out score is reported.

Every worker commits four tokens with none pending. Own-child physical peaks
are 2,009,417,720 / 2,009,319,416 / 2,009,319,416 / 2,009,368,568 bytes for
scalar-future, scalar-short, tiled-future and tiled-short. Admission is
2,136,856,460 bytes: weights 1,976,131,072; KV 25,190,400; scratch 89,608;
history 4,100; session logits 1,215,488; metadata 8,064; and separate
67,108,864-byte metadata and application allowances. The reusable diagnostic
serialization buffer is 607,744 bytes within the application allowance. No
redundant full model or all-position vocabulary matrix remains resident.

Root formatting, Clippy with warnings denied, all 76 ordinary Rust tests,
three explicit example tests, isolated locked/offline release compilation and
all 35 offline benchmark Python tests pass. Checks and all model work occur
after the medium scalar benchmark exits naturally, while its orchestration
parent is held. No timed CPU workload overlaps qualification.

The isolated binary SHA-256 is
`18c2b95fc6eab4622c9a088f522fd04e681cf148ec648d50bd72f116d0bc591a`.
The completed record SHA-256 is
`6fb800232abdb0b2e3e008b4f61002390bdcdfeb9168cb51ea0ea232e6ad6901`;
exact native output SHA-256 is
`7f62449bf6883bd539b4c722bdab224aede09e8e289dfee7770d762e1708c626`.
The [empty stderr](../../benchmarks/results/2026-10-10-held-out-loss-probe-v1.json.stderr.txt)
is retained. Source, helper, model/contract/fixture, corpus and binary identities
are revalidated at completion. Build/check stdout/stderr remain in the record.

```sh
.venv/bin/python -m unittest discover -s tools/bench -p 'test_*.py'
.venv/bin/python tools/bench/quality.py --probe --cpu-qualification benchmarks/results/2026-10-10-cpu-tiled-qualification-v1.json --output benchmarks/raw/loss-probe-fresh.json --build-dir artifacts/loss-probe-fresh
```
