# Bounded CPU row-loss reference scorer

The new `quality_rows` worker retains one immutable FP32 model and one CPU
session, with capacity 1025 and chunks 1 or 8. Projection uses the existing
ordered FP64 tiled math with FP32 weights/storage. Caller-owned output is
`[chunk, vocabulary]`; output plus serialization fit inside the separately
admitted 64 MiB application allowance. The old independent CPU worker and its
protected collection helpers are unchanged.

Scoring preserves the frozen independent-block policy, one-token overlap,
logical causal labels, final-target commitment, stable FP64 logsumexp and Kahan
aggregation. The driver verifies each target, chunk, block, completion and
parsed-input hash. Idle handshakes allow libproc v4 accounting for the owned
worker and driver; failure kills and explicitly reaps the owned process group.
Transport is bounded to 300 seconds per event, six hours per worker, 32768
events and 8 MiB stdout. Initial and endpoint hashes bind source, config,
tokenizer, code, executable, corpus, helpers and probe manifests.

The [small-model probes](../../benchmarks/results/2026-10-13-cpu-row-loss-small-probes-v1.json)
complete four clean workers. Chunk-1/8 losses agree exactly; maximum difference
from the independently frozen FP32 probe is 0.000217481 nats, within the unchanged
0.001 guard. Highest combined native lifetime charge is 2,051,983,952 bytes.

The [larger probes](../../benchmarks/results/2026-10-13-cpu-row-loss-larger-probes-v1.json)
also complete four clean workers with exact batch/single loss agreement. Chunk-8
admission is 6,370,269,388 bytes including application allowance; its external
logits buffer is 4,861,952 bytes, included in that allowance. Highest combined
native lifetime charge is 6,288,004,880 bytes, below 6,442,450,944. These are short
load/admission/batching checks, not full-context physical qualification or an
independent numerical oracle for the new geometry. Native charge can omit clean
or shared pages and does not replace allocation admission.

The [source receipt](../../benchmarks/results/2026-10-13-larger-source-verification-v1.json)
records exact official SHA256/size for the downloaded checkpoint and exact Git
blob/size for config, tokenizer and license assets. Header/config match the prior
preselection. Tokenizer JSON, vocabulary, merges and chat configuration are
byte-identical to the pinned 0.5B assets; the frozen token-ID semantics are kept.
The first tiny-asset download was rejected by curl's size bound on a redirect;
a bounded 10 MiB transfer was followed by exact final size/blob verification.

The [task binding](../../benchmarks/results/2026-10-13-cpu-row-loss-qualification-v1.json)
retains formatting, strict Clippy in both configurations, 85/91 ordinary Rust
tests, three scorer math checks, 153 Python experiment checks and native receipts.
Astra accepted code/check/native evidence. This accepts the scorer mechanism only. Independent larger geometry/short
numerical qualification precedes accepting a full 10000-target FP32 reference.
GPU quality, 100-task grading, performance, sustained runs and M5 exit stay open.
