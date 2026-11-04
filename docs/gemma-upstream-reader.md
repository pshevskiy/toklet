# Official BF16 upstream reader admission

The pinned upstream state maps 541 persistent keys onto all540 selected BF16
source tensors, with the tied head sharing the embedding extent. All35 persistent
layer scalars and60 unused shared-consumer K/V tensors are checked explicitly.
Raw BF16 reads preserve signed zero and subnormals; embedding/PLE tables can only
be gathered, and the head reads at most128 vocabulary rows in original order.

A prospective logical request sequence covers sorted root weights, unique token
rows, each stock layer state and every head tile. The actual journal must match
this sequence; an independent layout traversal reconstructs byte spans/read-chain
hashes. Source authentication and descriptor ownership remain in `bf16_source`.

Five cold controls pass: complete inventory, wrong-shape/extra-consumer/tied-alias
rejection, BF16 bits, duplicate/out-of-order gathers, scalar and multi-tile reads,
final vocabulary rows, invalid requests without reads, short-read poisoning,
prospective-vs-actual tiny request order, and prior resident/streamed equivalence
and retained-exception recovery. No Rust inputs changed; prior required Rust
checks remain applicable. The initial metadata-only unset-head-count error was
corrected by using the stock attention's resolved head grouping.

The four-token admission ledger is777,801,218bytes, including512MiB runtime,
27,528,704root bytes,140,001,282largest-layer bytes,6MiB read/decode tiles and64MiB
captures/attention. This is an estimate; the real run must separately measure
RSS and physical footprint below6GiB. No real checkpoint inference or new
numerical/quality qualification follows from reader admission.
[Evidence](../benchmarks/results/2026-11-04-gemma-upstream-reader-v1.json).
