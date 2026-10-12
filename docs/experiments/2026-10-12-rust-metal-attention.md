# Tiny causal Metal GQA, 12 October 2026

All ten saved-input tiny attention jobs pass their original scores,
probabilities and context gates. Every score matches its frozen FP32 bits;
one context position differs by one normal Half ULP after small GPU softmax
rounding differences. No gate changes. Astra accepted the completed code and evidence without findings after byte-identical independent replay.

The [shader](../../tools/experiments/rust_metal_attention.metal) consumes
Q `[tokens,4,8]` and separate head-major K/V `[2,32,8]`. Query heads0/1 share
KV head0; query heads2/3 share KV head1. Each lane owns one token/query-head
score/probability row and context eight-tuple. It computes ordered FP32 dot
products divided by FP32 `sqrt(8)`, stores all diagnostic scores, subtracts
the maximum valid score, sums exponentials in increasing position order,
normalizes and reduces probabilities times Half V in FP32. Future probability
bits are exactly positive zero; only positions0 through query-token index
contribute to Half context. Products/sums are uncontracted. No K/V head
expansion, inter-lane barrier or intermediate host wait.

The [host](../../tools/experiments/rust_metal_attention.rs) admits one case per
child process, rows1–32, query heads4/KV heads2/dimension8/capacity32. JSON is
bounded to one MiB; unsigned topology and exact finite Half input counts are
checked before native resources. Queried device/pipeline widths control the
rounded query grid. Q/context cover rounded query count times8 plus seven
guards; scores/probabilities cover rounded query count times rows plus seven.
K/V each have full capacity plus seven guards. Six distinct tracked buffers
remain initialized, aligned and owned through completion; copied row count
has four-byte size/alignment. Every lane writes disjoint regions. CPU accesses
occur before submission or after checked completion.

The [driver](../../tools/experiments/rust_metal_attention.py) covers both
layers of every frozen case with saved Q rotary and K/V cache inputs. Cache
placement is local offset0, including the absolute rotary-offset5 fixture.
This is full-prefix attention of those local tokens; it does not qualify a
nonzero committed cached prefix or propagated projections/rotary/cache writes.
Score/probability matrices are materialized for diagnostics. Tiled online
softmax, decode/prefill tuning and longer contexts remain later tasks.

| First intermediate frame | Values | Equal frozen bits | Maximum original source gate fraction | Maximum source absolute error |
| --- | ---: | ---: | ---: | ---: |
| FP32 scores | 8,528 | 8,528 | 0.495719949277 | 0.002866387367 |
| FP32 probabilities | 8,528 | 7,398 | 0.474988628550 | 0.000481843948 |
| Half context | 2,816 | 2,815 | 0.494596089993 | 0.001479625702 |

The [native record](../../benchmarks/results/2026-10-12-rust-metal-attention-v1.json)
contains130 attention commands and36,608 final context values across three
warmups and ten samples per case. First complete frames and2,354 guards are
retained and independently replayed. The host requires all twelve subsequent
complete score/probability/context frames, including guards, to equal the
first; raw later nonfinal frames are not archived. Every final context is
retained and independently compared. Immutable-input guards are checked
natively; complete input readbacks are not archived.

Independent ordered FP32 reconstruction matches all8,528 score bits.
The GPU softmax's1,130 differing first-frame FP32 probability positions differ
from the frozen Half-storage reconstruction by at most three FP32 ULPs and
0.00000005960464477539063 absolute. All original source gates pass. These
observations do not establish general transcendental bit identity.

The single context discrepancy is capacity/layer0, flat index605 (token18,
query head3,feature5): GPU `0x1cab`, frozen `0x1caa`. Both are normal Half,
with spacing `2^-18` (0.000003814697265625). Independent ordered FP32 reduction
using recorded GPU probabilities gives0.004556663334369659; using frozen
probabilities gives0.004556652158498764. The Half midpoint
0.0045566558837890625 lies between them. FP64 reduction with GPU probabilities
is0.004556665264317417. This isolates the observed Half boundary to the small
probability differences. Every first context equals independent ordered FP32
probability×V reduction followed by Half RNE; all repetitions retain the same
final contexts. Changing the future token preserves the first three context
tokens exactly in each layer, and all masked probability bits remain zero.

Resources/pipeline are resident across repetitions, with one command commit/
wait per job step. Same-binary cache-layout, rotary and composed MLP checks
plus required failures total **754 positive commands and four negative jobs**.
Fourteen build checks include formatting, Clippy with warnings denied,
31 standalone Rust tests and locked offline arm64 release build. Binary SHA-256:
`e6da9a2a2da9e858f3ba37c5f54bfbcf54fb2e7cd76c76e401f365c6e0d6bbea`.

Maximum attention admission is **67,225,936 bytes**: shared allocations once,
conservative full host input baselines/readbacks and first/current output
frames separately, plus64MiB for bounded fixtures/runtime/JSON. Peak own-child
physical charge across all jobs is **22,594,160 bytes**. Output guard fills
occur outside command timing. No allocation-free inference, full-model memory
or performance claim is made.

Hardware remains M1 Pro / MacBookPro18,3 /16GiB / macOS26.6.2(25G83), Rust1.90.0
and SDK26.5; minimum macOS15, shader language2.3, safe math/precise functions.
Start/end observations retain AC power, configured AC low-power mode off and
no thermal/performance warning. Continuous temperature/clocks, energy, total
system/device memory and GPU traces remain unavailable.

The [idle proof](../../benchmarks/results/2026-10-12-rust-metal-attention-idle-v1.json)
and [archive producer](../../benchmarks/results/2026-10-12-rust-metal-attention-archive-v1.py.txt)
retain natural completion, prior numerical-chain absence, stopped VM and
protected endpoint identities unchanged. All native children are reaped before
offline checks/hashing. The [analysis](../../benchmarks/results/2026-10-12-rust-metal-attention-analysis-v1.json)
and [exact producer](../../benchmarks/results/2026-10-12-rust-metal-attention-analysis-v1.py.txt)
replay every validator, original source gate, first-frame guard, each final
context, independent ordered score/context reconstruction, causal masks,
layouts/admission and source/copied-source/input/driver/manifests/binary hashes.
Three targeted and full offline suites pass:94 experiment Python tests,
193 benchmark Python tests, root formatting/Clippy and76 ordinary Rust tests
in an isolated debug target. Real-model checks were not rerun; root release
binaries remain unchanged.

Retain bounded full-prefix GQA. Cached-prefix attention, complete GPU layer/
decoder propagation, session commitment/cancellation, Qwen/backend quality,
scheduling and M4 completion remain open.
