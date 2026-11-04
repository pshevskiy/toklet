# Independent real-E2B512 reference

The pinned original four-bit E2B source now has a bounded independent512-token
reference before native execution. The [frozen contract](../benchmarks/gemma-e2b-context-v1/manifest.json)
uses the first512 IDs from the authenticated quality corpus. Both mathematical
FP64 and staged-FP32 graphs retain the historical operator order, Decimal80
rotary and ordered `math.fsum` reductions. The old32-token and tiny long oracles
remain unchanged. This reconstructs the original packed artifact into FP32
weights; it is not the unquantized BF16 quality baseline.

Gathers and projections split into at most32 rows. The complete causal graph
runs over all inputs; selected final hidden rows pass through the tied head and
softcap. The512 phase retains full-vocabulary rows at absolute positions0,31,255
and511. It avoids a512×262144 head allocation of512MiB/1GiB per FP32/FP64 array.
The reference uses no incremental KV session or Rust numerical implementation.

An ordered-product helper forms the same individually rounded FP64 products in
NumPy, then passes them in order to `math.fsum`. Two microprobes over42 real/
adversarial vector pairs match bit-for-bit and measure3.41/3.43× for that operation.
Four cold controls additionally cover dtype/subnormal/cancellation cases, all
intermediate tiny states through64 positions, selected-head equivalence, public
profile/position bounds and mutated input files. These are reference-tool
measurements, without an engine-performance claim.

| Phase | Inputs / selected rows | Elapsed | Peak RSS |
| --- | --- | ---: | ---: |
| Real compatibility |32 /0,15,30,31 |46.83s |302,809,088 bytes |
| Timed pilot |64 /0,31,63 |97.10s |342,507,520 bytes |
| Full reference |512 /0,31,255,511 |2,070.99s |860,078,080 bytes |

Compatibility rows are byte-identical to the authenticated prior32 reference in
both contracts. Positions0 and31 also remain byte-identical between32 and512,
checking causal-prefix preservation. Full512 mathematical/staged passes take
1,080.37/986.30s. Collection overlaps independent Mac work; these timings are
descriptive. The1,072MiB reference allowance and6GiB process policy both exceed
the observed peak. Only12MiB of512-phase rows are written.

Each owned process exits cleanly. Inputs, source/config/index, numerical/runtime
files and output identities are pinned; endpoint and disk-reserve checks pass.
The initial cold fixture used IDs outside its17-token vocabulary; correction and
failed log are retained. Rust numerical code and dependencies are unchanged by
this reference task. Combined Astra code/results review passes.

This supplies the independent oracle for the native512 boundary task. It does
not establish native agreement at positions32–511, waive earlier four-token
near-zero failures, close full unquantized quality, enable Gemma Chat, qualify
phone execution or establish throughput. Native capture and separate numerical
verdicts remain next.
[Evidence](../benchmarks/results/2026-11-04-gemma-e2b-context-reference-v1.json).
