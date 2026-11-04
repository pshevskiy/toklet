# Real Gemma4 hybrid512 boundary

The explicitly requested diagnostic512 model passes a real512-position state/
schedule boundary and selected-row comparison against the
[independent oracle](gemma-e2b-context-reference.md). Default E2B construction
remains32 positions; the [diagnostic constructor](gemma-context-probe.md) and
numerical implementation are unchanged by this test task. Gemma remains outside
the ordinary app catalog.

The [native contract](../benchmarks/gemma-hybrid512-v1/manifest.json) binds the
same512 corpus IDs and oracle manifest before execution. Schedules16×32 and
1+16×31+15 fill the cache at position zero. All512 tokens, all15 physical K/V
owners, absolute slot mappings and the final logit row agree exactly. Epochs are
checked per commit before exclusion from cross-schedule state comparison.
The append at absolute position512 (token513) returns the exact capacity error;
full snapshot, pointers, allocations and published logits stay intact.

After each full context, reset invalidates every slot and clears history/logits.
A fresh32-token prefix reproduces every logit row and valid K/V prefix; retired
physical cache contents remain invalid. The initial32 rows, eight main selected
rows and two full32-row reset blocks form11 records/104MiB. The cold checker
rejects coherent reset changes, wrong row inventories, failed wrappers and
truncation. A separate numerical-rejection fixture retains a passing state
verdict while reporting both oracle failures.

| Independent contract | Context/position rows | Values compared | Failed | Maximum gate ratio |
| --- | ---: | ---: | ---: | ---: |
| Mathematical FP64 |8 |2,097,152 |0 |0.2460932893 |
| Staged FP32 |8 |2,097,152 |0 |0.3465543086 |

Four main positions0,31,255,511 are compared at512 inputs; prefix positions
0,15,30,31 are compared at32. Duplicate schedule rows and full reset rows must
agree byte-for-byte and are not counted again as independent numerical samples.
Both native and reference causal-prefix rows0/31 agree exactly across context
lengths. The unchanged gate is`abs(delta) <= 1e-5 + 1e-4*abs(reference)`.

Session allocation stays263,761,128 bytes. Native peak RSS is3,063,037,952 bytes,
physical footprint3,056,438,056 bytes, below6GiB. Source/config/index, frozen
inputs, reference rows/runtime, release executable and parser dependencies are
bound before/after execution. Both owned native and replay processes exit
cleanly. Two cold checker controls,550 default/658 Metal-feature tests,
formatting, strict Clippy in both configurations and release build pass.
Combined Astra review is recorded in the evidence.

This covers the512 state boundary and the selected numerical rows, not every
intermediate or input. The local512 ring is full but has not evicted: wrap checks
need a larger candidate. Earlier four-token near-zero mathematical failures,
unquantized full quality, resident GPU scheduling, ordinary Gemma Chat, phone
qualification and performance remain open. The raw test does not qualify longer
free generation or the declared131,072-position window.
[Evidence](../benchmarks/results/2026-11-04-gemma-hybrid512-v1.json).
