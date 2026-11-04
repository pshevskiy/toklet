# Gemma4 real32-position hybrid boundary

The pinned original four-bit Gemma4 E2B hybrid session passes its implemented
32-position state boundary on Apple M1 Pro. This is the existing packed Metal
projection/gather plus CPU shared graph, with FP32 activations/KV and the existing
reductions. Numerical kernels, app eligibility and the native40 package do not
change. The [frozen contract](../benchmarks/gemma-hybrid32-v1/manifest.json) selects
the first32 IDs from the authenticated quality corpus before either execution.

The ignored sibling test shares the old bounded loader and runs schedules
`[32]`, `[31,1]` and `[1,15,16]` at position zero. All32 committed tokens, all15
physical KV owners' complete contents, absolute slot mappings and the final
logit row match exactly. Cache generations are checked as commit/reset epochs
before being excluded from numerical-prefix comparisons. Each rejected append
at absolute position32 (the33rd token) returns the exact capacity error and
preserves the complete snapshot, storage pointers and allocated byte count.
Reset empties validity/history/logits, then a full32-token replay reproduces the
same prefix. Immutable resident owners and source descriptor/path identity stay
unchanged. Each raw session allocates245,081,448 bytes.

Independent reference execution precedes native capture. The same authenticated
four-bit source is reconstructed into FP32 weights and evaluated by the existing
mathematical FP64 and staged-FP32 graphs, including their independent rotary
calculation. This is not the original unquantized BF16 quality baseline. Only
full-vocabulary rows at absolute positions0,15,30,31 are retained, while the
reference computes all32 inputs. All24 native rows (four positions × three
schedules × before/after reset) match byte-for-byte at each selected position.

| Numerical contract | Compared values, including repeats | Failed | Maximum gate ratio |
| --- | ---: | ---: | ---: |
| Mathematical FP64 | 6,291,456 | 0 | 0.0391258774 |
| Staged FP32 | 6,291,456 | 0 | 0.0013620592 |

There are1,048,576 unique selected values per contract. The unchanged gate is
`abs(native-reference) <= 1e-5 + 1e-4*abs(reference)`. Raw rows, counts, coordinates
and maxima remain replayable. State and numerical verdicts are separate; a
successful checker exit alone does not mean numerical acceptance.

Reference peak RSS is410,910,720 bytes. Native peak RSS is3,005,054,976 bytes and
peak physical footprint2,998,668,048 bytes, below the6GiB process policy. Source,
config/index, input, runtime and executable identities are bound before/after
execution. Disk reserve includes outputs. An initial cold-test syntax error,
a completed reference with denied BSDtime measurement, and an interrupted retry
remain preserved. The fully measured repeat's row hash equals the first completed
reference. The final runs exit cleanly; no native result was discarded.

Three cold Python controls,549 default/657 Metal-feature tests, formatting,
strict Clippy in both configurations and the release harness build pass.
Final combined Astra code/results review passes.

This closes this real32-position state/schedule boundary and selected-row
comparison. It does not resolve the earlier27 four-token mathematical failures
on the earlier four-token input, qualify every32-position intermediate, enlarge context,
provide full unquantized quality, enable ordinary Gemma Chat, qualify iPhone or
establish performance. Next work is a separately admitted512-position candidate
and its state/numerical/memory checks, with those earlier failures retained.
[Evidence](../benchmarks/results/2026-11-04-gemma-hybrid32-v1.json).
