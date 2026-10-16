# M6 complete long numerical quality

Both selected representations pass the unchanged paired long-context loss gate
at 4K, 8K, 16K and 32K: 128 scored targets per depth and 512 per model. Each depth
and each pooled model mean stays below the frozen +0.05 NLL increase limit.
The only two changed winners are the separately reviewed exact larger4K cases.
This closes long numerical quality; ordinary context boundary coverage and the
final KV/M6 decisions remain open.

## Frozen scope and results

The input is the frozen cyclic held-out stress sequence, with 128 reserved
continuation tokens at each total depth. It tests numerical behavior and state;
it does not establish natural long-document ability. The accepted small model
uses compensated FP16 weights. The larger model uses the frozen group8 Q4 v2
compensated tiled configuration. Both retain FP16 KV and use chunk 128 with
FullLogits, shared immutable weights and separate paired session state on the
Apple M1 Pro, MacBookPro18,3 with 16 GiB, macOS 26.6.2 and Rust 1.90.0.

| Total tokens | Small mean NLL increase | Larger mean NLL increase | Changed winners, small / larger |
| ---: | ---: | ---: | ---: |
| 4,096 | -0.00024724887088295867 | -0.0003384621888100002 | 0 / 2 |
| 8,192 | +0.00018373894511906317 | -0.00016554190014048012 | 0 / 0 |
| 16,384 | -0.0000122889085267407 | +0.0000002920944404943501 | 0 / 0 |
| 32,768 | +0.0000017188451345992332 | -0.0000014703308812434907 | 0 / 0 |
| Pooled 512 targets | -0.00001851999728900924 | -0.0001262955813478074 | 0 / 2 |

The successful larger32K v2 uses the separately qualified producer-typed
collector and the unchanged original numerical worker. It emits all 128 scored
rows, exits naturally with code 0, and has no cleanup or endpoint errors.
Admission is 4,442,652,308 bytes and maximum sampled joint native lifetime
charge is 4,297,527,400 bytes, below 6 GiB. Its source/input identities remain
unchanged. The earlier failed v1 remains failed and preserved; its emitted rows
never substitute for a clean lifecycle.

The [exact winner investigation](2026-10-16-m6-winner-investigation.md) reproduces
both complete original distribution pairs and accepts only the original record,
model/depth/index, token-prefix and distribution SHA keys for targets 90 and 127.
Independent ordering, candidate bits, margins, error and losses explain two
unique top-two reversals. No general near-tie rule, internal attention cause or
loss/task-gate waiver is introduced. Original automatic audits remain false
because they deliberately require separate manual review of any changed winner.
This accepted manual review is recorded separately from those immutable results.

## Replay and acceptance boundary

The exact mixed-format replay covers eight collections, 1,048 raw events and
334 bindings. Every original collection uses its original replay route; only
the fresh typed larger32K run uses the explicit typed format. All eight workers
complete naturally, with exact ordered target coverage, state and memory checks.
No tolerance, numerical kernel, representation, request or quality gate changes.
Astra independently reproduces the replay byte for byte, verifies all bindings,
and recomputes both pooled means and every depth's loss gate.

The [qualification receipt](../../benchmarks/results/2026-10-16-m6-full-long-quality-qualification-v1.json)
binds five current source files, five exact published copies and six evidence
pins, including the already accepted exact-case review. Astra accepted the final publication after verifying every source/evidence
pin, exact published copy and reported value, with no remaining findings. The ordinary 32K timing
attempt interrupted later contributes nothing to these quality results and
remains an incomplete, unqualified attempt.

This result qualifies long numerical quality only. All sixteen ordinary and
independent boundary runs, full context synthesis and the final FP16 KV/M6
decision must complete separately. Existing full 10,000-target loss, 100-task,
short performance and maximum-context application qualifications remain as
previously recorded. Defaults remain unchanged, and M7 is outside this scope.
