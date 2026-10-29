# Gemma3 real Mac native decoder

29 October2026. Status: **selected bounded Mac native conformance qualified**
after fresh collection/checking and independent Astra review. This report records
the selected bounded engine task, not ordinary
app admission. Follow the [native conformance policy](gemma3-native-conformance.md)
and [independent checker](gemma3-conformance-checker.md).

## Selected model and implementation

The artifact is `mlx-community/gemma-3-270m-it-4bit` revision
`ff1143e3a10547c9f2129e94ca37059b096b23f4`:18 layers, hidden640, FF2048,
4 query heads/1 KV head/head256, Q4/group64/BF16 affine weights and tied
vocabulary262144. Fifteen layers use local window512; layers5/11/17 use global
attention. Original raw BF16 norms receive the explicitly qualified FP32 unit
offset. Both Half residuals use the source-supported FP32 add/clip/store profile.

The [original-byte loader](gemma3-original-loader.md), shared projection upload,
FP32-scale RMS, Half pointwise/rotary, staged attention and transactional
Model/Session were implemented in earlier reviewed tasks. This task adds the
bounded public-API capture example and verifies their real composition. It
introduces no numerical kernel or reference change.

Declared32768, implemented capacity1–544 and this tested capacity544 are distinct.
Absolute tested positions end at543. Small prompts and intermediate prefixes do
not qualify every capacity/input/configuration combination. Ordinary app limits
and eligibility remain unchanged.

## Fixed workload and checks

- Two plain-text inputs of6/11 tokens, whole and tokenwise, with selected rows
  0/5 and0/10. No Chat template is rendered.
- A544-token input with partitions17×32 and16×31+[15,1,1,1,30], selected rows
  511/512/513/543. All544 logical logit rows agree exactly across partitions.
- A changed final token preserves all543 earlier logit rows. Ordinary last-row
  output agrees with the diagnostic path at every32-token chunk end.
- Thirty-two fixed teacher distributions and64 native-free distributions.
  Greedy selection, codec/stop/state accounting and repeated free generation
  are checked; free outputs are not independently graded for content quality.
- Empty input, excluded image ID262144, oversized chunk33 and capacity545
  reject without corrupting the published prefix. Reset/replay, tied ownership,
  CPU payload drop, actual exact-budget/budget-minus-one and memory ledgers pass
  structural checks. Full physical real-model KV fault injection is not claimed;
  independent tiny fault/state evidence remains separately scoped.

The finite native process makes187 forward calls, commits2,351 tokens and saves
6,496 records/141,258,016 bytes. It never reads the source numerical arrays.
All6,400 selected stage records,32 teacher rows and64 free distributions are
retained before the independent checker runs.

The source and all3,232 original coefficient dictionaries are frozen. The new
policy checks624 coupled attention/output groups and5,808 actual-input groups.
All18,537,288 original coupled comparisons remain visible separately; the old
92- and84-value rejection screens stay rejected. A native conformance decision
does not relabel them as CPU-trajectory fidelity passes.

## Hardware and observed resources

Apple M1 Pro Mac,16GiB RAM, macOS26.6.2. The frozen optimized native example is
5,360,352 bytes, SHA256
`a49c5c34d09f5c57b1cd0d41aec3ad14632916691d910d9aea680357dc3a982b`,
release opt-level3 with Metal and debug assertions disabled. All237 current
Rust/Metal/Cargo source files match its passing cold checks, including strict
Clippy, formatting and four example controls in debug/release. Prior default431
and Metal-library327 test results remain scoped, overlapping checks.

| Process | Completed seconds | Maximum observed RSS bytes | Lifetime physical counter bytes |
| --- | ---: | ---: | ---: |
| Fresh native capture | 73.0274496 | 976,453,632 | 815,334,720 |
| Fresh independent reader | 261.0726587 | 823,181,312 | 371,427,632 |

The native owner retains688 direct process observations, enforces6GiB/900s and
closes its owned PID/group. This diagnostic time includes capture/readback and
does not measure generation throughput. Process counters do not establish a
continuous/process-tree peak or whole-app/mobile fit.

Astra independently authenticates all23,577 native endpoints/outputs, all6,496
raw record hashes,688 memory observations and process closure. The entire raw
capture is byte-identical to the historical capture. All15,003 journal events
agree after excluding only the independently checked changed start metadata
and187 forward timing fields; the summary is byte-identical. The separate fresh
reader passes all624 coupled groups/13,185,444 values and5,808 own-input
groups/5,351,844 values with zero fatal failures. It completes2,445 direct process
observations and closes its owned group. Original coupled MAIN84/WIDE98
diagnostics remain unchanged and rejected; fresh eligibility becomes true only
after all endpoint checks. Raw component reports keep qualification false until
the independent outer decision. Astra independently replays all13,185,444
coupled gates with zero failures and authenticates all5,351,844 local results,
weight ranges, masks, process samples and closed endpoints. Eight exact fallback
branches are aggregate reports supported by reviewed code/controls; individual
fallback certificates were not retained/replayed independently.

The [machine-readable release decision](../benchmarks/results/2026-10-29-gemma3-real-native-v1.json)
records the bounded qualification, exact evidence and all excluded claims. It
supplies the outer decision; raw capture/checker reports retain their original
component qualification flags. Historical trajectory rejections are unchanged.

## Remaining scope

Full original-unquantized10,000-target/100-task adoption quality, production
Chat/Hugging Face admission, context controls/whole-app memory, declared full
window and independent iPhone qualification remain open. There is no whole
Gemma-family, arbitrary-model or performance claim. After this bounded B3 task,
the next mechanism is the [LFM2 short-convolution block](lfm2-hybrid-plan.md),
then B5 user controls/integration.
