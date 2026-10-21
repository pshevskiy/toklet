# M7 bounded raw task-response coordinator

A cold Swift coordinator now assembles one raw task response across bounded native
commands. This prepares the frozen100-task phone collector. It does not collect
or grade those physical responses in this component.

The caller owns the admitted worker, immutable prompt, capacity and persistent
cancellation/foreground latch. Preflight accepts1..306 prompt IDs,1..256 sampled
outputs,1..2 distinct valid stop IDs, a positive vocabulary and enough capacity
for the whole prompt plus maximum, before reset or other mutation. Each command
samples at most128 IDs; commandLimit is explicit for shorter diagnostic chunks.
Visible response ownership stays bounded to256 IDs. Numerical kernels and the
worker's128-token callback buffers are unchanged; tokenization/formatting remain
outside the coordinator.

Each case resets, uses operation0 for the initial prompt and operation2 with no
input for continuation. Successful output-limit completion must emit the exact
requested count and have no hidden-stop metadata. A stop ends the case, is absent
from visible IDs, belongs to the explicit stop set and consumes one sampled slot.
Every chunk checks live idle owners, successful status, no score, exact callback
coverage/order, valid visible IDs, command-local monotonic native elapsed times,
globally monotonic callback uptime and history/evaluated/pending/saved counts.
For prompt lengthp and sampled totals, expected counts arep+s,p+s-1,1,p+s.
A chunk becomes output only after these checks; failures yield no completed
response. Command-local callback indices restart at zero; chunk.index preserves
the global chunk order and the response concatenates visible IDs without gaps.

The required checkActive closure must enforce a persistent abort latch before
and after every command and after the output sink. This prevents a cancellation
between chunks from being cleared by NativeWorker.submit's per-command flag.
Cancelled reason3 is rejected even after the requested final callback. The caller
must close/reset after failure rather than count a partial response as complete.

NativeSnapshot exposes one optional-in-meaning stopToken scalar, valid only for
successful idle reason2 completion. Accepted commands, interruption, close,
failed completion, queued cleanup and non-stop results clear it toUInt32.max.
Astra found that successful suspend/unload resets the C report to zeros; merely
clearing before submission would republish token0. Reason2 is now part of the
publication guard. Direct native checks cover a failed score after hidden EOS,
suspend, replay, unload and close so those paths cannot expose stale stop IDs.

Eleven simulated cases per backend cover256 samples via128+128 continuation,
hidden EOS at samples1/128/129/256, cancellation between chunks and at the final
callback, failed continuation, a failing second-chunk sink, invalid callbacks/
IDs/state and whole-request preflight including Int.max. These establish bounded
protocol behavior rather than256-token real-model correctness. Actual tiny CPU
and Metal each generate eight IDs once and again through3+3+2 commands, requiring
identical full IDs and final state. All IDs are47; the first greedy ID is derived
independently from the frozen tiny-v1 prompt FP32 logits. Each backend also hides
that first ID as EOS and requires one sampled/pending hidden token, then performs
the lifecycle clearing checks and orderly weak-owner release.

The [qualification receipt](../../benchmarks/results/2026-10-21-m7-task-response-qualification-v1.json)
binds210 input/output paths, five changed code files, the exact parent worker
source and82 published raw copies. Ten owned task commands and ten owned worker/
score regression commands complete without primary, cleanup or endpoint errors.
The regression includes40 score-state rows,20 legacy generation rows and four
deterministic score-publication races across CPU/Metal. Nine nested app commands
build both full unsigned iOS device and ad-hoc simulator apps. New coordinator
module compilation covers Mac arm64 and device/simulator arm64 iOS18 targets;
Swift uses optimization, complete concurrency checks and warnings as errors.
No simulator boots or device actions occur in this component.

All128 portable M7 methods pass in the final direct diagnostic verbose run;
seven final parser methods and ten package checks also pass. Two preceding full
runs failed only the unchanged timeout-cleanup test with a host process-group
EPERM. Their exact logs and observed exit1 receipts remain published. No source,
cleanup rule or test assertion was changed to obtain the passing run. The first
parser-only logs predate the final lifecycle parser; they are retained historical
output and are not used to qualify the final source. Final source inventories
match before/after the complete suite and all native checks.

Astra cleared source after the lifecycle correction and accepted the completed
code/evidence/index/privacy/link review with no findings. Retain this bounded coordinator; app transport, frozen task artifact
admission, physical100-task responses, post-exit decoding/literal grading and
full model-quality gates remain. Numerical modes, defaults, tolerances and
M7.3's physical lifecycle/sustained/base iPhone14 requirements stay unchanged.
