# M7 sealed teacher-forced quality harness

The iOS harness now has an explicit `--quality` diagnostic mode that admits a
config and corpus, scores the exact teacher-forced targets, streams raw scalar
rows and verifies cleanup and artifact endpoints. This prepares full physical
corpus collection; this task executes six actual tiny targets on Mac only.

The exact13-field JSON config pins model/packed inventories, token-file SHA,
backend, numerical/process budgets, target count, block size, capacity and chunk.
Unknown fields reject. Bounds are1..10000 targets and1..1024 targets per block,
capacity exactly block+1 and chunk1. The full frozen corpus still requires10000
targets and1024-target blocks; shorter settings exist for component checks.
Corpus bytes are exactly4*(targets+1), little-endian UInt32, and independently
hashed from the same bounded decoded buffer. Path traversal, child symlinks,
corpus placement inside model inventories and changed size/SHA reject. No
alignment assumption is made when reading byte groups.

Existing path/hash/inventory rules are shared through ordinary functions with
the measurement runner. Chunk pools, regular-file/child-link rejection, trusted
root canonicalization, exact file inventory and file limits remain. The existing
notification observer now serves both diagnostics; its weak owner and protected
interruption count retain the same cancellation/session-release/unload behavior.
Three exact parent app/project sources remain bound for historical receipts.

The off-main runner checks foreground/interruption state before and after every
bounded worker command and before output. It validates live numerical admission,
records own-process physical/lifetime peaks per target and at load/block/release
checkpoints, and streams only completed target rows. The process budget is an
observed post-operation rejection policy, not an OS allocation hard cap. It
closes the dedicated worker, checks weak release and rehashes model and corpus
endpoints on success and failure. Invalid target/block extremes cannot overflow
final result bookkeeping. A failed sink cannot return a completed summary.
Successful unload explicitly requires zero native admission and no live owners.

The app's quality mode requires exactly its config name/SHA arguments, disables
the diagnostic idle timer, waits for startup foreground and emits installed-file
proofs before/after the run. The Xcode project now includes both coordinator and
quality runner. No UI score controls, tokenizer, task generation or grading are
added. Full-quality/device/M7 flags remain false: successful collection alone is
not numerical or device qualification.

The [qualification receipt](../../benchmarks/results/2026-10-21-m7-quality-harness-qualification-v1.json)
binds196 paths,58 exact published copies, seven code files, three parent-source
mappings and26 retained helper snapshots. All109 portable M7 checks pass, including
11 targeted quality-parser methods with corruption mutations. Nineteen actual
native cases pass: tiny CPU/Metal collections, ten policy rejections, exact-field/
config-SHA rejection, corpus-size/symlink rejection, output failure and both
model/corpus endpoint mutations. Six actual target rows reproduce prefix/order/
loss aggregation; CPU losses match the independent pinned tiny FP32 oracle within
1e-5. The five expected failed-run results remain raw, with exact failure/proof/
cleanup correspondence. They are excluded from successful corpus results.

Seven owned outer steps complete cleanly. Strict optimized Swift and complete
concurrency checking pass with warnings as errors. The existing measurement suite
also passes17 cases/39 actual Mac requests, including the independently hashed
256MiB padding regression under192MiB. This checks the shared admission/hash
refactor, rather than inferring correctness from inspection. Both complete iOS
simulator and device Release apps build with the new sources; platform and
app-bundle checks pass. Existing orientation/AppIntents Xcode warnings
remain in exact logs. No simulator boot, signing or phone launch occurs here.

Astra's source review found that the first parser grouped marker types without
checking stream order and omitted validation of the load bracket. It now requires
exactly CPU3 rows/result, Metal3 rows/result, five ordered failures and final
summary; unknown/error output rejects. Load time must be finite, nonnegative and
non-Boolean. Corresponding mutations pass. Astra cleared final source before the
first native run, which passes. Astra accepted completed evidence with no findings after independently
verifying196 pins,58 exact staged copies, seven code pins, all historical/helper
mappings, quality/failure/measurement/SDK replays,109-test record,382 links,
privacy against actual private identifiers and whitespace.

Retain the quality harness and shared mechanisms. The physical collector still
must bind the exact new signed installation, frozen corpus/model inputs, storage
admission, actual device/foreground state and natural exit/absence, then compare
full corpus loss with the unchanged reference/gate. The full100 task responses
and grading, supported mobile limits, actual OS lifecycle recovery, sustained
and base iPhone14 evidence remain. No numerical kernel/default/gate change,
phone performance result or M7.3 completion follows from these component checks.
