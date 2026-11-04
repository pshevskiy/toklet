# Gemma task-grading adapter preparation

The adapter admits Gemma's frozen inference contract and journals separately
from the existing canonical grading payload. All100literal scoring fields
(ID/category plus code prefix/tests/entry point or expected answer) match byte
for byte. Canonical cases supply only the unchanged qualified grader's schema;
Gemma prompt IDs, tokenizer, caps and inference provenance remain authoritative.

For both roles, require the exited worker's `tasks_sha256`, the collection
manifest's journal hash and the actual bytes being parsed to agree. Validate
both complete100-case journals before decoding. Original trajectories must
verify every sampled ID and hidden terminal stop; invalid IDs/stops, truncated
caps, wrong pending counts or unresolved trajectories reject admission.

The exact32MB tokenizer bytes are authenticated. Decode each complete visible
sequence once with `skip_special_tokens=True`, retain whitespace/fences and
hash UTF-8 text. Both model workers/groups must have exited cleanly before
decoding or VM startup. The shared qualified isolated grader, source tests,
restoration checks and unresolved-outcome policy remain unchanged. The2-point
decision requires all200resolved model outcomes.

Seven cold controls pass, covering altered grading bytes/order, invalid Gemma
IDs/stops/state, unverified trajectories, literal decoding/hashes, threshold
boundaries, unresolved outcomes, live/failed workers and successful adapter
handoff. Astra identified a missing worker-to-journal hash link; the fix rejects
a successful receipt paired with an alternate valid journal, even when the
collection's path/hash are updated coherently. Both roles are checked before
any decoding. Initial fixture errors and review evidence are retained.

Read-only preparation verifies all100native records, literal grading-field
identity, authenticated tokenizer bytes, the qualified Podman binary and an
initially stopped VM. No decoding, VM start or generated-code execution occurred
while the original model collection runs. Astra approves preparation; actual
paired grading remains pending full source completion.
[Evidence](../benchmarks/results/2026-11-04-gemma-task-grading-preparation-v1.json).

The outer Gemma supervisor now retains the existing TERM-only window owner
through child/group exit and independently verifies the stopped VM. It binds
both exited model receipts and journals, checks inner collection/manifest
identity, and recomputes the decision from all200 ordered resolved outcomes
before publication. Astra caught the initially missing outcome-inventory check;
the corrected21-control suite passes, including empty outcomes, a contradictory
decision, changed evidence and unverified restoration. This is preparation;
actual paired grading remains pending. [Supervisor evidence](../benchmarks/results/2026-11-04-gemma-grading-window-v1.json).
