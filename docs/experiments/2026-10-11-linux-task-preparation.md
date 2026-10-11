# Literal task evaluator preparation, 11 October 2026

The [trusted child](../../tools/bench/linux_task_child.py),
[receipt validator](../../tools/bench/linux_task_evidence.py), and
[input transport](../../tools/bench/linux_task_transport.py) prepare literal
HumanEval source-test execution under the previously qualified
[Linux policy](2026-10-11-linux-code-isolation-retry.md). They are not yet an
adopted executor. No generated program or held-out quality case was executed.

The caller must provide the unchanged result of `task_cases.human_eval_program()`.
The child bounds the UTF-8 source to 128 KiB and its serialized input to 1 MiB,
checks its case/source binding, loads a fixed standard-library inventory, and
enters the existing policy before compiling or executing source. Assertions
remain enabled. Preloading does not inject module names into candidate globals.
The entire literal program, including its source-test invocation, must return
normally to pass. Caught exits, syntax errors and assertions fail. Import or
unsupported stream compatibility failures remain unresolved infrastructure
errors. Ordinary text output is framed in chunks of at most 1,024 characters.

The transport writes bounded stdin with partial-write handling and closes it
for EOF while draining stdout/stderr. It enforces the existing six-second wall
and combined 65,536-byte output bounds, preserves raw transport exit status,
and uses a separate twenty-second cleanup wait for its owned process group.
The receipt validator checks admission, source/input identities, raw captures,
terminal events and cleanup before returning a binary outcome. Exit 137 alone
is ambiguous; independently recorded wall/output stops or container OOM state
can resolve an admitted resource failure. Duplicate, malformed or trailing
status events cannot pass. A clean exit without terminal completion fails;
nonzero exits without a terminal outcome conservatively remain unresolved.

Container isolation limits external effects. Candidate code and source tests
share a Python interpreter: status events do not attest against deliberate
interpreter or grader manipulation. The fixed preload inventory is explicit,
but bootstrap dependencies may also be resident. These limitations are not
claims of adversarial grading integrity.

The authored offline tests cover
[child semantics](../../tools/bench/test_linux_task_child.py),
[receipt rejection](../../tools/bench/test_linux_task_evidence.py), and
[transport lifecycle](../../tools/bench/test_linux_task_transport.py).
All 39 targeted tests and the complete 136-test benchmark suite passed.
Transport tests mock all process creation, signals, descriptor operations and
process-group inspection; child admission tests mock policy entry. No native
container admission is inferred from these fixtures.

The initial complete suite run inside the filesystem sandbox retained two
failures from existing subprocess fixture tests: `ps` and owned-group signals
returned EPERM. Repeating with the required process access passed unchanged.
The [check receipt](../../benchmarks/results/2026-10-11-linux-task-preparation-checks-v1.json)
retains both logs, source identities, and Rust validation. Formatting, Clippy
with warnings denied, and ordinary Rust tests use the existing supported
configuration. Rust builds use an isolated target; frozen root release binaries
and protected numerical/quality helpers are unchanged.

Astra accepted the completed preparation task with no actionable findings,
independently passed all 39 targeted tests, and verified source/log identities,
test counts, scope, links and whitespace.

A fixed interactive container controller must still verify configuration and
ownership, wait/reinspect/remove each container, and restore the VM. Its input
and output transport, runtime compatibility and forbidden operations require
native authored qualification before generated programs can be graded. No
quality score, executor adoption or milestone completion follows from this task.
