# Grading provenance fixture repair

The full-plan audit found two existing provenance regressions failing before
their inventory and binary-replacement assertions: the mocked producer inventory added
only the old Q4v2 source, while three more Rust modules now exist. The test helper
now adds synthetic identities for missing current Rust sources, preserving all
historical identities. This applies only to mocked unit inputs; production
admission and historical native receipts are unchanged.

A new regression removes Cargo.lock from the mocked record and confirms the
helper does not restore it and production's exact inventory gate rejects it.
Existing per-entry deletion, extra-entry, binary-replacement and frozen-contract
replacement checks remain intact. This fixes test coverage without admitting old
evidence against new code or changing numerical/quality tolerances.

The [check receipt](../../benchmarks/results/2026-10-21-full-plan-checks-v1.json)
and [raw checks/snapshots](../../benchmarks/results/2026-10-21-full-plan-checks-raw-v1.json)
retain the initial failed suite and old test source, the repaired ten-test run,
the complete final suites and exact runner/receipt bytes. A second audit attempt
failed because its standalone Python invocation omitted tools/bench from
PYTHONPATH. Production's grading window already supplies that path; correcting
the audit invocation required no production import change. Both failed attempts
remain recorded separately from the final successful run.

Final local checks pass with no endpoint errors:

- Cargo 1.90 formatting and CPU/Metal-feature all-target Clippy with warnings denied.
- CPU: 111 passed, 8 ignored; Metal-feature: 123 passed, 59 ignored.
- Python benchmark/experiment modules: 816 tests across 113 modules, no skips.
- Python reference/quantization modules: 71 tests across 14 modules, no skips.
- Focused grading-admission regressions: 10 passed.

Rust used an isolated target directory. Both protected release CPU executables
retain their original hashes. All check source pins replay, using retained exact
snapshots for the original PLAN and pre-fix test. No model download, hardware
qualification or new speed/memory/quality claim is made. The ignored Rust suites
remain distinct from earlier explicit GPU/model/device evidence.

Astra accepted the source and completed evidence after independently replaying
all retained raw hashes, source pins, suite counts and unchanged production/CPU
identities. Retain the test-only repair.
