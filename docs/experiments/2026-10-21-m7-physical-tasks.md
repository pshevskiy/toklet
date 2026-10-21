# Sealed physical task collector

The collector prepares and admits raw phone task responses while preserving the
frozen100-case workload. It adds no tokenizer, answer repair or grading and
changes no kernel, prompt, sample cap or quality gate. Full physical collection
and independent grading remain pending; M7.3 remains open.

The job derives a bounded canonical58,475-byte JSON task set from the exact
original task JSONL and frozen manifest: all100 literal IDs/prompts/caps, original
EOS order, vocabulary151936, capacity562 and commands of at most128 samples.
Full payload SHA is `ccb24a6f0b6e4798cabc393c475e377d3b941a640e5b6414cc9359bf5ee5659b`.
The short real-model probe uses only the first case capped at8 and makes no task
quality claim. Tiny jobs are separate two-case component fixtures.

The existing signed-install/owned-process/storage transport gains a separate
`--tasks` mode tied to the reviewed additive score package and six reviewed app
runner source pins. Its task file stays outside the model/packed inventories.
Admission counts every transferred byte plus the existing2GiB reserve and
requires fresh same-device observations before each copy, including `copy-tasks`.
Ready reuse preserves the original storage decision and exact signed app/job.

The parser checks ordered chunks, hidden EOS accounting, exact concatenated IDs,
committed/pending state, per-case reset acknowledgements, serial caller/native/
callback clocks, positive actual admission, process peaks, environment,
installation proofs, release and raw receipt endpoints. Full task completions
remain unconstrained by expected answers. Tiny jobs require the independent
`[47,47,47]` and `[47,47]` output-limit responses.

Astra found that initial runner pins were verified once without being retained
and that protocol-valid wrong tiny responses could pass. Both are corrected:
all six pins persist through launch/endpoint guards; wrong IDs and premature EOS
fail the independent tiny oracle. Exact previous sources and draft are retained.

Final checks pass153 methods across16 isolated portable modules/16 clean owned
commands, including13 targeted collector methods. Synthetic checks cover100
cases/25,600 visible events, EOS at sample1/128/129/256, source mutation after
admission, wrong outputs, storage payload/copy clocks and sealed preparation.
The earlier source passes151 methods; its changed bindings map to exact saved
sources. No combined-suite pass is claimed. An explicit frozen-input audit
confirms100 unchanged cases and all six tiny/real short/full job shapes, without
loading a native model. This Python-only transport task reuses the accepted
CPU/Metal/app-build evidence from [literal-ID compatibility](2026-10-21-m7-task-id-compat.md);
no native/device run is attributed to these portable checks.

Astra accepted source and completed component evidence with no findings.
Retain this bounded raw collection path. Full loss/task quality, lifecycle,
sustained behavior, supported limits and base iPhone14 gates remain open.

The [qualification record](../../benchmarks/results/2026-10-21-m7-physical-tasks-qualification-v1.json)
binds current/parent/pre-review sources, all exact public copies, commands,
original frozen workload and the accepted runner proof.
