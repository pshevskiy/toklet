# Original Llama task comparison

On Apple M1 Pro, the pinned native MLX8 artifact passes the frozen 100-task
relative quality gate against the original BF16-to-FP32 reference. Both workers
use the [same frozen inputs](llama3-task-contract.md), 566-position diagnostic
sessions and greedy decoding. Every original trajectory matches its independent
stateless replay before grading. The reference uses the qualified streamed FP32
cache; this is a correctness comparison, not an app latency benchmark.

| Literal source-test outcomes | Original | Native |
| --- | ---: | ---: |
| HumanEval | 12/40 | 14/40 |
| TruthfulQA | 9/30 | 8/30 |
| LAMBADA | 1/30 | 1/30 |
| Equal-weight total | 22/100 | 23/100 |

The native pass-rate loss is −1 percentage point, passing the unchanged ≤2-point
gate. These small absolute scores describe this frozen prompting/scoring setup;
they do not establish general model quality. All 80 literal code programs run
in the existing qualified isolated Linux executor. No generated code executes
on the host. Its source-test evidence does not attest resistance to adversarial
code tampering with the shared test interpreter. The 120 text outcomes use the
existing exact graders, with no completion repair or case exclusion.

The shared runner now owns VM/client/image admission, execution journals,
classification and restoration for model-specific adapters. Qwen retains its
seven-phase collection admission. Llama admits its two completed named workers,
verifies all journals and decodes the exact verified tokenizer bytes. The outer
owner seals sources and inputs and accepts a decision only after independent
VM shutdown and process-group exit. Saved evidence replay reconstructs all
programs and recomputes all 200 outcomes and aggregates.

The first collection stopped below the disk reserve; its completed native worker
was retained. Astra reviewed cleanup of complete inactive object hardlink groups
under the Cargo lock, preserving linked products. The retry reran only the
original worker. Peaks are 1,205,125,120 bytes native and 5,345,181,696 bytes original,
both below 6 GiB. The first grading attempt ran zero programs: timezone-rendered
Podman build strings differed. Pinning the child timezone to Europe/Moscow
reproduces the qualified full version JSON; the exact admission gate remains.
Both failed receipts are retained.

Nineteen focused controls, saved-result replay and final Astra review pass. No numerical Rust or
Swift source changes; prior 549 default/657 Metal checks remain applicable.
The separate [10,000-target loss gate](llama3-original-loss.md) also passes.
A combined qualification/adoption decision remains separate: this task does not
enable ordinary discovery, enlarge the managed app's 512-position ceiling or
qualify iPhone 14 Pro. That physical device is unavailable.
[Evidence](../benchmarks/results/2026-11-03-llama3-task-quality-v1.json).
