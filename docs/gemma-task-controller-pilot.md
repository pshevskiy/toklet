# Original task-controller pilot

The first frozen code, prose and continuation cases pass the source controller
pilot with full output reserves256/32/16. They sample61/2/2IDs respectively;
all three terminate on hidden stop106. Every one of the65decisions, including
the hidden stops, matches a single stateless teacher-forced text pass.
No generated task was graded.

The controller uses the qualified resident BF16 model and session-owned stock
KV. Samples remain separate from committed history: the final sampled token
is pending, including an output-limit final visible token. Stops consume one
sample slot but remain invisible/uncommitted. Lowest-ID ties, complete output
reserve and empty resets are checked. Stateless verification projects one
sampling position at a time through the stock tied head/softcap and withholds
verification on any divergent ID.

Three cold controls cover stops/caps/ties/pending state, hidden-stop replay,
deliberate divergent trajectories and a tiny real model. Runtime/source/input
seals, exact prompt/recipe/case identity, finite state and owned exit pass.
The expected dynamic PLE journal is independently reconstructed from frozen
prompts and sampled histories; source extent replay matches all10,201reads/
4,567,886,918bytes. On M1 Pro16GiB peak RSS is4,995,743,744bytes and physical
footprint4,935,520,432bytes, below6GiB. Required Rust inputs are unchanged.
Astra approves code/results and reusing these three receipts in the full suite.

Observed generation/replay seconds are31.27/29.77 code,8.35/8.31 prose and
11.25/11.02 continuation. These are pilot diagnostic costs, not a full-suite
ETA or retained performance result. Next collect the remaining97source cases
with one CPU thread and exact trajectory verification, while preparing the
native existing-generation-path collector. Require all100before grading.
The task gate, prior fidelity failures and model/app adoption remain open.
[Evidence](../benchmarks/results/2026-11-04-gemma-task-controller-pilot-v1.json).
