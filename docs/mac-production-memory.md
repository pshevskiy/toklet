# Production Mac app memory witness

5 November 2026. The unmodified optimized production Toklet Mac bundle completes
cached managed Qwen2 load→Send→visible reply→normal Quit below its profile's
1.5GiB process allowance. This closes a representative Mac whole-app memory
check; the other four managed profiles have no new production-app peak evidence.

| Exact run | Value |
| --- | --- |
| Model | Qwen2.5-0.5B-Instruct-8bit, `6474d4c1a05bd7bd634a9c993560e4e7272eda3d` |
| Session |512positions, chunk32; selected reply limit128, decoder cap256 |
| Budgets |1GiB numerical,128MiB text;1.5GiB profile process allowance |
| Process-lifetime RSS peak |816,709,632bytes, Darwin `wait4.ru_maxrss` |
| Greatest observed physical high-water |728,090,064bytes, external `proc_pid_rusage` |
| Physical observations |662samples; last at68.895seconds after owner start |

The process-exit RSS peak covers the full process lifetime. The physical value
is the greatest reported lifetime high-water obtained before exit; it does not
claim observation of the unsampled final interval. Owner elapsed time including
post-exit file guards is69.379seconds. Timing and UI tok/s are not a benchmark or
performance qualification.

The existing production build uses current unmodified Swift sources and the
coherent native40 package. Before and after, hash every bundle file, current
app source and exact model/receipt. A supported `--data-root` argument selects
an exclusive artifact root; an APFS-cloned, already verified cache undergoes the
ordinary Models→Use action. No fixture entry, controller-only executable or
source instrumentation replaces the real app. The UI attachment is independently
checked against owned PID63081 and its exact launch arguments before interaction.

Send uses “Reply with the word hello.” The app visibly replies “Hello! How can I
assist you today?”; the saved conversation independently matches. Settings shows
512/chunk32/1GB and128 selected reply tokens. The pre-run plan recorded a256 reply cap; inspection distinguishes the actual
selected default128 from the decoder cap256. This corrects the configuration
record; the run does not claim a256-token request, and no setting was changed. Normal Quit logs successful shutdown, exits0 and leaves no process
group. Standard-root conversation/preferences hashes remain unchanged during
this measured run. Compare process peaks with the profile allowance, separately
from numerical admission and the UI's overall6GiB monitor label.

A preserved first attempt timed out after900seconds while the UI tool took2314
seconds to return. Owned PID61900 loaded no model and was killed; its receipt
stays failed. The delayed tool separately launched PID62972 without the isolated
argument; its empty window was closed and standard-root data was left untouched.
Neither process qualifies a load/reply peak. The fresh measured attempt uses a
new root and verifies PID/arguments before operating the app.

Saved raw receipts, source/model guards, monotone physical counters, exact
conversation/limit checks, clean exit, workspace preservation and policy replay
pass. Astra approves the representative whole-app check. Physical iPhone14Pro, whole-window execution, other
production profiles' peaks and numerical/quality qualification stay separate.
[Evidence](../benchmarks/results/2026-11-05-mac-production-memory-v1.json).
