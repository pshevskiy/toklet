# Task-based mobile model ranking

24 October 2026. Find best model offers Balanced, Coding and Fast replies.
It ranks up to five entries from the existing pinned Chat catalog. The catalog
currently has three profiles; unsupported Hugging Face results do not become
runnable recommendations. This is an independent small implementation inspired
by [llmfit's task-based scoring](https://github.com/AlexsJones/llmfit/blob/b8124fb6c4ae9a943dac86440020dc0bdefdada1/llmfit-core/src/fit.rs).
No llmfit source or dependency is bundled.

## Eligibility and ownership

The existing unified-GPU/physical-memory requirements and `HubAdmission.check`
filter candidates before scoring. A resident pinned profile can remain listed
without requiring room for a second copy. It does not receive invented free
memory: its headroom score still uses current available allowance and can be zero.
Each refresh or task change obtains a fresh admission reading. Failure clears
recommendations and displays the device-reading error.

Use/download retain pinned revision/file validation, storage admission, current
post-release process admission, native graph/tokenizer compatibility, exact
resource planning and the loader's final validation. Scores do not authorize
loads, admit new representations, change model eligibility or alter app/engine
limits. An estimate neither reserves memory nor guarantees loading later.

## Scores and assumptions

All three components are dimensionless estimates on a 0–100 scale:

- **Capacity:** relative stored-model size, logarithmically mapped from35 for
  the smallest supported artifact to100 for the largest. This is a capacity
  proxy; it is not measured answer quality, parameter count or a coding test.
- **Speed:** `100 * sqrt(smallest artifact bytes / candidate artifact bytes)`.
  This expresses relative weight-reading pressure, not predicted tok/s or TTFT.
  Kernel, context, model architecture and temperature effects remain unmeasured.
- **Headroom:** `100 * max(0, available - profile allowance) / available`, or0
  when no allowance is available. Admission uses the original integer policy;
  floating-point scores cannot relax its boundary.

| Preference | Capacity | Speed | Headroom |
| --- | ---: | ---: | ---: |
| Balanced | 65% | 15% | 20% |
| Coding | 80% | 5% | 15% |
| Fast replies | 15% | 65% | 20% |

These weights express the requested preference. Coding favors capacity without
claiming any exact artifact's code quality was measured. Sort by unrounded total,
then repository ID for deterministic ties; display rounded scores and components.
The UI labels scores Estimated and explains their basis in Why this rank.

Current models share the same512-position Chat load policy and native256-token
output cap. The UI displays those actual limits; context is not used as a
discriminating score when it is identical for every candidate. The capacity
constant is shared with actual Chat loading. Gemma4 and single-reply131072-token
support remain separate engine priorities, with no new eligibility in this UI.

## Verification

Controller tests cover task-dependent ordering, over-budget/unsupported-device
rejection, conservative resident treatment, deterministic finite bounded scores,
changing headroom, task retention and missing-memory failure. Existing model
replacement checks retain fresh post-release admission. The physical UI test
changes Fast replies to Coding, inspects score details and current context limits,
opens Qwen from the ranked result and generates a reply using an isolated history.
The final signed build passes all five physical tests on iPhone14Pro/iOS27.0.1:
ranking preferences/use, dashboard/deletion, actual local benchmark and unsupported
Gemma search, repeated model switching, and reply-limit/Expert output. Eight final
local commands also pass: strict Swift6 compilation, controller regressions,
CPU worker, light/dark rendering, real Smol benchmark/Chat, replacement admission
and both device/Simulator module checks. The
[receipt](../benchmarks/results/2026-10-24-mobile-model-ranking-v1.json) binds the
exact sources, signed app and unchanged native2d7 package. Owner chat bytes match
before/after; the passing app is installed and normally launched.

Two earlier phone attempts are retained with successful dialog-byte checks and
verified rollback. The first found a DisclosureGroup identifier overriding child
score identifiers; the container identifier was removed. The search results were
present but hidden by the focused keyboard, so Search now has an explicit44pt
button and resigns the actual iOS responder as well as SwiftUI focus. The test
waits for search completion and scrolls the list to a visible result. A reply
test also advanced while the native menu still showed128; it now waits for a
hittable option and confirms256 is committed before closing Settings.

Scores remain estimates. These checks do not qualify new numerical quality,
controlled throughput, peak memory, other devices, Gemma4 generation or131072-token
replies. The separate engine work retains those priorities.
