# MiniCPM5 console GPU attribution

The explicit CLI profiler passes two controlled cohorts with exact ordinary versus
instrumented logit/state parity. Checks and observations below are complete;
Astra approves scoped retention; [review/commit](../artifacts/minicpm5-profile-2026-11-10/publication-checks.json).
Runtime kernels, ABI, model defaults and
ordinary telemetry remain unchanged. Screen sharing is unnecessary.

## What the command observes

The [prospective contract](minicpm5-profile-contract.json) freezes the same
publisher-authenticated Heretic affine8 weights, original independent reference
and raw token history as the [short console evaluation](minicpm5-console.md).
Two sessions share immutable weights once; their KV, scratch and histories are
independent. Capacity 512, chunk 32, Half execution, compensated normalization,
Reference attention, FullLogits and the 6 GiB ceiling stay fixed.

Each cohort runs three warmups and ten measured samples. Each sample pairs the
four completed evaluations at contexts 4/5/6/7, reversing the first executor by
sample/step and reversing the initial order in the second cohort. Every pair
compares all 130,560 logits bitwise, state and the selected token. Its logit SHA
also equals the corresponding previously qualified capture, under the original
dual-profile gates. The final teacher token is EOS and remains pending; this
fixed-work diagnostic does not run text formatting or stop handling.

The existing bounded Metal counters expose 718 encoder intervals per phase:
embedding, 17 roles in each of 42 layers, final norm, untied head and validation.
The worker checks every kernel label and adds stable layer/role/index metadata.
Raw GPU ticks retain their paired CPU/GPU calibration; ticks are never assumed
to be nanoseconds. The reader checks the conversion, monotonic nonoverlapping
stage intervals, typed coverage, completed GPU/host clocks, disjoint load plus
all phase walls, exact owners and the completed owned receipt's report SHA.

Resolve, full-logit comparison, selection and JSON publication are outside each
phase clock. Interleaved pairs do not supply ordinary request time or TTFT.
Stage rankings describe instrumented execution; unobserved gaps remain
unattributed. Counter overhead cannot be subtracted from individual operators to
invent ordinary operator costs. The ordinary telemetry 1% bound stays unqualified.

## Repeated observations

Both cohorts ran on MacBookPro18,3, M1 Pro/16 GPU cores/16 GiB, macOS 26.6.2
(25G83), AC power, nominal thermal state and Low Power Mode off at all four
boundaries. No continuous thermal or tail-latency claim follows.

| Context and rows | Ordinary phase median, runs 1/2 | Instrumented median, runs 1/2 | Diagnostic phase inflation |
| --- | ---: | ---: | ---: |
| 4, four-row prefill | 192.181 / 192.203 ms | 214.310 / 214.542 ms | 11.51 / 11.62% |
| 5, one-row decode | 101.064 / 101.537 ms | 121.011 / 120.621 ms | 19.74 / 18.79% |
| 6, one-row decode | 102.110 / 102.531 ms | 122.013 / 122.075 ms | 19.49 / 19.06% |
| 7, one-row decode | 103.387 / 103.350 ms | 123.354 / 123.331 ms | 19.31 / 19.33% |

The down projection is the largest individual prefill role in both runs:
35.722 / 35.723 ms summed across layers. Gate/up are each about 35.3 ms.
Post-attention normalization is the largest individual decode role, about
22.4–22.5 ms across layers; input normalization is close behind. Projection roles
remain substantial when grouped. These are medians of stage sums, not additive
identities between separate medians or costs inferred for uninstrumented kernels.

Source inspection confirms the existing Half RMS kernel uses one thread per
row for both its serial compensated reduction and its column stores. The next
bounded hypothesis is to retain that exact reduction and arithmetic while
parallelizing only stores across a queried SIMD group. It needs its own numerical,
state and repeated ordinary performance gates; no such optimization is implemented
or accepted here. Other model/quality/adoption gates remain open.

Across both cohorts, **104 phase pairs**, **74,672 calibrated intervals** and
**13,578,240 complete-logit comparisons** pass. Eleven portable stdlib tests and
one positive/58 resealed actual-output damage controls pass. Raw samples,
median/IQR/stdev and counter timestamps remain retained.

Astra found that individually valid counter timelines could be repeated or
placed outside their containing scopes. The offline reader now reconciles
ordered CPU/GPU calibration origins and completed command windows across all
phases. Host sums and the first-to-last host span, including gaps, fit the encode
scope; the initial bracket fits clock calibration; calibrated GPU stage sums fit
completed command duration. Whole CPU/command series plus preceding load fit the
owned process lifetime. Metal's counter-error sentinel rejects explicitly.

Nearest-ns host endpoints allow at most 1 ns per interval and 2 ns for a span;
the bracket comparison allows 1 ns. GPU stage sums round once from exact integer
ratios; command epoch ULP propagation plus 2 ns is capped at 1,024 ns. GPU and
host intervals retain their overlapping scopes. Twelve resealed real-report
controls pass the exact archived original reader and reject under these fixes.
Both original cohorts and public CLI outputs reproduce exactly after the
consumer-only change; no new model run is needed. Astra approves the closure with
no unresolved findings; [review](../artifacts/minicpm5-profile-2026-11-10/astra-review-result.json).

## Memory and checks

Ordinary session admission is 2,785,878,866 B; the profile session adds a 64 MiB
counter allowance. Shared weights count once, producing combined admission
2,998,202,878 B including a 32 MiB application allowance. Maximum exact final
`wait4` peakRSS is 2,806,202,368 B; maximum sampled live-PID lifetime physical
counter is 2,853,390,856 B. These distinct counters do not establish final physical
peaks or an app/device memory qualification. Final physical peaks are unavailable.

Fresh fmt and strict all-target CPU/Metal Clippy pass, as do five explicit example
tests and the controlled Rustup 1.90 release build. The previous 670 CPU and
806 Metal core tests are reused only after all 4,020 non-example inputs are
verified unchanged; five fresh example tests replace the previous four. All
417 native64 source/ABI/builder inputs are unchanged, so its three 46-export
slices, package/link checks and CPU/Metal C smokes retain their recorded scope.
No new package or physical iPhone qualification follows.

The fresh worktree checkout briefly left less than 2 GiB disk free before any
compiler/model job began. Byte-identical APFS clones with independent inodes and
verified inactive compiler-cache recovery restored headroom. The release build's
first admission later rejected before starting a compiler; a second cache
recovery preceded its successful build. Static archives, executables, source,
models, raw results, user work and the app framework link are preserved. A sandbox
hardware-observation attempt also failed; the controlled external observation and
all cohort boundaries passed. These failures do not imply uninterrupted disk-floor
compliance or an engine memory effect.

## Reproduce and inspect

Use the controlled compiler/environment and authenticated local model/reference
from the linked evidence. Outputs must be fresh. The registered Metal example
uses the default Cargo target path below; qualified binary paths/hashes are in
the build receipt.

```sh
cargo build --locked --offline --release --features metal --example minicpm5_probe
target/release/examples/minicpm5_probe profile MODEL REFERENCE_MANIFEST MANIFEST_SHA FRESH_OUTPUT baseline-first
python3 -S -O tools/bench/minicpm5_profile.py FRESH_OUTPUT/profiles.jsonl OWNED_JSON docs/minicpm5-profile-contract.json FRESH_DERIVED_JSON baseline-first
```

The ordinary process does not create its own launcher receipt; the supplied owned
JSON must come from a completed, guarded launcher. The offline CLI prints the
largest instrumented role by depth and diagnostic phase inflation, then saves
all derived samples. It explicitly leaves environment evaluation false; power
eligibility belongs to the separate raw environment envelope.

[Raw cohorts/statistics](../artifacts/minicpm5-profile-2026-11-10/profile-summary-1.json),
[frozen inputs](../artifacts/minicpm5-profile-2026-11-10/profile-freeze-1.json),
[exact consumer-fix revalidation](../artifacts/minicpm5-profile-2026-11-10/revalidation-2.json),
[checks](../artifacts/minicpm5-profile-2026-11-10/checks-1/checks.json),
[release](../artifacts/minicpm5-profile-2026-11-10/release-2/checks.json),
[portable tests](../artifacts/minicpm5-profile-2026-11-10/counter-fix-portable-tests-3.json),
[resealed controls](../artifacts/minicpm5-profile-2026-11-10/damage-controls-2/checks.json),
[checkout recovery](../artifacts/minicpm5-profile-2026-11-10/worktree-clone-recovery-1.json),
[compiler-cache recovery](../artifacts/minicpm5-profile-2026-11-10/compiler-cache-recovery-2.json).
