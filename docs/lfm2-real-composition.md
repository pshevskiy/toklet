# Bounded LFM2 resident composition

B4e3b extends the existing concrete resident implementation from its qualified
tiny profile to the selected real LFM2 geometry. The five-file proposal and
narrow lint correction have Astra review. Formatting, both strict all-target
Clippy configurations, default457/Metal556 tests and the release Metal library
test build pass (20/181 ignored; suites overlap). Both existing Tiny release GPU
regressions pass, followed by independent saved-byte audit. No real model has
been constructed or executed on GPU in this task; ordinary factories/app
profiles remain unchanged.

One cold geometry resolves dimensions, exact layer routes, owner counts, stage
bases and chunk limits. Tiny keeps H256/I256/V64, three layers and capacity12.
Selected real has H1024/I4608/V65536, sixteen layers, six attention owners, ten
convolution histories and requested capacity1–512. Layers use concrete tensor
fields and one mechanism match; no graph VM, hot name lookup or new shader is
introduced. Scalar execution and all existing shader/cache code are preserved.

The singular attention/two-history storage becomes bounded owner vectors.
Every candidate history is stamped and checked on PhaseA; five existing paired
validators cover all ten real histories. KV remains the sole position/epoch
authority. Successful KV copies precede infallible history-role, logits and token
publication. Uncertain copies poison the whole session; reset preflights all
owners before mutation. Ordinary successful forwards allocate no new Rust
vectors by construction; the existing hardware allocation test remains required.

View banks cover only 1–min(capacity,32), and last-row views use chunk-relative
offsets. Hidden/head norms, BCx/FFN scratch, per-head cache strides and snapshot
layouts derive from the cold geometry. Real zero bias is65536+7 Half elements;
the early Half-buffer preflight includes I4608, which exceeds3H3072. There is no
capacity×vocabulary logit history: full diagnostic banks remain32 rows.

The snapshot plan has57 tiny or293 real stages, matching the independent source
inventory. Real physical numerical predictions include335 model buffers,
54,530,796 bytes of diagnostic GPU buffers and88,080,384 bytes of reserved host
readback. These are constructor equations, not observed real allocations.
Metadata, shared framework allowance, caller/source overlap and per-device
buffer limits remain separate checked components.

The first cold run stops at an unused enumeration index under strict Metal
Clippy. The failed receipt remains; the successor changes only loop iteration
and equivalent test-only fault targeting. Source fixtures, numerical gates and
workloads do not change. Existing tiny source/fault/state/allocation tests are
the GPU regression gate; unchanged cache/Gemma suites remain prior supporting
evidence. Real native capture/checker/session/text/memory qualification is B4e3c.

The saved Tiny audit covers all3135 mathematical groups/1,923,728 values and330
boundary groups/199,936 values against unchanged MAIN/WIDE arrays. Both have
zero exceedances; maximum absolute error is6.103515625e-5 and maximum fatal MAIN
fraction0.030098723814110283. All15 schedules/55 chunks/133 token rows,144 state
records, physical snapshot/scratch hashes,1044 masked positive zeros and66
immutable numerical model owners pass. Nine injected failures,18 replay chunks,
seven validator rejections and12 ordinary release Rust zero-allocation forwards
pass. The partial-copy witness changes exactly128 actual head0 K words before
forcing a protocol error; it is not a hardware-failure simulation.

Tiny numerical buffer sizes remain unchanged. The new owner vectors/cold layout
add168 bytes of Rust session metadata in each budget mode, all included in
admission. Ordinary/all-logits/diagnostic admission is101,719,882/101,735,754/
104,534,278 bytes, including32MiB caller and64MiB framework allowances; these are
accounting bounds, not measured resident peaks. The two raw logs retain
24,670,544/30,090,938 bytes. Direct samples26/8 observe maximum resident
43,188,224/49,463,296 bytes and lifetime physical-footprint35,177,120/44,335,800;
sparse sampling does not establish complete peaks or whole-app fit.

The first GPU attempt exits before model construction with no Metal device;
the second job is unrun. That receipt stays failed. The successor uses
platform-approved Metal access and the exact same retained executable, source,
controller and gates. Six cold, two successful GPU and three original source
groups are closed with empty endpoint errors; the failed group is also closed.
The seven unchanged shared-cache/Gemma jobs were not rerun. B4e3c now requires
the actual-input checker, complete native capture inventory and bounded real
session/text/memory experiment. [Machine-readable evidence](../benchmarks/results/2026-10-29-lfm2-real-composition-v1.json)
pins the code, actual executable, receipts, raw outputs and Astra audit.
