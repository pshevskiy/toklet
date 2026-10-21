# iPhone full held-out loss qualification — 21 October 2026

The wired iPhone 14 Pro (iPhone15,2), iOS 27.0.1, passes the unchanged held-out loss gate for both selected M7 model paths. This closes physical loss evidence for this device; task quality, lifecycle, sustained small-model testing, energy observations and the final supported-limit decision remain separate. The original base iPhone 14 target is not replaced by this Pro result. M7 remains open.

| Path | Scored targets | Mean NLL | Matched FP32 mean | Increase, nats/token | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| Small compensated Half | 10,000 | 2.8835716761956602 | 2.883725149929484 | −0.0001534737338237946 | Pass ≤0.05 |
| Larger frozen Q4 v2, group 8 | 10,000 | 2.5142232720120155 | 2.485958031019153 | +0.028265240992862317 | Pass ≤0.05 |

Both means exactly equal the corresponding frozen desktop M6 Online means. This is a numerical result, with no speedup or energy claim. The corpus, original target IDs, 1,024-target block policy with overlapping context token, aggregation and quality threshold are unchanged. Online attention, FP16 KV and full-logits diagnostic readback remain explicit; CPU/GPU defaults are unchanged.

## Collection and independent replay

Collection used engine revision `c2e44fb772a9dcea1fa70c8ca64e1baa20218e23`, the reviewed native package v5 and signed app preparation v14. The controller, app, engine and job inputs stayed frozen throughout all 14 stages. Two tiny CPU/Metal quality jobs precede small/larger short and full jobs; each real job has its own fresh tiny storage proof, sealed preparation and no-copy quality collection. Six scored jobs contain 20,012 targets in total. All 216 owned host commands exit cleanly; all 28 start/end owned-process checks observe natural absence without signals during this driver.

Independent replay checks exact consumed receipt and raw-console hashes, installation identity guards, launch metadata, device/boot endpoints, storage admissions and transfer seals, original target sequence and native score states, full target counts and stable `math.fsum` aggregation. The tiny CPU losses are independently reconstructed from the pinned FP32 fixture within the unchanged 1e−5 bound; the driver’s CPU/Metal tiny summaries must match consumed raw rows. Four driver regressions and ten analyzer regressions pass.

Full-loss peak whole-process physical footprints were 1,028,770,136 bytes (small) and 1,835,338,624 bytes (larger), below the explicit 2.5 GiB trial process cap. Numerical admission uses the separate small 1.5 GiB / larger 2 GiB caps. Small full collection observed thermal states 0/1/2; larger observed 1/2. Every scored sample was foreground with Low Power Mode disabled. These observations qualify these workloads on this device, not general mobile memory limits. Energy is unset.

## Retained preparation attempts

Preparation v12 refused an old library symlink before device commands. After its exact old binding was checked and the generated link switched to reviewed package v5, preparation v13 refused a pre-existing harness process. The separately reviewed identity-checked operational reset ended that process; v14 then built, signed and installed successfully. These preparations are outside the 216-command loss driver and its 28 natural absence checks.

Loss driver v1/tiny CPU v1 refused reuse before any child command because preparation v14 lacked two historical controller source copies. The exact source bytes were restored against v14’s existing bindings; new v2 output paths retained the failed v1 receipts. No target, gate or app/engine code changed to retry. Earlier analyzer source before the independent tiny-oracle review fix is also retained.

## Evidence and decision

The [qualification](../../benchmarks/results/2026-10-21-m7-phone-loss-qualification-v1.json) pins 1,308 inputs and preserves 152 exact public evidence copies, including 81 frozen collection source files. Private installation/device/signing evidence stays local, pinned by digest; public copies are checked for actual private identifiers. The [analysis](../../benchmarks/results/2026-10-21-m7-phone-loss/m7-phone-loss-analysis-v1/analysis.json) retains all six run summaries. The [driver](../../benchmarks/results/2026-10-21-m7-phone-loss/m7-phone-loss-driver-v2/execution.json) retains all stage receipts and tiny oracles.

Astra accepted the completed staged evidence with no findings after independent byte-for-byte analysis replay. Retain the physical loss result. Do not infer full task quality, actual OS lifecycle acceptance, sustained small-model qualification, energy, controlled speedup, supported limits or M7 completion from it. Previous failed/rejected configurations and the base-device requirement remain unchanged.
