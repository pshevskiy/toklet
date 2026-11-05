# M9 decision for Mac affine prefill

Retain four-token affine8 prefill as an explicit Mac opt-in for the pinned
Qwen2.5-0.5B-Instruct-8bit checkpoint
`6474d4c1a05bd7bd634a9c993560e4e7272eda3d` on Apple M1 Pro. The route
reuses reconstructed weights across four activation rows, with independent
compensated sums and the existing SIMD reduction width. Decode, the final
FP32 head and unsupported layouts use the established kernel.
[Implementation](../benchmarks/results/2026-11-05-m9-affine-tile4-v1.json).

| Independent measurement, 1,024 input tokens | Baseline | Tile | Reduction |
| --- | ---: | ---: | ---: |
| Completed prefill, capacity 1,152 and chunk 128 | 6.568 s | 5.669 s | 13.69% |
| Total teacher-forced request | 14.479 s | 13.579 s | 6.22% |

The P1 timing and reverse-order control reproduced the primary gain.
The small decode differences varied across the two P1 pairs. The
independent P2 pair passed all controls. These measurements use frozen
IDs and completed numerical work. [Baseline](m9-mac-baseline.md).
[First screen and control](m9-p1-affine-screen.md).
[Independent timing](../benchmarks/results/2026-11-05-m9-affine-independent-v1.json).

All 10,000 loss rows and 100 task trajectories match exactly. Active tiled
state passes failed-candidate recovery, cancellation and capacity rejection.
The separate 512-position/chunk-32 Mac controller diagnostic verifies cached
load, Send, Stop, New Chat and clean close. Its raw prefix command reuses
36 tokens with exact fresh-reset output and state. Sampled physical peak
was 722,633,928 bytes. [Qualification](m9-p2-affine-qualification.md).

This closes the M9 first delivery in the reviewed Mac opt-in scope.
The diagnostic build uses `affine-tile4-mac-probe`. Managed sessions
still use their existing policy. Capacity transitions and rendered UI
qualification remain open for the tile, and physical iPhone work remains
deferred. P3–P5 and other conditional mechanisms retain their own gates.
[Follow-on queue](next-performance-milestone.md).
