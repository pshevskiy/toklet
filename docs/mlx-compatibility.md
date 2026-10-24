# Local MLX compatibility report

The initial H1 foundation adds `mlx::compatibility::inspect` as a cold, CPU-only
Rust API. It admits local configuration and tensor metadata using the same
bounded, duplicate-safe configuration document and artifact reader as numerical
loading. It does not expand model weights or create a GPU. A redundant tied head
may still be streamed by the source reader to prove byte identity.

The schema-version-1 report separates these facts:

- `unknown`: a valid, bounded declaration has no implemented native decoder.
- `unsupported`: the request, configuration, artifact or selected Metal geometry
  is rejected, with a machine-readable reason and the loader's diagnostic.
- `compatible_untested`: metadata and graph policy are admitted. This label does
  not grant payload validity, available backend, memory fit, Chat or qualification.

`load_admission` is independent of that metadata label. A compatible artifact can
exceed the caller's memory limit. Scalar estimates use the actual loader's FP32
expansion, largest temporary packed matrix, 64 KiB stream and 64 MiB allowance.
They exclude KV, session scratch, tokenizers and UI. Source admission counts tied
ownership once. Metal runtime strides/padding and session resources require the
actual device planner; the report deliberately leaves those estimates absent.
A compiled backend flag is separate from runtime device checks.

Geometry reports hidden/query/KV widths separately. The requested context and
source `max_position_embeddings` also stay separate. Representation groups state
floating storage or MLX affine packing, element bits, group size and owned source
payload bytes. `dtype` describes floating tensor elements or affine scale/bias
metadata; affine matrix words are U32. Mixed floating exclusions remain visible.

The report never evaluates a qualification receipt. Payload, device and session
checks remain explicitly pending, and text-profile checking is false. Known
family names alone are insufficient: the original Qwen2, Qwen3, Llama and Mistral
source rules still reject unsupported options, missing tensors and malformed
layouts. Metal geometry admission is extracted without widening policy, so both
the report and numerical loader use one implementation. There is no dispatch or
allocation change in numerical loops.

## Validation and task boundary

Offline fixtures cover all four implemented families, tied/untied heads, exact
scalar budget boundaries, unknown versus malformed declarations, duplicate and
oversized JSON, invalid requests, context bounds, overflow, unsupported settings,
missing weights and nonfinite payloads. The latter remains metadata-compatible
but is rejected by actual scalar loading, proving the pending-payload distinction.
All nine existing Metal geometry policies have device-independent shape/error
checks; each family's rejected geometry also matches the actual Metal loader
before GPU creation.

The explicit real Qwen3 metadata probe uses the existing immutable eight-bit
checkpoint: 310 logical tensors, mixed BF16 floating/affine metadata and
633,364,480 owned payload bytes. Its source context is 40,960 while the requested
context is 512. It still grants no text, device-fit or qualification claim.
Formatting, strict all-target Clippy in both configurations, 171 CPU tests,
191 Metal-feature tests, all 19 selected optimized GPU regressions and the real
metadata probe pass. Ordinary suites leave 14 and 101 tests ignored respectively;
those counts are not hardware passes. Exact commands and captures are in the
[task receipt](../benchmarks/results/2026-10-24-mlx-compatibility-v1.json).

Retain this shared reporting foundation. H1 remains open for CLI/native/app
consumers, actual device/session resource planning and exact qualification
receipts. H2–H4 discovery, general downloads and downloaded Chat remain open.
The installed app still offers its existing three pinned profiles. This task
adds no new architecture, kernel, performance or physical-device qualification.
