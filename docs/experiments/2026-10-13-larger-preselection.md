# Single larger checkpoint preselection

Select [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/config.json),
revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, before any larger-model
quality evaluation. The official file inventory pins `model.safetensors` to
SHA256 `dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee`
and 3,087,467,144 bytes. This is one payload/admission-based candidate, with no
search across models or quantizer tuning using held-out feedback.

Only the 660-byte config and first 38,536 bytes of the tensor file were inspected
for this selection. The 38,528-byte header contains 338 BF16 tensors and an exact
3,087,428,608-byte payload. Configuration is hidden 1536, intermediate 8960,
query 12/KV 2 heads, head dimension 128 and 28 layers, tied vocabulary 151936,
rotary theta 1e6 and normalization epsilon 1e-6. It extends the implemented Qwen
family's dimensions; the GPU geometry is not yet qualified.

With the unchanged Q4 v1 quantizer/groups 64 and existing Half exclusions, 196
matrices would occupy 756,953,344 packed bytes; Half exclusions/guards add
467,039,172 bytes, for predicted resident 1,223,992,516 bytes. Largest packed
matrix staging is 8,028,224 bytes. These are header-derived predictions using
current layouts; actual queried strides, full-file hashes and loader/session
admission must confirm them before use.

The independent FP32 reference weights require 6,174,857,216 bytes. Capacity 1025
KV requires 58,777,600 bytes; adding 128 MiB framework/application allowances gives
a lower bound 6,367,852,544 bytes against the 6,442,450,944-byte ceiling. The
remaining 74,598,400 bytes must cover scratch, logits, history and native overhead.
Reference scoring therefore requires explicit bounded buffers and successful
native qualification. A predicted fit is not admission or device evidence.

The [preselection record](../../benchmarks/results/2026-10-13-m5-larger-preselection-v1.json)
binds config/header/API metadata identities and calculations. Full download,
exact hash, source/GPU geometry, memory, unchanged 10,000-target and 100-task
quality comparisons, performance and sustained measurements remain pending.
Its future outcome cannot rehabilitate the rejected 0.5B Q4 configuration.
