# Independent 9B storage fixture

This 211-byte artificial stream has a nine-byte unaligned prefix, two distinct
original affine4/BF16 matrices (embedding2x64, head3x64), a three-value vector
and a two-channel/four-tap convolution. Physical order differs from logical
recipe order. The Python generator packs original integer cells and records
independent expected rows and exact FP32 small-value bits, including signed zero.

Only private test constructors can supply its deliberately tiny cold recipe.
It is rejected by every pinned public loader and grants no real-model execution,
numerical, memory-fit, context, quality or managed-route qualification.

[Generator](../../../tools/reference/qwen35_9b_storage_fixture.py).
