# Independent untied-output hybrid fixture

The original four-layer tied fixture weights are retained as an exact prefix.
An independent32x64affine4/G64/BF16 output matrix is appended, with different
values from the same-vocabulary input table. The original independent graph and
its idealFP64/stagedFP32 arithmetic stay unchanged. A reference-only final-matrix
selector uses the new head while direct token lookup uses the original embedding.

Every intermediate, committed state and complete logit row is captured for the
original one/context17/offset/future-token cases. These toy fixtures establish no
real9B, GPU, quality, context, memory-fit or managed-route qualification.

[Producer](../../../tools/reference/qwen35_untied_graph.py).
