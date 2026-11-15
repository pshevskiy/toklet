# Independent Huihui4B conversion mechanism controls

These literals follow the pinned MLX0.31.2/MLX-LM0.31.3 primary source notes
(SHA7ab43fe40e7ec90aa96df8400708f7a664ac3587a6c149e8d22a4a66cd33ca56),
not values computed by the new producer. The two canonical CPU/Metal profiles
are fixed; no additional rounding/search variant is authorized by these tests.

The BF16 norm exhaustive reference rounds an exact signed dyadic sum using
integers; it does not reuse the producer's FP32bit-round implementation.
Group fixtures include separate unrounded code generation and BF16metadata
store,rounding/clamp,negative-only max0,epsilon,q0zero andlow-nibble packing.
All tests are ordinary CPU work,with tiny byte readers and descriptor controls.
They never read real model payloads,create sparse9GBfiles,download,run a GPU,
or establish the actual426/924array source correspondence or historical origin.

A complete real case requires all65,699,840groups and177unquantized arrays
under one canonical profile,independent fullfile authentication and root's
explicit bounded driver run. Any tiny positive helper is not that admission.
