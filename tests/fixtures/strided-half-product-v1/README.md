# Strided Half product reference

Five independent source cases for B*x and C*z, borrowing one physical BCx
parent. Width256/1024, one through32 active rows, nonzero view/output offsets,
padding and seven guard words. MAIN uses an exact FP32 product then RNE Half;
WIDE uses FP64 then Half. Both use identical original stored Half operands.

Eight controls, exact four-file export/replay and an independent integer-dyadic
RNE audit pass. The source retains67,072 values per profile,17 descriptors and
738,248 raw bytes. Ten exact store witnesses and eight invalid controls are
additional requirements. Read/read aliasing is allowed; writable aliases reject.

This is a source fixture. It does not qualify a native kernel, resident graph,
real model, application, memory peak, speed or mobile device.
