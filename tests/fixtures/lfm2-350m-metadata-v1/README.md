# LFM2-350M cold metadata fixture

Exact original config/index/header bytes from public mlx-community/LFM2-350M-4bit revision18dc72abf3b2337f9123cfd6eeeb58dfa7947066. Only the first8-byte length and36,740-byte JSON header were acquired from the weight container during research; no tensor payload is present here. The advertised whole-file LFS digest has not been verified. These files support download-independent structural inspection only.

The current pinned-loader profile establishes direct stored BF16 RMS gains, original [1024,3,1] convolution orientation,93affine/45norm/10conv owners and30borrowed fused B/C/x row spans. No coefficient conversion or numerical operation occurs in this task. The historical converter runtime remains unknown.

Future NativeHalf uses direct BF16→FP32 gains and the existing one-store Neumaier RMS contract, not Gemma+1 or MLX intermediate-store fidelity. Existing RMS plan admits epsilon1e-6 only: epsilon1e-5 must be independently qualified at BOTHhead64 andhidden1024, along with new GQA16/KV8/D64 geometry. No runtime/text/app/backend eligibility follows from this metadata fixture.
