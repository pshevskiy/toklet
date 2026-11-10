"""Independent canonical BF16 norm controls for epsilon and signed-zero cases."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import lfm3_bf16_fixture as source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['contract', 'runtime', 'fresh_output']:
        parser.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    args = parser.parse_args()
    source.need(not args.fresh_output.exists() and args.fresh_output.parent.is_dir(),
                'Fresh corner output required')
    source.need(source.digest(args.contract) == source.CONTRACT_SHA
                and source.digest(args.runtime) == source.RUNTIME_SHA, 'Frozen corner inputs differ')
    runtime = json.loads(args.runtime.read_text())
    source.verify_runtime(runtime)
    import numpy as np
    import torch
    torch.set_num_threads(1)
    cases = []
    for width in [128, 2048]:
        for word in [0, 0x8000, 1, 0x8001, 0x3a80, 0x3c00, 0x3f80, None]:
            input_words = [word]*width if word is not None else [0x4180] + [0x8000]*(width-1)
            gain_words = [0x3fc0 if i % 2 == 0 else 0xbf00 for i in range(width)]
            tensor = lambda values: torch.from_numpy(np.array(values, dtype=np.uint16)).view(torch.bfloat16)
            x, gain = tensor(input_words), tensor(gain_words)
            xf = x.float()
            variance = xf.pow(2).mean()
            inverse = torch.rsqrt(variance + 1e-5)
            normalized = xf * inverse
            middle = normalized.to(torch.bfloat16)
            result = middle * gain
            blob = result.view(torch.int16).numpy().astype('<u2').tobytes()
            cases.append(dict(width=width, input_word=word, impulse_word=0x4180,
                background_word=0x8000, even_gain_word=0x3fc0, odd_gain_word=0xbf00,
                variance_fp32_bits=struct.unpack('<I', struct.pack('<f', variance.item()))[0],
                inverse_fp32_bits=struct.unpack('<I', struct.pack('<f', inverse.item()))[0],
                output_sha256=hashlib.sha256(blob).hexdigest(),
                first_output_bits=list(struct.unpack('<8H', blob[:16]))))
    source.verify_runtime(runtime)
    result = dict(format='toklet-lfm3-bf16-norm-corners-v1', fixture_completed=True,
        contract_sha256=source.CONTRACT_SHA, runtime_sha256=source.RUNTIME_SHA,
        producer_sha256=source.digest(Path(__file__)), source_helper_sha256=source.digest(Path(source.__file__)),
        cases=cases, device='cpu', torch_version=torch.__version__, model_execution_qualified=False)
    with args.fresh_output.open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(fixture_completed=True, cases=len(cases),
                         fixture_sha256=source.digest(args.fresh_output))))


if __name__ == '__main__':
    main()
