"""Explicit admitted CPU/GPU literal-control run, before any parent comparison."""
import argparse
import json
from pathlib import Path
import sys
import time

import minicpm5_quant_runtime as runtime
import minicpm5_quantize as quant


def sources():
    folder = Path(__file__).resolve().parent
    return [folder/name for name in ['minicpm5_quant_probe.py', 'minicpm5_quantize.py',
            'test_minicpm5_quantize.py', 'minicpm5_quant_runtime.py']]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['environment_root', 'runtime_manifest', 'runtime_sha256',
                 'controls_file', 'controls_sha256', 'fresh_output']:
        parser.add_argument('--'+name.replace('_','-'), required=True)
    args = parser.parse_args()
    output = Path(args.fresh_output);output.mkdir()
    runtime.bind_environment(args.environment_root)
    raw = runtime.common.captured(Path(args.controls_file), 2*1024**2,
                                  runtime.common.sha(args.controls_sha256))
    controls = json.loads(raw, object_pairs_hook=runtime.common.json_unique,
                          parse_constant=runtime.common.bad_json_number)
    quant.need(controls == dict(format='toklet-minicpm5-quantizer-literals-v1',
                               cases=quant.controls()), 'Original literal controls differ')
    admitted, endpoint = runtime.admit(Path(args.runtime_manifest), args.runtime_sha256, sources())
    import numpy as np
    import mlx.core as mx
    mx.set_cache_limit(0)
    rows = []
    for case in controls['cases']:
        for device in ['cpu', 'gpu']:
            started = time.monotonic_ns()
            actual = quant.quantize(bytes.fromhex(case['input_bf16']), case['rows'],
                                    case['columns'], device, np, mx)
            completed = time.monotonic_ns()
            expected = {key:bytes.fromhex(value) for key,value in case[device].items()}
            compared = quant.compare(actual, expected)
            row = dict(name=case['name'], device=device,
                       passed=all(v['mismatching_bytes']==0 for v in compared.values()),
                       completed_quantize_and_readback_ns=completed-started,
                       buffers={key:value.hex() for key,value in actual.items()}, comparison=compared)
            rows.append(row)
            with (output/'records.jsonl').open('a') as stream:
                stream.write(json.dumps(row, separators=(',',':'))+'\n')
    final = endpoint()
    result = dict(format='toklet-minicpm5-quantizer-controls-v1',
                  passed=len(rows)==12 and all(row['passed'] for row in rows),
                  admitted_runtime=admitted, endpoint_origins=final,
                  controls_sha256=args.controls_sha256, cases=len(controls['cases']),
                  cpu_cases=6, gpu_cases=6, device=mx.metal.device_info(),
                  model_or_parent_weights_consumed=False, conversion_lineage_qualified=False,
                  performance_or_model_quality_adoption_qualified=False)
    (output/'observation.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key:value for key,value in result.items()
                      if key not in ['admitted_runtime','endpoint_origins','device']}))
    return 0 if result['passed'] else 1


if __name__=='__main__':sys.exit(main())
