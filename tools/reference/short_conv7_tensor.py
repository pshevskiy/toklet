"""Independent CPU tensor convolution correspondence for frozen seven-tap fixtures.

Uses admitted local Torch code, not a model's remote code. This checks only raw
FP32-input convolution/layout; BF16 graph execution/adoption remain unqualified.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import struct


def need(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path, limit=256*1024):
    with path.open('rb') as stream:
        raw = stream.read(limit+1)
    need(len(raw) <= limit, 'Bounded seven-tap input required')
    return json.loads(raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['fixture', 'runtime', 'contract', 'fresh_report']:
        parser.add_argument('--'+name.replace('_','-'), type=Path, required=True)
    parser.add_argument('--fixture-sha256', required=True)
    parser.add_argument('--runtime-sha256', required=True)
    args = parser.parse_args()
    need(not args.fresh_report.exists() and args.fresh_report.parent.is_dir(), 'Fresh tensor report required')
    need(digest(args.fixture) == args.fixture_sha256 and digest(args.runtime) == args.runtime_sha256,
         'Frozen fixture/runtime identity differs')
    policy = read(args.contract)
    need(policy['before_tensor_outcomes'] is True and policy['device']=='cpu'
         and policy['dtype']=='float64' and policy['numerical_gate']==dict(abs=1e-5,rel=1e-4),
         'Frozen tensor comparison policy differs')
    runtime = read(args.runtime, 8*1024**2)
    for name, version in runtime['versions'].items():
        need(importlib.metadata.version(name) == version, 'Admitted distribution version differs')
    for path, pin in runtime['files'].items():
        need(digest(Path(path)) == pin, 'Admitted tensor runtime input changed')
    need(runtime['versions']['torch'] == policy['torch_version']
         and runtime['versions']['numpy'] == policy['numpy_version'], 'Pinned numerical runtime version differs')
    import torch
    import numpy
    need(str(Path(torch.__file__).resolve()) == runtime['imports']['torch']
         and str(Path(numpy.__file__).resolve()) == runtime['imports']['numpy'], 'Imported tensor runtime path differs')
    torch.set_num_threads(1)
    fixture = read(args.fixture)
    need(fixture['format']=='toklet-short-conv7-exact-v1' and fixture['kernel']==7
         and fixture['numerical_gate']==policy['numerical_gate'], 'Frozen fixture schema differs')
    records=[]
    total=failures=bit_differences=0
    for row in fixture['cases']:
        c,t=row['channels'],row['tokens']
        decode=lambda values:[struct.unpack('<f',struct.pack('<I',n))[0] for n in values]
        data=decode(row['initial_history_bits']+row['input_bits'])
        joined=torch.tensor(data,dtype=torch.float64,device='cpu').reshape(6+t,c)
        weight=torch.tensor(decode(row['weight_bits']),dtype=torch.float64,device='cpu').reshape(c,1,7)
        if t:
            out=torch.nn.functional.conv1d(joined.t().unsqueeze(0),weight,groups=c)
            actual=out.squeeze(0).t().contiguous().to(torch.float32).reshape(-1).tolist()
        else:
            actual=[]
        expected=decode(row['output_bits'])
        need(len(actual)==len(expected)==t*c,'Tensor output extent differs')
        maximum=0.
        local=0
        observed_bits=[]
        for got,want,pin in zip(actual,expected,row['output_bits']):
            need(math.isfinite(got) and math.isfinite(want),'Nonfinite tensor result')
            error=abs(got-want);maximum=max(maximum,error)
            bad=error>1e-5+1e-4*abs(want)
            local+=bad;failures+=bad;total+=1
            observed=struct.unpack('<I',struct.pack('<f',got))[0];observed_bits.append(observed)
            bit_differences+=observed!=pin
        final=joined[t:t+6].to(torch.float32).reshape(-1).tolist()
        final_bits=[struct.unpack('<I',struct.pack('<f',value))[0] for value in final]
        need(final_bits==row['final_history_bits'],'Independent tensor final raw history differs')
        records.append(dict(name=row['name'],values=len(actual),failures=local,
            maximum_absolute_error=maximum,output_bits=observed_bits,final_history_bits=final_bits))
    for path,pin in runtime['files'].items():
        need(digest(Path(path))==pin,'Tensor runtime changed during reference')
    need(digest(args.fixture)==args.fixture_sha256,'Fixture changed during tensor comparison')
    result=dict(format='toklet-short-conv7-tensor-reference-v1',passed=failures==0,
        fixture_sha256=args.fixture_sha256,runtime_sha256=args.runtime_sha256,
        contract_sha256=digest(args.contract),producer_sha256=digest(Path(__file__)),
        torch_version=torch.__version__,numpy_version=numpy.__version__,device='cpu',dtype='float64',
        values=total,failures=failures,fp32_bit_differences=bit_differences,cases=records,
        raw_history_matches=True,bf16_activation_profile_qualified=False,
        full_model_execution_qualified=False,quality_adoption_qualified=False)
    with args.fresh_report.open('x') as stream:
        stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'}))
    need(result['passed'],'Independent seven-tap tensor comparison failed')


if __name__=='__main__':
    main()
