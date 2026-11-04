"""Independent original packed Gemma32 selected-row mathematical references."""
from pathlib import Path
import argparse
import hashlib
import json
import struct
import sys
import gemma4_text_reference as reference

ROOT=Path(__file__).resolve().parents[2]
CONTRACT=ROOT/'benchmarks/gemma-hybrid32-v1'
POSITIONS=[0,15,30,31]


def inputs():
    manifest=json.loads((CONTRACT/'manifest.json').read_text())
    raw=(CONTRACT/'tokens.u32le').read_bytes()
    if len(raw)!=128 or hashlib.sha256(raw).hexdigest()!=manifest['tokens']['sha256']:
        raise ValueError('Frozen32-token payload differs')
    parent=manifest['parent_tokens'];body=(ROOT/parent['path']).read_bytes()
    if hashlib.sha256(body).hexdigest()!=parent['sha256']or raw!=body[:128]:
        raise ValueError('Frozen corpus prefix differs')
    return list(struct.unpack('<32I',raw))


def collect(output,admission,guard):
    import numpy as np
    ids=inputs();output=output.resolve()
    if not output.is_relative_to(ROOT/'artifacts'):raise ValueError('Fresh artifact output required')
    output.mkdir();arrays=[];record=dict(completed=False,tokens=ids,selected_positions=POSITIONS,
        gate=dict(atol=1e-5,rtol=1e-4),runtime_admission=admission,source_pins=reference.source.PINS,
        prior_precision_failures_waived=False)
    try:
        for name,sha in reference.source.PINS.items():
            if reference.source.sha(reference.source.MODEL/name)!=sha:raise ValueError('Original source identity differs')
        with(reference.source.MODEL/'model.safetensors').open('rb')as stream,(output/'rows.bin').open('xb')as binary:
            weights=reference.weights(stream)
            for contract,staged in [('math_fp64',False),('staged_fp32',True)]:
                seen=[]
                def capture(name,value):
                    if name!='logits':return
                    if value.shape!=(1,32,262144)or not np.isfinite(value).all():raise ValueError('Full reference logits differ')
                    for position in POSITIONS:
                        raw=np.asarray(value[0,position],dtype='<f4'if staged else'<f8').tobytes()
                        arrays.append(dict(contract=contract,position=position,offset=binary.tell(),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
                        binary.write(raw);seen.append(position)
                reference.graph.evaluate(weights,ids,staged,sink=capture)
                if seen!=POSITIONS:raise ValueError('Missing selected reference rows')
        record.update(completed=True,arrays=arrays,rows_sha256=reference.source.sha(output/'rows.bin'))
    except BaseException as error:record['failure']=dict(type=type(error).__name__,message=str(error)[:2048])
    finally:
        try:record['runtime_endpoint']=guard()
        except BaseException as error:record.update(completed=False,endpoint_failure=str(error)[:2048])
        (output/'result.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--runtime-manifest',type=Path,required=True);parser.add_argument('--expected-runtime-sha256',required=True)
    parser.add_argument('--native-manifest',type=Path,required=True);parser.add_argument('--expected-native-sha256',required=True)
    args=parser.parse_args()
    admission,guard=reference.source.runtime_context(args.runtime_manifest,args.expected_runtime_sha256,args.native_manifest,args.expected_native_sha256,
        extra_sources=[Path(__file__).resolve(),Path(reference.__file__).resolve(),Path(reference.graph.__file__).resolve()])
    result=collect(args.output,admission,guard);print(json.dumps(dict(completed=result['completed'],failure=result.get('failure'))))
    raise SystemExit(0 if result['completed']else 1)
