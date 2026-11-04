"""Explicit real-E2B context reference; select hidden rows before the tied head."""
from pathlib import Path
import argparse
import hashlib
import json
import struct
import time
import gemma4_text_reference as source_graph
import gemma4_e2b_context_math as graph

ROOT=Path(__file__).resolve().parents[2]
CONTRACT=ROOT/'benchmarks/gemma-e2b-context-v1'


def inputs(phase):
    manifest=json.loads((CONTRACT/'manifest.json').read_text())
    raw=(CONTRACT/'tokens.u32le').read_bytes();parent=manifest['parent_tokens'];body=(ROOT/parent['path']).read_bytes()
    if (len(raw)!=2048 or hashlib.sha256(raw).hexdigest()!=manifest['tokens']['sha256']
            or raw!=body[:2048]or hashlib.sha256(body).hexdigest()!=parent['sha256']):raise ValueError('Frozen corpus prefix differs')
    spec=manifest['phases'][phase]
    return list(struct.unpack('<512I',raw))[:spec['tokens']],spec['positions']


def collect(output,phase,admission,guard):
    import numpy as np
    ids,positions=inputs(phase);output=output.resolve()
    if not output.is_relative_to(ROOT/'artifacts'):raise ValueError('Fresh artifact output required')
    output.mkdir();arrays=[];planned=len(positions)*262144*12
    record=dict(completed=False,phase=phase,tokens=ids,selected_positions=positions,planned_rows_bytes=planned,
        source_pins=source_graph.source.PINS,runtime_admission=admission,ledger_mib=graph.LEDGER_MIB,
        frozen_math_source_sha256=graph.ORIGINAL_SOURCE_SHA256,contract_times={},prior_precision_failures_waived=False)
    try:
        if hashlib.sha256(Path(graph.frozen.__file__).read_bytes()).hexdigest()!=graph.ORIGINAL_SOURCE_SHA256:raise ValueError('Historical mathematical source differs')
        if sum(graph.LEDGER_MIB.values())*1024**2>6*1024**3:raise ValueError('Reference allowance exceeds process policy')
        for name,sha in source_graph.source.PINS.items():
            if source_graph.source.sha(source_graph.source.MODEL/name)!=sha:raise ValueError('Original source identity differs')
        with(source_graph.source.MODEL/'model.safetensors').open('rb')as stream,(output/'rows.bin').open('xb')as binary:
            weights=source_graph.weights(stream)
            for contract,staged in [('math_fp64',False),('staged_fp32',True)]:
                start=time.perf_counter_ns();seen=[]
                print(json.dumps(dict(event='contract-start',phase=phase,contract=contract,tokens=len(ids))),flush=True)
                def capture(name,value):
                    if name!='logits':return
                    if value.shape!=(1,len(positions),262144)or not np.isfinite(value).all():raise ValueError('Selected reference logits differ')
                    for row,position in enumerate(positions):
                        raw=np.asarray(value[0,row],dtype='<f4'if staged else'<f8').tobytes()
                        arrays.append(dict(contract=contract,position=position,offset=binary.tell(),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
                        binary.write(raw);seen.append(position)
                graph.evaluate(weights,ids,staged,selected_positions=positions,sink=capture)
                if seen!=positions:raise ValueError('Missing selected reference rows')
                record['contract_times'][contract]=(time.perf_counter_ns()-start)/1e9
                print(json.dumps(dict(event='contract-complete',phase=phase,contract=contract,seconds=record['contract_times'][contract])),flush=True)
            if binary.tell()!=planned:raise ValueError('Selected output size differs')
        record.update(completed=True,arrays=arrays,rows_sha256=source_graph.source.sha(output/'rows.bin'))
    except BaseException as error:record['failure']=dict(type=type(error).__name__,message=str(error)[:2048])
    finally:
        try:record['runtime_endpoint']=guard()
        except BaseException as error:record.update(completed=False,endpoint_failure=str(error)[:2048])
        (output/'result.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--phase',choices=['compat32','pilot64','full512'],required=True)
    parser.add_argument('--runtime-manifest',type=Path,required=True);parser.add_argument('--expected-runtime-sha256',required=True)
    parser.add_argument('--native-manifest',type=Path,required=True);parser.add_argument('--expected-native-sha256',required=True)
    args=parser.parse_args()
    admission,guard=source_graph.source.runtime_context(args.runtime_manifest,args.expected_runtime_sha256,args.native_manifest,args.expected_native_sha256,
        extra_sources=[Path(__file__).resolve(),Path(graph.__file__).resolve(),Path(source_graph.__file__).resolve(),Path(source_graph.graph.__file__).resolve()])
    result=collect(args.output,args.phase,admission,guard);print(json.dumps(dict(completed=result['completed'],failure=result.get('failure'))))
    raise SystemExit(0 if result['completed']else 1)
