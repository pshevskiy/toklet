"""Cold full-sequence tiny long oracle; no incremental cache or Rust execution."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import signal
import sys
import bf16_source_reference as runtime
import check_gemma4_long_witness as contract
import gemma4_long_witness as graph

ROOT = Path(__file__).resolve().parents[2]


def extras():
    return [Path(__file__).resolve(), Path(graph.__file__).resolve(),
            Path(contract.__file__).resolve(), ROOT/'tools/reference/test_gemma4_text_math.py',
            Path(graph.frozen.__file__).resolve()]


def sources(): return runtime.sources(extras())


def run(output, specification, admission, endpoint):
    record = dict(format='toklet-gemma-long-witness-capture-v1', passed=False,
                  primary_error=None, endpoint_errors=[], arrays=[], files={},
                  runtime_admission=admission, runtime_endpoint=None,
                  memory_ledger_mib=contract.LEDGER_MIB, tokens=specification['tokens'],
                  **{key:False for key in contract.SCOPE})
    output.mkdir(); resources={}
    try:
        import test_gemma4_text_math as fixture
        runtime.safe.same(hashlib.sha256(Path(fixture.graph.__file__).read_bytes()).hexdigest(), graph.ORIGINAL_SOURCE_SHA256)
        runtime.safe.same(graph.LEDGER_MIB, contract.LEDGER_MIB)
        weights, _, _ = fixture.fixture()
        for sequence, ids in specification['tokens'].items():
            file=(output/(sequence+'.bin')).open('xb');resources[sequence]=file
            for name, staged in [('math',False),('staged_fp32',True)]:
                def write(state_name, value):
                    wanted=specification['captures'][len(record['arrays'])]
                    body=value.tobytes(order='C')
                    actual=dict(sequence=sequence,contract=name,name=state_name,
                                shape=list(value.shape),dtype=value.dtype.str,offset=file.tell(),
                                bytes=len(body),sha256=hashlib.sha256(body).hexdigest(),captured=False)
                    record['arrays'].append(actual)
                    runtime.safe.require(file.tell()+len(body)<=specification['file_bytes'],'long capture bound exceeded')
                    runtime.safe.same(file.write(body),len(body));file.flush();actual['captured']=True
                    runtime.safe.same({key:actual[key]for key in wanted},wanted)
                graph.evaluate(weights,ids,staged,sink=write)
            runtime.safe.same(file.tell(),specification['file_bytes']);file.close()
        runtime.safe.same(len(record['arrays']),360);record['passed']=True
    except BaseException as error:
        record['primary_error']=type(error).__name__+': '+str(error)
    finally:
        old={sig:signal.signal(sig,signal.SIG_IGN)for sig in(signal.SIGINT,signal.SIGTERM)}
        try:
            for name,file in resources.items():
                try:file.close()
                except BaseException as error:record['endpoint_errors'].append(name+' close: '+str(error))
            try:record['runtime_endpoint']=endpoint()
            except BaseException as error:record['endpoint_errors'].append('runtime: '+type(error).__name__+': '+str(error))
            for sequence in specification['tokens']:
                path=output/(sequence+'.bin')
                if os.path.lexists(path):
                    try:record['files'][sequence]=contract.identity(path,specification['file_bytes'])
                    except BaseException as error:record['endpoint_errors'].append('file '+sequence+': '+str(error))
            record['passed']=record['passed']and record['primary_error']is None and not record['endpoint_errors']
            with(output/'observation.json').open('xb')as file:file.write(contract.encoded(record))
        finally:
            for sig,handler in old.items():signal.signal(sig,handler)
    print(json.dumps({key:record[key]for key in('passed','primary_error','endpoint_errors','files')}))
    return 0 if record['passed']else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--plan-sha',required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--runtime-manifest',type=Path,required=True);parser.add_argument('--runtime-sha',required=True)
    args=parser.parse_args();specification=contract.load_plan(args.plan,args.plan_sha)
    selected=runtime.runtime.runtime_paths(sources())
    admission=runtime.runtime.admit_runtime(args.runtime_manifest,args.runtime_sha,sources())
    manifest=runtime.safe.document(runtime.safe.read(args.runtime_manifest,4*1024**2))
    def stop(number,_):raise InterruptedError('Received signal '+str(number))
    old={sig:signal.signal(sig,stop)for sig in(signal.SIGINT,signal.SIGTERM)}
    try:code=run(args.output,specification,admission,lambda:runtime.endpoint(manifest,selected,extras()))
    finally:
        for sig,handler in old.items():signal.signal(sig,handler)
    sys.exit(code)
