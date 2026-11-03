"""Frozen incremental original-reference conformance and short decode probe."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import llama3_original_long as long  # Pin threads before NumPy import.
import llama3_original_session as cached


def contract(path, expected_sha):
    raw=Path(path).read_bytes()
    if len(raw)>65536 or hashlib.sha256(raw).hexdigest()!=expected_sha:
        raise ValueError('Frozen incremental contract differs')
    value=json.loads(raw,object_pairs_hook=long.inputs.parent.unique)
    if (value['format']!='toklet-llama3-original-incremental-contract-v1' or value['capacity']!=566
            or value['cases']!=[dict(name='short',prefill_tokens=17,continuation_end=21),dict(name='boundary512',prefill_tokens=510,continuation_end=513),dict(name='capacity566',prefill_tokens=565,continuation_end=566)]
            or value['tile']!=256 or value['model_executed'] is not False
            or value['consistency']!=dict(full_row_atol=1e-3,full_row_rtol=1e-4,target_nll_absolute=1e-4,exact_lowest_id_greedy=True,record_top_two_margin=True)
            or value['decode_probe']!=dict(warmups=3,repetitions=5,prefill_tokens=17,continuation_end=21,publish_throughput_claim=False)):
        raise ValueError('Frozen incremental policy differs')
    return value


def kv_sha(session):
    digest=hashlib.sha256()
    digest.update(memoryview(session.k).cast('B'));digest.update(memoryview(session.v).cast('B'))
    return digest.hexdigest()


def produce(model, contract_file, expected_sha, output):
    output=Path(output);output.mkdir()
    report=dict(format='toklet-llama3-original-incremental-source-v1',passed=False,
                contract_sha256=expected_sha,full_quality_qualified=False,cases=[])
    reader=None
    try:
        if importlib.metadata.version('numpy')!='2.3.2' or sys.byteorder!='little':
            raise ValueError('Reference runtime differs')
        policy=contract(contract_file,expected_sha)
        quality=long.graph.ROOT/'benchmarks/llama3-original-quality-v1'
        manifest,ids=long.inputs.load(quality,policy['quality_manifest_sha256'])
        if manifest['tokens']['sha256']!=policy['tokens_sha256']:raise ValueError('Frozen token stream differs')
        paths=[Path(__file__),Path(cached.__file__),Path(long.graph.__file__),Path(long.__file__),
               Path(long.short.__file__),Path(long.inputs.__file__),Path(long.inputs.parent.__file__),
               long.graph.ROOT/'tools/reference/llama3_loss1024.py',long.graph.ROOT/'tools/bench/acquire_llama3_original.py']
        report['source_sha256']={str(p):long.short.sha(p)for p in paths}
        provider={str(p):long.short.sha(p)for p in sorted(Path(long.np.__file__).resolve().parent.rglob('*'))
                  if p.is_file()and p.suffix in ['.py','.so','.dylib']and '__pycache__'not in p.parts}
        pins=output/'runtime-pins.json';pins.write_text(json.dumps(provider,indent=2)+'\n')
        report['runtime_pins']=dict(file=pins.name,sha256=long.short.sha(pins),entries=len(provider))
        report['runtime']=dict(python=sys.version,numpy=long.np.__version__,threads={n:long.os.environ[n]for n in ['VECLIB_MAXIMUM_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']})
        report['original_acquisition_sha256']=long.short.sha(Path(model)/'acquisition.json')
        long.resource_gate(output);reader=long.graph.Reader(model);session=cached.Session(reader,566)
        report['kv_bytes']=session.kv_bytes
        if session.kv_bytes!=37093376:raise ValueError('Reference KV accounting differs')
        for case in policy['cases']:
            session.reset();first=case['prefill_tokens']
            for count in range(first,case['continuation_end']+1):
                print('reference prefix',count,flush=True)
                actual=session.forward(ids[:count] if count==first else [ids[count-1]])
                reference=long.graph.quality_forward(reader,ids[:count],[count-1])[0]
                row=dict(prefix_tokens=count,logit_row=count-1,target_position=count,target_id=ids[count],
                         case=case['name'],consistency=long.short.comparison(actual,reference,ids[count]),arrays={})
                row['greedy_matches']=cached.greedy(actual)==cached.greedy(reference)
                if session.position!=count or session.history!=ids[:count]:raise ValueError('Reference committed state differs')
                for label,values in [('batch',actual),('prefix',reference)]:
                    raw=values.astype('<f4',copy=False).tobytes();path=output/f'prefix-{count}-{label}.f32le';path.write_bytes(raw)
                    row['arrays'][label]=dict(file=path.name,bytes=len(raw),sha256=long.short.sha(path))
                report['cases'].append(row);long.resource_gate(output)
                if not row['consistency']['passed'] or not row['greedy_matches']:
                    raise ValueError('Frozen incremental row/greedy conformance failed')
        before=(session.position,list(session.history),session.logits.copy(),kv_sha(session))
        rejected=False
        try:session.forward([ids[566]])
        except ValueError:rejected=True
        if (not rejected or (session.position,session.history)!=before[:2]
                or not long.np.array_equal(session.logits,before[2])or kv_sha(session)!=before[3]):
            raise ValueError('Reference overflow changed valid prefix')
        report['overflow_preserves_prefix']=True
        session.reset();reused=session.forward(ids[:17]);fresh=cached.Session(reader,566).forward(ids[:17])
        if not long.np.array_equal(reused,fresh):raise ValueError('Reset leaked stale KV')
        report['reset_matches_fresh']=True
        if cached.greedy(long.np.zeros(128256,dtype=long.np.float32))!=0:raise ValueError('Lowest-ID tie differs')
        report['all_tied_lowest_id']=0
        samples=[]
        for repetition in range(8):
            session.reset();session.forward(ids[:17])
            for count in range(18,22):
                started=time.perf_counter_ns();session.forward([ids[count-1]]);elapsed=time.perf_counter_ns()-started
                if repetition>=3:samples.append(dict(repetition=repetition-3,context_before=count-1,elapsed_ns=elapsed))
            long.resource_gate(output)
        report['decode_probe']=dict(warmups=3,repetitions=5,samples=samples,weight_cache=False,head_tile=256,publish_throughput_claim=False)
        reader.close();reader=None;gc.collect()
        if any(long.short.sha(Path(p))!=v for p,v in provider.items()):raise ValueError('Runtime changed')
        if any(long.short.sha(Path(p))!=v for p,v in report['source_sha256'].items()):raise ValueError('Source changed')
        contract(contract_file,expected_sha);long.inputs.load(quality,policy['quality_manifest_sha256'])
        report.update(passed=True,peak_rss_bytes=long.resource_gate(output))
    except BaseException as error:
        report.update(error=repr(error),peak_rss_bytes=long.resource.getrusage(long.resource.RUSAGE_SELF).ru_maxrss)
    finally:
        if reader is not None:
            try:reader.close()
            except BaseException as error:report['close_error']=repr(error)
        (output/'source.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model',type=Path);parser.add_argument('contract',type=Path)
    parser.add_argument('contract_sha');parser.add_argument('new_output',type=Path)
    args=parser.parse_args();value=produce(args.model,args.contract,args.contract_sha,args.new_output)
    print(json.dumps(value,indent=2));raise SystemExit(0 if value['passed']else 1)
