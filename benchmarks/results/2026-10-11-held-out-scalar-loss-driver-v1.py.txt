"""Frozen held-out CPU loss, with independent short probes before full scoring."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import struct
import subprocess
import threading
import time

import cpu
import mlx_baseline as common
import prepare

ROOT=prepare.ROOT
POLICY="independent_blocks_of_up_to_1025_tokens_one_token_overlap_score_every_target_once"
PROBE_SHA="691429f672aa89875e06475f48b5b691343f03c08e47fd204fa60fec2eef20d1"
FULL_SHA="bdd368d21dfae574197e3b64423b75d4f1b2b37c083c6172c429e36bb4dd69c7"
LIMIT=6*1024**3
FORMAT="toklet-held-out-loss-v1"


def stable_loss(logits,target):
    if not logits or not 0<=target<len(logits) or not all(math.isfinite(v) for v in logits):
        raise ValueError("Invalid independent loss distribution")
    maximum=max(logits)
    return (maximum-logits[target])+math.log(math.fsum(math.exp(v-maximum) for v in logits))


def check_probe(losses,reference):
    if len(losses)!=len(reference):raise ValueError('Short-probe NLL coverage differs')
    delta=[row['negative_log_likelihood']-value for row,value in zip(losses,reference)]
    if any(not math.isfinite(value) or abs(value)>1e-3 for value in delta):
        raise ValueError('Unchanged short-probe NLL gate failed')
    return delta


def validate_ready(ready,tokens,projection,corpus_sha):
    count=len(tokens)-1
    expected={'event':'ready','format':'toklet-held-out-loss-events-v1','cpu_projection':projection,
        'target_tokens':count,'corpus_sha256':corpus_sha,'context_policy':POLICY,'block_targets':1024,
        'context_capacity':1025,'prefill_chunk_size':1,'weight_dtype':'FP32','kv_dtype':'FP32'}
    if any(ready.get(key)!=value for key,value in expected.items()):raise ValueError('Loss worker admission/mode/policy differs')
    memory_fields=('logical_weights_bytes','allocated_kv_bytes','scratch_bytes','history_bytes','logits_bytes','metadata_bytes','metadata_allowance_bytes','application_allowance_bytes')
    if (any(type(ready.get(key)) is not int or ready[key]<0 for key in memory_fields)
            or ready['memory_admitted_bytes']!=sum(ready[key] for key in memory_fields)
            or not 0<ready['memory_admitted_bytes']<=LIMIT):
        raise ValueError('Loss worker memory evidence differs')


def validate_run(run,tokens,projection,corpus_sha):
    validate_ready(run['ready'],tokens,projection,corpus_sha);count=len(tokens)-1
    if not 0<run['physical_process_charge_peak_bytes']<=LIMIT:raise ValueError('Loss worker physical peak differs')
    losses=run['losses'];blocks=run['blocks'];done=run['done']
    if len(losses)!=count or len(blocks)!=(count+1023)//1024:raise ValueError('Loss target/block coverage differs')
    for index,row in enumerate(losses):
        integers={'target_index':index,'block_index':index//1024,'position':index%1024,
                  'prefix_tokens':index%1024+1,'target_token':tokens[index+1]}
        if (row.get('event')!='loss' or any(type(row.get(key)) is not int or row[key]!=value for key,value in integers.items())
                or not math.isfinite(row['negative_log_likelihood']) or row['negative_log_likelihood']<0
                or type(row['target_logit_f32_bits']) is not int or not 0<=row['target_logit_f32_bits']<=0xffffffff
                or not isinstance(row['logits_sha256'],str) or len(row['logits_sha256'])!=64
                or any(c not in '0123456789abcdef' for c in row['logits_sha256'])):
            raise ValueError('Loss label/prefix/numerical evidence differs')
        value=struct.unpack('<f',struct.pack('<I',row['target_logit_f32_bits']))[0]
        if not math.isfinite(value):raise ValueError('Loss target logit is nonfinite')
    for index,block in enumerate(blocks):
        start=index*1024;end=min(start+1024,count);size=end-start
        expected={'event':'block_done','block_index':index,'target_start':start,'target_end':end,
                  'scored_targets':size,'evaluated_tokens':size+1,'pending_tokens':0}
        total=math.fsum(row['negative_log_likelihood'] for row in losses[start:end])
        if (any(block.get(key)!=value for key,value in expected.items())
                or not math.isclose(block['total_negative_log_likelihood'],total,rel_tol=1e-13,abs_tol=1e-10)):
            raise ValueError('Loss block commitment/aggregation differs')
    total=math.fsum(row['negative_log_likelihood'] for row in losses)
    final_size=count-(len(blocks)-1)*1024
    if (done.get('event')!='done' or done['scored_targets']!=count or done['evaluated_tokens']!=final_size+1
            or done['pending_tokens']!=0 or not math.isclose(done['total_negative_log_likelihood'],total,rel_tol=1e-13,abs_tol=1e-10)
            or done['mean_negative_log_likelihood']!=done['total_negative_log_likelihood']/count):
        raise ValueError('Loss completion/aggregation differs')
    return total/count


def probe_inputs(directory):
    probe_root=ROOT/'tests/fixtures/qwen-v2';full_root=ROOT/'artifacts/qwen-forward-v2'
    if prepare.digest(probe_root/'manifest.json')!=PROBE_SHA or prepare.digest(full_root/'manifest.json')!=FULL_SHA:
        raise ValueError('Loss probe frozen reference identity differs')
    probes=prepare.load_json(probe_root/'manifest.json');full=prepare.load_json(full_root/'manifest.json');cases=[]
    for name in ('future_changed','short'):
        spec=probes['cases'][name]
        if full['cases'][name]!=spec or spec['offset']!=0 or len(spec['tokens'])!=4:
            raise ValueError('Loss probe token/position identity differs')
        key=f'cases.{name}.logits';item=full['arrays'][key];anchor=probes['arrays'][key]
        if (item['shape']!=[4,151936] or item['bytes']!=4*151936*4 or item['sha256']!=anchor['full_array_sha256']
                or item['bytes']!=anchor['full_array_bytes'] or item['shape']!=anchor['shape']):
            raise ValueError('Loss probe complete-logit metadata differs')
        source=full_root/item['file']
        if not source.resolve().is_relative_to(full_root.resolve()):raise ValueError('Loss probe payload path escapes fixture')
        with source.open('rb') as file:data=file.read(item['bytes']+1)
        if len(data)!=item['bytes'] or hashlib.sha256(data).hexdigest()!=item['sha256']:
            raise ValueError('Loss probe complete-logit identity differs')
        reference=[]
        for index,target in enumerate(spec['tokens'][1:]):
            row=struct.unpack('<151936f',data[index*151936*4:(index+1)*151936*4])
            reference.append(stable_loss(row,target))
        token_path=directory/f'{name}.u32le'
        with token_path.open('xb') as file:file.write(struct.pack('<4I',*spec['tokens']))
        cases.append({'name':name,'tokens':spec['tokens'],'path':str(token_path),'corpus_sha256':prepare.digest(token_path),
                      'source_logit_path':str(source),'source_logit_sha256':item['sha256'],'source_nll':reference})
    return cases


def child(binary,case,projection,raw,errors):
    command=[str(binary),str(ROOT/'models/Qwen2.5-0.5B-Instruct'),case['path'],projection]
    worker=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=errors,text=True,start_new_session=True)
    killed=[]
    def kill():
        try:os.killpg(worker.pid,signal.SIGKILL);killed.append(True)
        except ProcessLookupError:pass
    watchdog=threading.Timer(12*60*60,kill);watchdog.start()
    record={'role':f'{projection}.{case["name"]}','command':command,'losses':[],'blocks':[],'passed':False}
    stream_sha=hashlib.sha256()
    try:
        while True:
            line=worker.stdout.readline(65537)
            if not line:raise ValueError('Loss worker exited before completed handshake')
            raw.write(json.dumps({'role':record['role'],'line':line})+'\n');raw.flush();stream_sha.update(line.encode())
            if len(line)>65536 or not line.endswith('\n'):raise ValueError('Loss worker line exceeds bound')
            def finite_float(text):
                value=float(text)
                if not math.isfinite(value):raise ValueError('Nonfinite JSON loss number')
                return value
            def invalid_constant(text):raise ValueError(f'Invalid JSON constant {text}')
            event=json.loads(line,parse_float=finite_float,parse_constant=invalid_constant);kind=event.get('event')
            if kind=='ready' and 'ready' not in record and not record['losses']:
                validate_ready(event,case['tokens'],projection,case['corpus_sha256']);record['ready']=event
            elif kind=='loss' and 'ready' in record:
                if len(record['losses'])>=len(case['tokens'])-1 or len(record['blocks'])!=len(record['losses'])//1024:
                    raise ValueError('Loss worker emitted extra or unordered targets')
                record['losses'].append(event)
            elif kind=='block_done' and 'ready' in record:
                if (len(record['blocks'])>=(len(case['tokens'])-2)//1024+1
                        or len(record['losses'])!=min((len(record['blocks'])+1)*1024,len(case['tokens'])-1)):
                    raise ValueError('Loss worker emitted extra or unordered blocks')
                record['blocks'].append(event)
            elif kind=='done' and 'ready' in record:record['done']=event;break
            else:raise ValueError('Loss worker event order/type differs')
        peak=common.footprint_peak(worker.pid)
        record['physical_process_charge_peak_bytes']=peak
        record['mean_negative_log_likelihood']=validate_run(record,case['tokens'],projection,case['corpus_sha256'])
        worker.stdin.write('exit\n');worker.stdin.flush();worker.stdin.close()
        if worker.wait(timeout=20)!=0 or killed:raise ValueError('Loss worker failed after completion')
        if worker.stdout.read(65537):raise ValueError('Unexpected loss worker trailing output')
        record['passed']=True
    except BaseException as error:
        record['failure']={'type':type(error).__name__,'message':str(error)}
    finally:
        watchdog.cancel()
        if worker.poll() is None:kill()
        worker.wait(timeout=20)
        record.update(exit_code=worker.returncode,watchdog_killed=bool(killed),stdout_sha256=stream_sha.hexdigest())
        worker.stdout.close()
        if not worker.stdin.closed:worker.stdin.close()
    return record


def checked(command,record):
    started=time.perf_counter();worker=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True)
    stdout=stderr='';failure=None;killed=False;cleanup_errors=[]
    try:stdout,stderr=worker.communicate(timeout=1200)
    except BaseException as error:
        failure=error
        # TimeoutExpired may retain bytes even when Popen uses text mode.
        for name in ('stdout','stderr'):
            value=getattr(error,name,None)
            if isinstance(value,bytes):value=value.decode(errors='replace')
            if isinstance(value,str):
                if name=='stdout':stdout=value
                else:stderr=value
    finally:
        if failure is not None or worker.returncode:
            # The group belongs to this invocation. Kill it even if its leader
            # exited, because descendants may still hold our output pipes.
            try:os.killpg(worker.pid,signal.SIGKILL);killed=True
            except ProcessLookupError:pass
            except BaseException as error:cleanup_errors.append({'type':type(error).__name__,'message':str(error)})
            try:stdout,stderr=worker.communicate(timeout=20)
            except BaseException as error:
                cleanup_errors.append({'type':type(error).__name__,'message':str(error)})
                try:worker.wait(timeout=20)
                except BaseException as error:cleanup_errors.append({'type':type(error).__name__,'message':str(error)})
        result={'command':command,'exit_code':worker.returncode,'stdout':stdout,'stderr':stderr,
                'timed_out':isinstance(failure,subprocess.TimeoutExpired),'process_group_killed':killed,
                'cleanup_errors':cleanup_errors,'wall_seconds':time.perf_counter()-started}
        if failure is not None:result['failure']={'type':type(failure).__name__,'message':str(failure)}
        record['checks'].append(result)
        for stream in (worker.stdout,worker.stderr):
            if stream is not None:stream.close()
    if failure is not None:raise failure
    if worker.returncode or cleanup_errors:raise ValueError(f'Loss build/check failed: {command}')


def check_paths(output,build):
    if not output.resolve().is_relative_to((ROOT/'benchmarks/raw').resolve()) or not build.resolve().is_relative_to((ROOT/'artifacts').resolve()):
        raise ValueError('Loss output/build must use raw evidence and isolated artifact directories')
    evidence=[output,Path(str(output)+'.native.jsonl'),Path(str(output)+'.stderr.txt')]
    if build.exists() or any(path.exists() for path in evidence):raise ValueError('Refusing to overwrite loss evidence/build')
    if any(build.resolve()==path.resolve() or build.resolve() in path.resolve().parents or path.resolve() in build.resolve().parents for path in evidence):
        raise ValueError('Loss evidence and build must be disjoint')


def run(args):
    check_paths(args.output,args.build_dir)
    qualification_sha=prepare.digest(args.cpu_qualification)
    _,compiler,numerical=cpu.qualified_cpu_worker(args.cpu_qualification)
    if prepare.digest(args.cpu_qualification)!=qualification_sha:raise ValueError('Loss numerical qualification changed')
    if common.command(['rustc','--version'])!=compiler or os.environ.get('RUSTFLAGS')!=numerical['rustflags'] or os.environ.get('CARGO_ENCODED_RUSTFLAGS'):
        raise ValueError('Loss compiler/build flags differ from qualified CPU')
    prepare.verify(prepare.CONTRACT);manifest=prepare.load_json(prepare.CONTRACT/'manifest.json');policy=manifest['quality']
    if policy['loss_block_targets']!=1024 or policy['loss_context_policy']!=POLICY or policy['scored_loss_tokens']!=10000:
        raise ValueError('Frozen loss context/scoring policy differs')
    sources=prepare.load_json(prepare.SOURCES)
    assets=[sources['model']['weights']]+[asset for asset in sources['assets'] if asset['role']=='model_metadata_or_tokenizer']
    for asset in assets:prepare.validate_source(asset)
    code={str(path.relative_to(ROOT)):prepare.digest(path) for path in (ROOT/'src').rglob('*.rs')}
    code.update({name:prepare.digest(ROOT/name) for name in ('Cargo.toml','Cargo.lock','examples/quality_bench.rs')})
    helpers={str(Path(module.__file__).relative_to(ROOT)):prepare.digest(Path(module.__file__)) for module in (cpu,common,prepare)}
    helpers['tools/bench/quality.py']=prepare.digest(Path(__file__))
    record={'format':FORMAT,'completed':False,'qualification':'functional_probe' if args.probe else 'held_out_10000',
        'engine_revision':common.command(['git','rev-parse','HEAD']),'working_tree_dirty':bool(common.command(['git','status','--porcelain'])),
        'compiler':compiler,'rustflags':os.environ.get('RUSTFLAGS'),'source_sha256':code,'helper_sha256':helpers,
        'numerical_qualification':{'path':str(args.cpu_qualification),'sha256':qualification_sha},
        'contract_manifest_sha256':prepare.digest(prepare.CONTRACT/'manifest.json'),'model':sources['model'],
        'context_policy':POLICY,'checks':[],'runs':[],
        'environment':{'hardware':common.command(['/usr/sbin/sysctl','-n','hw.model']),'chip':common.command(['/usr/sbin/sysctl','-n','machdep.cpu.brand_string']),
            'os_version':common.command(['/usr/bin/sw_vers','-productVersion']),'os_build':common.command(['/usr/bin/sw_vers','-buildVersion']),
            'power_start':common.command(['/usr/bin/pmset','-g','batt'])+'\n'+common.command(['/usr/bin/pmset','-g','custom']),
            'thermal_start':common.command(['/usr/bin/pmset','-g','therm'])},
        'scope':'Completed held-out loss or explicitly short independent functional probes. No timing acceptance, task pass-rate, lossy-model, full quality-suite or M2.5/M3 milestone completion.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    raw_path=Path(str(args.output)+'.native.jsonl');error_path=Path(str(args.output)+'.stderr.txt')
    try:
        for command in (["cargo","fmt","--check"],["cargo","clippy","--locked","--offline","--all-targets","--","-D","warnings"],
                        ["cargo","test","--locked","--offline"],["cargo","test","--locked","--offline","--example","quality_bench"]):checked(command,record)
        checked(['cp','-R',str(ROOT/'artifacts/cpu-tiled-cli-v1'),str(args.build_dir)],record)
        command=['env',f'CARGO_TARGET_DIR={args.build_dir.resolve()}','cargo','build','--locked','--offline','--release','--example','quality_bench','-j','2']
        checked(command,record);binary=args.build_dir/'release/examples/quality_bench'
        record['binary_path']=str(binary);record['binary_sha256']=prepare.digest(binary)
        directory=args.build_dir/'quality-inputs';directory.mkdir()
        if args.probe:cases=probe_inputs(directory);modes=('scalar','tiled')
        else:
            item=manifest['arrays']['held_out_loss'];path=prepare.CONTRACT/item['file'];tokens=prepare.read_tokens(path)
            if len(tokens)!=10001 or item['count']!=10001 or prepare.digest(path)!=item['sha256']:raise ValueError('Frozen loss corpus identity differs')
            cases=[{'name':'held_out','tokens':tokens,'path':str(path),'corpus_sha256':item['sha256']}];modes=(args.cpu_projection,)
        with raw_path.open('x') as raw,error_path.open('x') as errors:
            for projection in modes:
                for case in cases:
                    result=child(binary,case,projection,raw,errors);record['runs'].append(result)
                    if not result['passed']:raise ValueError(f'Loss worker failed: {result.get("failure")}')
                    if args.probe:
                        delta=check_probe(result['losses'],case['source_nll'])
                        result['source_nll']=case['source_nll'];result['source_nll_deltas']=delta;result['source_logit_sha256']=case['source_logit_sha256']
        if args.probe:
            for name in ('future_changed','short'):
                matching=[run for run in record['runs'] if run['role'].endswith('.'+name)]
                if matching[0]['losses']!=matching[1]['losses']:raise ValueError('Scalar/tiled complete logit/loss evidence differs')
            record['probe_manifest_sha256']=PROBE_SHA;record['full_reference_manifest_sha256']=FULL_SHA
            if prepare.digest(ROOT/'tests/fixtures/qwen-v2/manifest.json')!=PROBE_SHA or prepare.digest(ROOT/'artifacts/qwen-forward-v2/manifest.json')!=FULL_SHA:
                raise ValueError('Loss probe manifests changed after execution')
            for case in cases:
                if prepare.digest(Path(case['source_logit_path']))!=case['source_logit_sha256']:
                    raise ValueError('Loss probe reference logits changed after execution')
        for case in cases:
            if prepare.digest(Path(case['path']))!=case['corpus_sha256']:raise ValueError('Loss input tokens changed after execution')
        for name,sha in {**code,**helpers}.items():
            if prepare.digest(ROOT/name)!=sha:raise ValueError(f'Loss source/helper changed: {name}')
        if prepare.digest(binary)!=record['binary_sha256'] or os.environ.get('RUSTFLAGS')!=record['rustflags']:
            raise ValueError('Loss binary/build flags changed')
        for asset in assets:prepare.validate_source(asset)
        prepare.verify(prepare.CONTRACT);cpu.qualified_cpu_worker(args.cpu_qualification)
        if prepare.digest(args.cpu_qualification)!=qualification_sha:raise ValueError('Loss numerical qualification changed after execution')
        record['completed']=True
    except BaseException as error:
        record['failure']={'type':type(error).__name__,'message':str(error)}
        raise
    finally:
        record['native_raw_sha256']=prepare.digest(raw_path) if raw_path.exists() else None
        record['stderr_sha256']=prepare.digest(error_path) if error_path.exists() else None
        record['environment']['thermal_end']=common.command(['/usr/bin/pmset','-g','therm'])
        with args.output.open('x') as file:json.dump(record,file,indent=2,allow_nan=False);file.write('\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--build-dir',type=Path,required=True)
    parser.add_argument('--cpu-qualification',type=Path,required=True);parser.add_argument('--cpu-projection',choices=('scalar','tiled'),default='scalar')
    parser.add_argument('--probe',action='store_true',help='two independent short cases in both CPU modes; no held-out quality acceptance')
    args=parser.parse_args();run(args)


if __name__=='__main__':main()
