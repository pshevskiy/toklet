"""Owned pure-stdlib routing capture; no MLX or numerical device execution."""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import sys

import m7_apple_package as owner
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/reference'))
import gemma_shared_kv as routing


def pin(path):
    if path.is_symlink() or not path.is_file():raise ValueError('Regular routing input required')
    with path.open('rb') as stream:raw=stream.read(128*1024*1024+1)
    if len(raw)>128*1024*1024:raise ValueError('Bounded routing input required')
    return dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def admitted_observation(path,expected):
    raw=routing.captured(path,expected['sha256'],routing.MAX_JSON)
    if len(raw)!=expected['bytes']:raise ValueError('Retained observation size differs')
    return json.loads(raw)


def collect(source,destination):
    if destination.is_symlink() or not destination.is_absolute() or not destination.parent.is_dir() or not destination.resolve().is_relative_to(ROOT/'artifacts'):
        raise ValueError('New absolute output below artifacts required')
    destination.mkdir();inputs={};outputs={};commands=[];primary=None;endpoint=[];capture=[];previous={}
    interpreter=Path(sys.executable).resolve()
    def guard():
        if Path(sys.executable).resolve()!=interpreter:raise ValueError('Python interpreter changed')
        for path,expected in inputs.items():
            if pin(Path(path))!=expected:raise ValueError('Routing input changed: '+path)
        for path,expected in outputs.items():
            if pin(destination/path)!=expected:raise ValueError('Routing output changed: '+path)
    def interrupted(number,_):
        for value in (signal.SIGTERM,signal.SIGINT):signal.signal(value,signal.SIG_IGN)
        raise InterruptedError('Received signal '+str(number))
    def retain(path,maximum):
        actual=pin(path)
        if actual['bytes']>maximum:raise ValueError('Routing output exceeds bound')
        key=str(path.relative_to(destination))
        if key in outputs and outputs[key]!=actual:raise ValueError('Retained output changed')
        outputs[key]=actual
    def invoke(label,command,expected=()):
        guard();cache=destination/(label+'-pycache');cache.mkdir()
        row=dict(label=label,command=command,owned_result=None,invocation_error=None);commands.append(row)
        try:row['owned_result']=owner.owned(command,destination/'logs'/(label+'.stdout'),destination/'logs'/(label+'.stderr'),dict(environment,PYTHONPYCACHEPREFIX=str(cache)),600)
        except (Exception,KeyboardInterrupt) as error:row['invocation_error']=type(error).__name__+': '+str(error)
        finally:
            for path,maximum in [(destination/'logs'/(label+'.'+suffix),8*1024*1024) for suffix in ('stdout','stderr')]+list(expected):
                try:retain(path,maximum)
                except (Exception,KeyboardInterrupt) as error:capture.append(str(path)+': '+type(error).__name__+': '+str(error))
        result=row['owned_result']
        if row['invocation_error'] or result['exit_code']!=0 or result['primary_error'] or result['cleanup_errors'] or capture:raise ValueError('Owned routing command failed')
        guard()
    environment=dict(HOME=str(Path.home()),PATH='/usr/bin:/bin:/usr/sbin:/sbin',LANG='en_US.UTF-8',PYTHONNOUSERSITE='1',TMPDIR=str(destination/'tmp'))
    try:
        for value in (signal.SIGTERM,signal.SIGINT):previous[value]=signal.signal(value,interrupted)
        raw=routing.captured(source,routing.SOURCE_SHA,128*1024);source_copy=destination/'gemma4_text.py'
        with source_copy.open('xb') as stream:stream.write(raw)
        for path in (source,source_copy):
            actual=pin(path)
            if actual!=dict(bytes=len(raw),sha256=routing.SOURCE_SHA):raise ValueError('Source buffer changed before pinning')
            inputs[str(path)]=actual
        selected=[ROOT/'tools/reference/gemma_shared_kv.py',ROOT/'tools/reference/test_gemma_shared_kv.py',ROOT/'tools/experiments/gemma_shared_kv_reference.py',ROOT/'tools/experiments/test_gemma_shared_kv_reference.py',Path(owner.__file__).resolve()]
        # Pure-stdlib control replay: retain the running interpreter and imported
        # Python/extension files, not an MLX numerical-runtime qualification.
        loaded=sorted({Path(module.__file__).resolve() for module in list(sys.modules.values()) if getattr(module,'__file__',None) and Path(module.__file__).is_file()})
        for path in [interpreter,*loaded,*selected,ROOT/'target/release/toklet',ROOT/'target/release/examples/cpu_bench']:inputs[str(path)]=pin(path)
        for path in selected:
            target=destination/'source-snapshot'/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
            raw=path.read_bytes()
            if hashlib.sha256(raw).hexdigest()!=inputs[str(path)]['sha256']:raise ValueError('Selected source changed before snapshot')
            with target.open('xb') as stream:stream.write(raw)
            inputs[str(target)]=pin(target)
        (destination/'logs').mkdir();(destination/'tmp').mkdir()
        invoke('ownership',[sys.executable,'-s','-m','unittest','discover','-s',str(ROOT/'tools/experiments'),'-p','test_gemma_shared_kv_reference.py','-v'])
        invoke('portable',[sys.executable,'-s','-m','unittest','discover','-s',str(ROOT/'tools/reference'),'-p','test_gemma_shared_kv.py','-v'])
        script=ROOT/'tools/reference/gemma_shared_kv.py';reference=destination/'expected.json'
        invoke('export',[sys.executable,'-s',str(script),'export','--output',str(reference)],[(reference,routing.MAX_JSON)])
        inputs[str(reference)]=pin(reference)
        observed=destination/'observation.json'
        invoke('compare',[sys.executable,'-s',str(script),'compare','--source',str(source_copy),'--reference',str(reference),'--expected-reference-sha256',inputs[str(reference)]['sha256'],'--output',str(observed)],[(observed,routing.MAX_JSON)])
        result=admitted_observation(observed,outputs[str(observed.relative_to(destination))])
        if result.get('passed') is not True or any(result.get(key) is not False for key in ('numerical_execution_qualified','native_stale_rejection_qualified','graph_qualified','device_qualified')):raise ValueError('Routing observation scope differs')
    except (Exception,KeyboardInterrupt) as error:primary=type(error).__name__+': '+str(error)
    finally:
        for value in previous:signal.signal(value,signal.SIG_IGN)
        try:guard()
        except (Exception,KeyboardInterrupt) as error:endpoint.append(type(error).__name__+': '+str(error))
    report=dict(format='toklet-gemma-shared-kv-owned-routing-v1',passed=primary is None and not endpoint and not capture and len(commands)==4,provisional=True,
        inputs=inputs,outputs=outputs,commands=commands,controlled_environment=environment,interpreter=dict(requested=sys.executable,resolved=str(interpreter),version=sys.version),
        primary_error=primary,endpoint_errors=endpoint,capture_errors=capture,fixture_accepted=False,numerical_execution_qualified=False,graph_qualified=False,device_qualified=False)
    try:
        with (destination/'capture.json').open('xb') as stream:stream.write(routing.encoded(report))
    finally:
        for value,handler in previous.items():signal.signal(value,handler)
    print(json.dumps(dict(passed=report['passed'],primary_error=primary,output=str(destination))))
    return 0 if report['passed'] else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',required=True,type=Path);parser.add_argument('--output',required=True,type=Path);options=parser.parse_args()
    raise SystemExit(collect(options.source,options.output))
