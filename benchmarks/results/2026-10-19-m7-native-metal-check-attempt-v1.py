"""Bounded Mac Metal ABI checks; no latency/native-footprint/phone qualification."""
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / 'artifacts/m7-logs-v1'
OUTPUT = LOG / 'metal-native-validation-v1.json'
CPU_RECEIPT = 'benchmarks/results/2026-10-19-m7-native-abi-qualification-v1.json'
FROZEN = 'benchmarks/results/2026-10-19-m6-context-reviewed-closure-v1.json'
FROZEN_SHA = '9873fe34fb887a6dac2936f5ea505ee57bc2bbe7ae0da742b6e1cfc52ddb9320'
# CPU_SHA is the exact accepted first M7 task receipt, not a mutable review flag.
CPU_SHA = 'c5a1792857b5e791c1b36cb6fa71fb946a203b6a9ea755ff2e066919be926bd7'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def metadata(path, expected=None, jsonl=False):
    with path.open('rb') as stream:
        raw = stream.read(16*1024**2+1)
    actual = hashlib.sha256(raw).hexdigest()
    if len(raw)>16*1024**2 or (expected is not None and actual!=expected):
        raise ValueError('Metadata size or exact identity differs: '+str(path))
    value=[json.loads(row) for row in raw.splitlines()] if jsonl else json.loads(raw)
    return value, actual


def verify_frozen(paths, expected):
    if {str(path.relative_to(ROOT)) for path in paths} != set(expected):
        raise ValueError('Frozen payload inventory differs')
    for path in paths:
        if sha(path)!=expected[str(path.relative_to(ROOT))]:
            raise ValueError('Frozen payload differs: '+str(path))


def run(command, label, timeout=600):
    stdout, stderr = LOG/(label+'.stdout'), LOG/(label+'.stderr')
    result=dict(command=command,exit_code=None,primary_error=None,cleanup_errors=[],pid=None)
    process=None
    with stdout.open('xb') as out, stderr.open('xb') as err:
        try:
            process=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err,start_new_session=True)
            result['pid']=process.pid
            result['exit_code']=process.wait(timeout=timeout)
            if result['exit_code']!=0:
                result['primary_error']='native exit '+str(result['exit_code'])
        except BaseException as error:
            result['primary_error']=type(error).__name__+': '+str(error)
        finally:
            if process is not None:
                for sig in (signal.SIGTERM,signal.SIGKILL):
                    if process.poll() is None:
                        try:
                            os.killpg(process.pid,sig)
                        except ProcessLookupError:
                            pass  # natural-exit race; still perform bounded reap
                        except OSError as error:
                            result['cleanup_errors'].append('signal: '+str(error))
                    try:
                        result['exit_code']=process.wait(timeout=15)
                        break
                    except subprocess.TimeoutExpired:
                        if sig==signal.SIGKILL:
                            result['cleanup_errors'].append('owned process not reaped after bounded SIGKILL wait')
                    except BaseException as error:
                        result['cleanup_errors'].append('reap: '+type(error).__name__+': '+str(error))
    result['stdout']=str(stdout.relative_to(ROOT))
    result['stderr']=str(stderr.relative_to(ROOT))
    return result


def preflight():
    builds, build_sha=metadata(LOG/'metal-test-build.jsonl',jsonl=True)
    executables=[row['executable'] for row in builds if row.get('executable') and row['target']['kind']==['rlib','staticlib'] and row['target']['name']=='toklet' and row.get('profile',{}).get('test') is True]
    if len(executables)!=1:
        raise ValueError('Expected one compiled library test runner')
    executable=Path(executables[0])
    if executable.resolve().parent!=(ROOT/'artifacts/m7-check-target-v1/debug/deps').resolve():
        raise ValueError('Unexpected compiled test runner path')
    cpu,cpu_sha=metadata(ROOT/CPU_RECEIPT,CPU_SHA)
    proof,proof_sha=metadata(ROOT/FROZEN,FROZEN_SHA)
    wanted={f'models/{model}/{name}' for model in ('Qwen2.5-0.5B-Instruct','Qwen2.5-1.5B-Instruct')
            for name in ('config.json','model.safetensors')}
    packed='artifacts/m5-larger-q4-v2/'
    expected={path:expected for path,expected in proof['bindings'].items() if path in wanted or path.startswith(packed)}
    if len(expected)!=201 or not wanted.issubset(expected) or packed+'manifest.json' not in expected:
        raise ValueError('Accepted two-model/196-packed-payload inventory differs')
    frozen_paths=[ROOT/path for path in wanted]
    frozen_paths.extend([ROOT/(packed+'manifest.json'),*sorted((ROOT/packed).glob('*.q4'))])
    verify_frozen(frozen_paths,expected)
    tiny=['artifacts/m7-native-tiny-v2/config.json','artifacts/m7-native-tiny-v2/model.safetensors']
    for path in tiny:
        if sha(ROOT/path)!=cpu['evidence_pins'][path]:
            raise ValueError('Qualified tiny fixture changed')
    paths=[ROOT/'Cargo.toml',ROOT/'Cargo.lock',*sorted((ROOT/'src').rglob('*.rs')),
           ROOT/'src/metal/kernels.metal',ROOT/'apple/include/toklet.h',ROOT/'apple/tests/native_smoke.c',
           Path(__file__).resolve(),executable,ROOT/'artifacts/m7-native-metal-c-smoke-v1',
           *[ROOT/path for path in tiny],ROOT/CPU_RECEIPT,ROOT/FROZEN,
           LOG/'metal-test-build.jsonl']
    before={str(path.relative_to(ROOT)):sha(path) for path in paths}
    before.update(expected)
    for path,expected_sha in {CPU_RECEIPT:cpu_sha,FROZEN:proof_sha,
            'artifacts/m7-logs-v1/metal-test-build.jsonl':build_sha}.items():
        if before[path]!=expected_sha:
            raise ValueError('Parsed metadata changed before launch: '+path)
    return before,executable


def main():
    if OUTPUT.exists():
        raise ValueError('Exclusive validation receipt exists')
    before,workers,models,errors={ },[],[],[]
    primary=None
    try:
        before,executable=preflight()
        workers.append(run([str(executable),'native::tests::metal_native_selected_models_match_direct_sessions','--ignored','--nocapture'],'metal-native-selected-v1'))
        if workers[-1]['primary_error'] or workers[-1]['cleanup_errors']:
            raise ValueError('Selected native worker failed')
        output=(ROOT/workers[-1]['stdout']).read_text()
        rows=re.findall(r'native_metal_model=(\S+) ids=(\[[^\]]+\]) state=7/6/1 combined_admission_bytes=(\d+) passed',output)
        if len(rows)!=2 or {row[0] for row in rows}!={'Qwen2.5-0.5B-Instruct','Qwen2.5-1.5B-Instruct'}:
            raise ValueError('Incomplete selected model evidence')
        for name,ids,admission in rows:
            ids=json.loads(ids)
            if len(ids)!=3 or int(admission)>6*1024**3:
                raise ValueError('Invalid IDs or combined shared-weight admission')
            models.append(dict(model=name,token_ids=ids,combined_admission_bytes=int(admission)))
        workers.append(run([str(ROOT/'artifacts/m7-native-metal-c-smoke-v1'),str(ROOT/'artifacts/m7-native-tiny-v2'),'metal'],'metal-native-c-v1'))
        if workers[-1]['primary_error'] or workers[-1]['cleanup_errors']:
            raise ValueError('C native worker failed')
        tiny=json.loads((ROOT/workers[-1]['stdout']).read_text())
        if tiny.get('passed') is not True or tiny['ids']!=[47]*4:
            raise ValueError('C Metal result differs from frozen oracle')
    except BaseException as error:
        primary=type(error).__name__+': '+str(error)
    finally:
        for path,expected in before.items():
            try:
                if sha(ROOT/path)!=expected:
                    errors.append('Endpoint changed: '+path)
            except OSError as error:
                errors.append('Endpoint unavailable: '+path+': '+str(error))
    passed=primary is None and not errors and len(workers)==2
    result=dict(format='toklet-m7-native-metal-validation-v1',passed=passed,bindings=before,
                selected_models=models,workers=workers,primary_error=primary,endpoint_errors=errors,
                scope='Mac Metal ABI integration/state and combined logical admission only; no latency/native-footprint/mobile qualification')
    with OUTPUT.open('x') as out:
        json.dump(result,out,indent=2);out.write('\n')
    print(json.dumps(dict(passed=passed,bindings=len(before),models=models,primary_error=primary,endpoint_errors=errors)))
    return 0 if passed else 1


if __name__=='__main__':
    sys.exit(main())
