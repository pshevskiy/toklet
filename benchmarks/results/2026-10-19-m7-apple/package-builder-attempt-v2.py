"""Build and inspect separate Apple arm64 static-library slices, without signing.

Requires already installed Xcode SDKs and Rust targets. Outputs are exclusive.
Build/link qualification is distinct from simulator and physical-device runs.
"""
import hashlib
import json
import os
from pathlib import Path
import plistlib
import re
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
SPECS=[
    dict(target='aarch64-apple-darwin',sdk='macosx',triple='arm64-apple-macos15.0',platform=1,minos='15.0',name='macos',variant=''),
    dict(target='aarch64-apple-ios',sdk='iphoneos',triple='arm64-apple-ios18.0',platform=2,minos='18.0',name='ios',variant=''),
    dict(target='aarch64-apple-ios-sim',sdk='iphonesimulator',triple='arm64-apple-ios18.0-simulator',platform=7,minos='18.0',name='ios',variant='simulator'),
]


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def small(path):
    with path.open('rb') as stream:raw=stream.read(16*1024**2+1)
    if len(raw)>16*1024**2:raise ValueError('Oversize tool metadata: '+str(path))
    return raw.decode()


def group_exists(pid):
    try:os.killpg(pid,0);return True
    except ProcessLookupError:return False


def drain_group(process,errors,timeout):
    # Reaping Cargo alone does not reap its rustc/linker descendants.
    for sig in (signal.SIGTERM,signal.SIGKILL):
        if group_exists(process.pid):
            try:os.killpg(process.pid,sig)
            except ProcessLookupError:pass
            except OSError as error:errors.append('signal: '+str(error))
        deadline=time.monotonic()+timeout
        while True:
            process.poll()  # Reap the direct child before testing its group.
            if not group_exists(process.pid):return
            if time.monotonic()>=deadline:break
            time.sleep(min(.05,max(0,deadline-time.monotonic())))
    errors.append('owned process group remains after bounded SIGKILL wait')


def owned(command,stdout,stderr,env,timeout=1800,cleanup_timeout=15):
    record=dict(command=command,exit_code=None,primary_error=None,cleanup_errors=[],pid=None)
    process=None
    with stdout.open('xb') as out,stderr.open('xb') as err:
        try:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=out,stderr=err,start_new_session=True)
            record['pid']=process.pid;record['exit_code']=process.wait(timeout=timeout)
            if record['exit_code']!=0:record['primary_error']='tool exit '+str(record['exit_code'])
        except BaseException as error:record['primary_error']=type(error).__name__+': '+str(error)
        finally:
            if process is not None:
                try:drain_group(process,record['cleanup_errors'],cleanup_timeout)
                except BaseException as error:record['cleanup_errors'].append('group cleanup: '+type(error).__name__+': '+str(error))
                try:record['exit_code']=process.wait(timeout=cleanup_timeout)
                except BaseException as error:record['cleanup_errors'].append('reap: '+type(error).__name__+': '+str(error))
    record.update(stdout=str(stdout.relative_to(ROOT)),stderr=str(stderr.relative_to(ROOT)))
    return record


def build_environment():
    home=Path.home()
    env=dict(HOME=str(home),PATH='/usr/bin:/bin:/usr/sbin:/sbin:'+str(home/'.cargo/bin'),
             CARGO_HOME=str(home/'.cargo'),RUSTUP_HOME=str(home/'.rustup'),
             DEVELOPER_DIR='/Applications/Xcode.app/Contents/Developer',LANG='en_US.UTF-8')
    # Offline caches are reused, but ambient Cargo configuration is not admitted.
    candidates={parent/'.cargo'/name for parent in (ROOT,*ROOT.parents) for name in ('config','config.toml')}
    candidates.update(Path(env['CARGO_HOME'])/name for name in ('config','config.toml'))
    for path in candidates:
        if path.exists():raise ValueError('Ambient Cargo configuration is not admitted: '+str(path))
    return env,sorted(str(p) for p in candidates)


def native_symbols(path):
    symbols=set()
    with path.open() as stream:
        while True:
            line=stream.readline(65537)
            if not line:break
            if len(line)>65536:raise ValueError('Oversize symbol line')
            words=line.split()
            if words and words[-1].startswith('_toklet_'):symbols.add(words[-1][1:])
    return symbols


def check_load_commands(text,spec):
    blocks=re.findall(r'cmd LC_BUILD_VERSION\s+cmdsize \d+\s+platform (\d+)\s+minos ([\d.]+)',text)
    if blocks!=[(str(spec['platform']),spec['minos'])]:
        raise ValueError('Linked executable platform/minimum mismatch: '+repr(blocks))


def check_bundle(bundle,slices,headers):
    info=plistlib.loads(small(bundle/'Info.plist').encode());rows=info['AvailableLibraries']
    if len(rows)!=len(slices):raise ValueError('XCFramework slice count mismatch')
    found=set()
    for row in rows:
        key=(row['SupportedPlatform'],row.get('SupportedPlatformVariant',''))
        if key in found or row['SupportedArchitectures']!=['arm64']:raise ValueError('Duplicate or wrong-architecture slice')
        found.add(key)
        spec=next((s for s in slices if (s['name'],s['variant'])==key),None)
        if spec is None:raise ValueError('Unexpected XCFramework platform')
        library=(bundle/row['LibraryIdentifier']/row['LibraryPath']).resolve()
        header_dir=(bundle/row['LibraryIdentifier']/row['HeadersPath']).resolve()
        if not library.is_relative_to(bundle.resolve()) or not header_dir.is_relative_to(bundle.resolve()):
            raise ValueError('XCFramework path escape')
        if sha(library)!=spec['library_sha256']:raise ValueError('XCFramework library copy differs')
        for name,expected in headers.items():
            if sha(header_dir/name)!=expected:raise ValueError('XCFramework header copy differs')
    if found!={(s['name'],s['variant']) for s in slices}:raise ValueError('Missing XCFramework platform')
    return info


def interrupted(signum,frame):
    raise InterruptedError('Received signal '+str(signum))


def main(destination):
    destination=destination.resolve()
    if not destination.is_relative_to(ROOT/'artifacts'):raise ValueError('Build destination must be inside repository artifacts')
    destination.mkdir()  # Never replace a prior package or raw log.
    logs=destination/'logs';logs.mkdir();commands=[];before={};slices=[];primary=None;endpoints=[]
    previous_handler=signal.signal(signal.SIGTERM,interrupted)
    controlled={};config_absence=[];bundle_pins={}
    env={}
    def execute(command,label,environment=None,timeout=1800):
        record=owned(command,logs/(label+'.stdout'),logs/(label+'.stderr'),environment or env,timeout)
        commands.append(record)
        if record['primary_error'] or record['cleanup_errors']:
            raise RuntimeError(record['primary_error'] or repr(record['cleanup_errors']))
        return ROOT/record['stdout']
    try:
        env,config_absence=build_environment()
        paths=[ROOT/'Cargo.toml',ROOT/'Cargo.lock',*sorted((ROOT/'src').rglob('*.rs')),ROOT/'src/metal/kernels.metal',
               ROOT/'apple/include/toklet.h',ROOT/'apple/include/module.modulemap',ROOT/'apple/tests/native_smoke.c',Path(__file__).resolve(),Path(__file__).with_name('test_m7_apple_package.py')]
        before={str(p.relative_to(ROOT)):sha(p) for p in paths}
        rustc=Path(small(execute(['rustup','which','--toolchain','1.90','rustc'],'rustc-path')).strip())
        cargo=Path(small(execute(['rustup','which','--toolchain','1.90','cargo'],'cargo-path')).strip())
        rustdoc=rustc.with_name('rustdoc')
        env.update(RUSTC=str(rustc),RUSTDOC=str(rustdoc),RUSTFLAGS='-Cpanic=unwind',
                   CARGO_TARGET_DIR=str(destination/'target'),MACOSX_DEPLOYMENT_TARGET='15.0',IPHONEOS_DEPLOYMENT_TARGET='18.0',
                   CARGO_PROFILE_RELEASE_PANIC='unwind')
        env.pop('CARGO_ENCODED_RUSTFLAGS',None)
        env.pop('RUSTC_WRAPPER',None);env.pop('RUSTC_WORKSPACE_WRAPPER',None)
        controlled=dict(env)
        controlled['cargo']=str(cargo)
        execute([str(rustc),'-vV'],'rustc-version')
        execute(['xcodebuild','-version'],'xcode-version',timeout=120)
        installed=set(small(execute(['rustup','target','list','--installed','--toolchain','1.90'],'installed-targets')).splitlines())
        if not {s['target'] for s in SPECS}.issubset(installed):raise ValueError('Required installed Rust target missing')
        expected_symbols=set(re.findall(r'\buint32_t\s+(toklet_\w+)\s*\(', (ROOT/'apple/include/toklet.h').read_text()))
        if not expected_symbols:raise ValueError('No C entrypoints in header')
        header_pins={p.name:sha(p) for p in (ROOT/'apple/include').iterdir() if p.is_file()}
        for spec in SPECS:
            environment=dict(env)
            sdk=small(execute(['xcrun','--sdk',spec['sdk'],'--show-sdk-path'],spec['target']+'-sdk')).strip()
            if not Path(sdk).is_dir():raise ValueError('SDK path does not exist')
            environment['SDKROOT']=sdk
            execute([str(cargo),'build','--locked','--offline','--release','--lib','--features','metal','--target',spec['target']],spec['target']+'-build',environment)
            library=destination/'target'/spec['target']/'release/libtoklet.a'
            nm=execute(['xcrun','nm','-gU',str(library)],spec['target']+'-symbols',environment,120)
            if native_symbols(nm)!=expected_symbols:raise ValueError('Exported C symbols differ from header')
            executable=destination/(spec['target']+'-link-smoke')
            execute(['xcrun','--sdk',spec['sdk'],'clang','-target',spec['triple'],'-isysroot',sdk,
                     '-std=c11','-Wall','-Wextra','-Werror','-I'+str(ROOT/'apple/include'),
                     str(ROOT/'apple/tests/native_smoke.c'),str(library),'-framework','Metal','-framework','MetalPerformanceShaders',
                     '-framework','Foundation','-framework','Security','-lobjc','-liconv','-lresolv','-lc++',
                     '-Wl,-fatal_warnings','-o',str(executable)],spec['target']+'-link',environment,120)
            load=execute(['xcrun','otool','-l',str(executable)],spec['target']+'-load-commands',environment,120)
            check_load_commands(small(load),spec)
            slices.append(dict(spec,library=str(library.relative_to(ROOT)),library_sha256=sha(library),link_executable=str(executable.relative_to(ROOT)),link_sha256=sha(executable),sdk=sdk,exported_c_symbols=sorted(expected_symbols)))
        bundle=destination/'Toklet.xcframework'
        command=['xcodebuild','-create-xcframework']
        for spec in slices:command.extend(['-library',str(ROOT/spec['library']),'-headers',str(ROOT/'apple/include')])
        command.extend(['-output',str(bundle)])
        execute(command,'xcframework',timeout=120)
        check_bundle(bundle,slices,header_pins)
        bundle_pins={str(p.relative_to(ROOT)):sha(p) for p in sorted(bundle.rglob('*')) if p.is_file()}
    except BaseException as error:primary=type(error).__name__+': '+str(error)
    finally:
        signal.signal(signal.SIGTERM,previous_handler)
        for name in config_absence:
            if Path(name).exists():endpoints.append('Cargo configuration appeared: '+name)
        for path,expected in before.items():
            try:
                if sha(ROOT/path)!=expected:endpoints.append('Changed source: '+path)
            except OSError as error:endpoints.append('Source unavailable: '+path+': '+str(error))
    passed=primary is None and not endpoints and len(slices)==3
    result=dict(format='toklet-m7-apple-package-build-v1',passed=passed,source_pins=before,controlled_environment=controlled,absent_cargo_configs=config_absence,package_pins=bundle_pins,commands=commands,slices=slices,
                primary_error=primary,endpoint_errors=endpoints,precise_math_unchanged=True,panic='unwind',
                supported_build_platforms=['macOS15-arm64','iOS18-arm64','iOS18-simulator-arm64'],
                ios_runtime_qualified=False,physical_device_qualified=False,m7_complete=False)
    with (destination/'build.json').open('x') as out:json.dump(result,out,indent=2);out.write('\n')
    print(json.dumps(dict(passed=passed,output=str(destination.relative_to(ROOT)),primary_error=primary,endpoint_errors=endpoints)))
    return 0 if passed else 1


if __name__=='__main__':sys.exit(main(Path(sys.argv[1])))
