"""Build unsigned device and ad-hoc simulator harness apps from admitted slices."""
import json
from pathlib import Path
import plistlib
import sys
import m7_apple_integration as integration
import m7_apple_package as package
ROOT=package.ROOT


def prepare(package_folder):
    apple=ROOT/'apple'
    for name in ('Binaries','Generated'):
        folder=apple/name
        if folder.is_symlink() or (folder.exists() and not folder.is_dir()):raise ValueError('Foreign generated directory: '+str(folder))
        folder.mkdir(exist_ok=True)
    link=apple/'Binaries/Toklet.xcframework';expected=package_folder/'Toklet.xcframework'
    if link.is_symlink():
        if link.resolve()!=expected.resolve():raise ValueError('Refuse to replace a different library link')
    elif link.exists():raise ValueError('Refuse to replace foreign library input')
    else:link.symlink_to(expected,target_is_directory=True)
    tiny=apple/'Generated/Tiny'
    if tiny.is_symlink():raise ValueError('Foreign fixture directory link')
    tiny.mkdir(exist_ok=True)
    if {p.name for p in tiny.iterdir()}-{'config.json','model.safetensors'}:raise ValueError('Unexpected generated fixture file')
    for source,pin in integration.FIXTURE.items():
        with (ROOT/source).open('rb') as stream:raw=stream.read(1024**2+1)
        if len(raw)>1024**2 or package.hashlib.sha256(raw).hexdigest()!=pin:raise ValueError('Frozen tiny fixture differs')
        target=tiny/Path(source).name
        if target.is_symlink():raise ValueError('Foreign generated fixture link')
        if target.exists():
            if package.sha(target)!=pin:raise ValueError('Generated fixture differs')
        else:
            with target.open('xb') as stream:stream.write(raw)
    return {str(p.relative_to(ROOT)):package.sha(p) for p in tiny.iterdir()}


def bind_library_link():
    link=ROOT/'apple/Binaries/Toklet.xcframework'
    if not link.is_symlink():raise ValueError('Generated library link is missing')
    return dict(path=str(link.relative_to(ROOT)),target=str(link.readlink()))


def verify_library_link(binding):
    link=ROOT/binding['path']
    if not link.is_symlink() or str(link.readlink())!=binding['target']:
        raise ValueError('Generated library link changed')


def read_plist(path):
    with path.open('rb') as stream:raw=stream.read(1024**2+1)
    if len(raw)>1024**2:raise ValueError('Oversize app metadata')
    return plistlib.loads(raw)


def inspect_app(app,platform):
    info=read_plist(app/'Info.plist')
    name=info['CFBundleExecutable']
    if Path(name).name!=name or name in ('','..','.'):raise ValueError('Invalid app executable')
    if info['CFBundleIdentifier']!='dev.toklet.harness' or info['MinimumOSVersion']!='18.0':raise ValueError('Wrong app identity/deployment')
    if info['CFBundleSupportedPlatforms']!=[platform]:raise ValueError('Wrong app platform')
    for source,pin in integration.FIXTURE.items():
        if package.sha(app/'Tiny'/Path(source).name)!=pin:raise ValueError('Bundled tiny fixture differs')
    return app/name


def main(package_folder,expected,destination):
    package_folder=package_folder.resolve();destination=destination.resolve()
    if not package_folder.is_relative_to(ROOT/'artifacts') or not destination.is_relative_to(ROOT/'artifacts'):raise ValueError('Expected repository artifact folders')
    destination.mkdir();logs=destination/'logs';logs.mkdir();temporary=destination/'tmp';temporary.mkdir()
    before={};commands=[];apps=[];primary=None;endpoints=[];bundle_pins={};link_binding=None
    env={}
    previous=package.signal.signal(package.signal.SIGTERM,package.interrupted)
    def execute(command,label,timeout=900):
        record=package.owned(command,logs/(label+'.stdout'),logs/(label+'.stderr'),env,timeout);commands.append(record)
        if record['primary_error'] or record['cleanup_errors']:raise RuntimeError(record['primary_error'] or repr(record['cleanup_errors']))
        return ROOT/record['stdout']
    try:
        env,_=package.build_environment();env['TMPDIR']=str(temporary)
        integration.merge(before,integration.admit_build(package_folder,expected))
        paths=[ROOT/'.gitignore',*sorted((ROOT/'apple/Harness').glob('*')),ROOT/'apple/tests/swift/WorkerChecks.swift',
               *sorted((ROOT/'apple/TokletHarness.xcodeproj').rglob('project.pbxproj')),
               *sorted((ROOT/'apple/TokletHarness.xcodeproj').rglob('*.xcscheme')),
               *sorted((ROOT/'apple/TokletHarness.xcodeproj').rglob('*.xcworkspacedata')),
               Path(__file__).resolve(),Path(__file__).with_name('m7_apple_integration.py'),Path(__file__).with_name('test_m7_swift_harness_build.py')]
        integration.merge(before,{str(p.relative_to(ROOT)):package.sha(p) for p in paths if p.is_file()})
        integration.merge(before,integration.FIXTURE)
        for name,pin in before.items():
            if package.sha(ROOT/name)!=pin:raise ValueError('Preflight changed: '+name)
        integration.merge(before,prepare(package_folder));link_binding=bind_library_link()
        execute(['xcodebuild','-version'],'xcode-version',120)
        for sdk,platform,spec in [('iphonesimulator','iPhoneSimulator',package.SPECS[2]),('iphoneos','iPhoneOS',package.SPECS[1])]:
            derived=destination/('derived-'+sdk)
            command=['xcodebuild','-project',str(ROOT/'apple/TokletHarness.xcodeproj'),'-scheme','TokletHarness','-configuration','Release',
                     '-sdk',sdk,'-destination','generic/platform=iOS Simulator' if sdk=='iphonesimulator' else 'generic/platform=iOS',
                     '-derivedDataPath',str(derived),'-disableAutomaticPackageResolution','CODE_SIGNING_ALLOWED=NO','ARCHS=arm64',
                     'SDK_STAT_CACHE_DIR='+str(destination/'SDKStatCaches'),'CLANG_MODULE_CACHE_PATH='+str(destination/'ModuleCache'),
                     'COMPILER_INDEX_STORE_ENABLE=NO','build']
            verify_library_link(link_binding)
            execute(command,sdk+'-build')
            app=derived/'Build/Products'/('Release-'+sdk)/'TokletHarness.app'
            executable=inspect_app(app,platform)
            arch=execute(['xcrun','lipo','-archs',str(executable)],sdk+'-arch',120)
            if package.small(arch).strip()!='arm64':raise ValueError('Wrong app architecture')
            load=execute(['xcrun','otool','-l',str(executable)],sdk+'-load-commands',120)
            package.check_load_commands(package.small(load),spec)
            if sdk=='iphonesimulator':
                execute(['codesign','--force','--sign','-','--timestamp=none',str(app)],'simulator-sign',120)
                execute(['codesign','--verify','--strict',str(app)],'simulator-sign-verify',120)
            pins={str(p.relative_to(ROOT)):package.sha(p) for p in sorted(app.rglob('*')) if p.is_file()}
            integration.merge(bundle_pins,pins)
            apps.append(dict(platform=platform,app=str(app.relative_to(ROOT)),executable=str(executable.relative_to(ROOT)),
                             executable_sha256=package.sha(executable),signed='ad-hoc' if sdk=='iphonesimulator' else False))
    except BaseException as error:primary=type(error).__name__+': '+str(error)
    finally:
        package.signal.signal(package.signal.SIGTERM,previous)
        if link_binding is not None:
            try:verify_library_link(link_binding)
            except OSError as error:endpoints.append('Library link unavailable: '+str(error))
            except ValueError as error:endpoints.append(str(error))
        for name,pin in before.items():
            try:
                if package.sha(ROOT/name)!=pin:endpoints.append('Changed endpoint: '+name)
            except OSError as error:endpoints.append('Unavailable endpoint: '+name+': '+str(error))
    passed=primary is None and not endpoints and len(apps)==2
    result=dict(format='toklet-m7-swift-harness-build-v1',passed=passed,reviewed_package_sha256=expected,
                bindings=before,library_link=link_binding,commands=commands,controlled_environment=env,apps=apps,bundle_pins=bundle_pins,
                primary_error=primary,endpoint_errors=endpoints,physical_device_signed=False,physical_device_qualified=False,m7_complete=False)
    with (destination/'build.json').open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(passed=passed,primary_error=primary,endpoint_errors=endpoints)))
    return 0 if passed else 1


if __name__=='__main__':sys.exit(main(Path(sys.argv[1]),sys.argv[2],Path(sys.argv[3])))
