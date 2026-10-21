"""Run the admitted Swift app self-test and capture its idle simulator UI."""
import json
from pathlib import Path
import sys
import time
import m7_apple_integration as integration
import m7_apple_package as package
import m7_swift_harness_build as builder
ROOT=package.ROOT


def admit(folder,expected):
    build=integration.metadata(folder/'build.json',expected)
    if not build['passed'] or build['format']!='toklet-m7-swift-harness-build-v1':raise ValueError('Reviewed successful app build required')
    rows=build['apps']
    if len(rows)!=2 or {r['platform'] for r in rows}!={'iPhoneOS','iPhoneSimulator'}:raise ValueError('Exact app inventory required')
    row=next(r for r in rows if r['platform']=='iPhoneSimulator')
    app=folder/'derived-iphonesimulator/Build/Products/Release-iphonesimulator/TokletHarness.app'
    if row['app']!=str(app.relative_to(ROOT)) or row['executable']!=str((app/'TokletHarness').relative_to(ROOT)) or row['signed']!='ad-hoc':raise ValueError('Remapped/unsigned simulator app')
    pins={};integration.merge(pins,build['bindings']);integration.merge(pins,build['bundle_pins'])
    integration.merge(pins,{str((app/'TokletHarness').relative_to(ROOT)):row['executable_sha256'],str((folder/'build.json').relative_to(ROOT)):expected})
    builder.verify_library_link(build['library_link'])
    for name,pin in pins.items():
        if package.sha(ROOT/name)!=pin:raise ValueError('App input changed: '+name)
    return app,pins,build['library_link']


def installed_copy(original,installed,pins):
    copies={}
    for name,pin in pins.items():
        source=ROOT/name
        if source.is_relative_to(original):
            target=installed/source.relative_to(original)
            if package.sha(target)!=pin:raise ValueError('Installed app differs: '+str(target))
            copies[str(target)]=pin
    if not copies:raise ValueError('No installed app files admitted')
    return copies


def main(folder,expected,simulator,output):
    folder=folder.resolve();output=output.resolve()
    if not folder.is_relative_to(ROOT/'artifacts') or not output.is_relative_to(ROOT/'artifacts'):raise ValueError('Repository artifacts required')
    output.mkdir();logs=output/'logs';logs.mkdir();temporary=output/'tmp';temporary.mkdir()
    commands=[];before={};installed={};primary=None;endpoints=[];cleanup=None;owned=False;result=None;screenshot=None;link=None;env={}
    previous=package.signal.signal(package.signal.SIGTERM,package.interrupted)
    def execute(command,label,timeout=180):
        row=package.owned(command,logs/(label+'.stdout'),logs/(label+'.stderr'),env,timeout);commands.append(row)
        if row['primary_error'] or row['cleanup_errors']:raise RuntimeError(row['primary_error'] or repr(row['cleanup_errors']))
        return ROOT/row['stdout']
    try:
        env,_=package.build_environment();env['TMPDIR']=str(temporary)
        app,before,link=admit(folder,expected)
        for path in [Path(__file__).resolve(),Path(__file__).with_name('test_m7_swift_harness_run.py')]:integration.merge(before,{str(path.relative_to(ROOT)):package.sha(path)})
        integration.merge(before,integration.admit_simulator(simulator));owned=True
        execute(['xcrun','simctl','bootstatus',simulator,'-b'],'boot')
        execute(['codesign','--verify','--strict',str(app)],'signature')
        execute(['xcrun','simctl','install',simulator,str(app)],'install')
        location=execute(['xcrun','simctl','get_app_container',simulator,'dev.toklet.harness','app'],'container')
        installed_app=Path(package.small(location).strip()).resolve()
        inventory_name,inventory_pin=next(iter(integration.OWNED_SIMULATOR.items()))
        inventory=integration.metadata(ROOT/inventory_name,inventory_pin)
        device_root=Path(inventory['dataPath'])/'Containers/Bundle/Application'
        if not installed_app.is_relative_to(device_root.resolve()) or installed_app.name!='TokletHarness.app':raise ValueError('Installed bundle escapes owned simulator')
        installed=installed_copy(app,installed_app,before)
        raw=execute(['xcrun','simctl','launch','--console','--terminate-running-process',simulator,'dev.toklet.harness','--self-test'],'self-test')
        lines=[line.removeprefix('TOKLET_SELFTEST_JSON=') for line in package.small(raw).splitlines() if line.startswith('TOKLET_SELFTEST_JSON=')]
        if len(lines)!=1:raise ValueError('Expected one app self-test result')
        result=json.loads(lines[0])
        expected_states=[('load',0,0,0,0,0),('prompt',6,5,1,6,1),('continue',8,7,1,8,1),('suspend',0,0,0,8,0),('replay-cancelled',0,0,0,8,3),('replay',8,8,0,8,1),('after-replay',9,8,1,9,1),('reset',0,0,0,0,0),('cancel',5,4,1,5,3),('resume',6,5,1,6,1)]
        observed=[tuple(s[k] for k in ('name','history','evaluated','pending','saved','reason')) for s in result['states']]
        if observed!=expected_states or not result['passed'] or result['backend']!='cpu' or result['tokens']!=[47]*4 or not result['closed'] or not result['released'] or result['physicalDeviceQualified']:raise ValueError('App self-test oracle/state differs')
        execute(['xcrun','simctl','launch','--terminate-running-process',simulator,'dev.toklet.harness'],'ui-launch')
        time.sleep(2) # Allow initial UI layout; no latency measurement is made.
        image=output/'ui.png'
        execute(['xcrun','simctl','io',simulator,'screenshot',str(image)],'ui-screenshot')
        screenshot=dict(path=str(image.relative_to(ROOT)),sha256=package.sha(image))
    except BaseException as error:primary=type(error).__name__+': '+str(error)
    finally:
        if owned:
            try:cleanup=integration.shutdown_simulator(output,simulator,env,prefix='harness')
            except BaseException as error:cleanup=dict(errors=[str(error)],shutdown_verified=False)
        package.signal.signal(package.signal.SIGTERM,previous)
        if link is not None:
            try:builder.verify_library_link(link)
            except (OSError,ValueError) as error:endpoints.append(str(error))
        for name,pin in {**{str(ROOT/p):v for p,v in before.items()},**installed}.items():
            try:
                if package.sha(Path(name))!=pin:endpoints.append('Changed endpoint: '+name)
            except OSError as error:endpoints.append('Unavailable endpoint: '+name+': '+str(error))
    passed=primary is None and not endpoints and result is not None and screenshot is not None and cleanup is not None and cleanup['shutdown_verified'] and not cleanup['errors']
    record=dict(format='toklet-m7-swift-harness-run-v1',passed=passed,reviewed_build_sha256=expected,bindings=before,installed_app_pins=installed,
                commands=commands,result=result,screenshot=screenshot,primary_error=primary,endpoint_errors=endpoints,simulator_cleanup=cleanup,
                physical_device_qualified=False,m7_complete=False)
    with (output/'run.json').open('x') as stream:json.dump(record,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(passed=passed,primary_error=primary,endpoint_errors=endpoints)))
    return 0 if passed else 1


if __name__=='__main__':sys.exit(main(Path(sys.argv[1]),sys.argv[2],sys.argv[3],Path(sys.argv[4])))
