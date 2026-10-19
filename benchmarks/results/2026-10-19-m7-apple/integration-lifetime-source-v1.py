"""Bounded tiny C integration on Mac CPU/Metal and the owned lab simulator.

The caller supplies a reviewed package-build SHA. The dedicated simulator's
prior shutdown/owned boot are frozen here; it is shut down and verified even
when its host launcher fails. No physical-device qualification follows.
"""
import json
from pathlib import Path
import sys
import m7_apple_package as package

FIXTURE={
    'artifacts/m7-native-tiny-v2/config.json':'84b6e6078f35006f49ae2e2a8663093cdc8d2813e2b7f28ebfa6142113e9c1d9',
    'artifacts/m7-native-tiny-v2/model.safetensors':'b01a114bf2d44f0c6cb438035e85535b658e803c76691729facd78b15eefbce2',
}
OWNED_SIMULATOR={
    'artifacts/m7-selected-simulator-v1.json':'54d8540358c1157d36038e4a7a976e725c4b46775030ec6cb8aed2fb334edde5',
    'artifacts/m7-simulator-bootstatus-v2.json':'539a4131c54807746cf2e0b5ca176128cc44e1ac092250136849a820123d6760',
}


def metadata(path,expected):
    text=package.small(path)
    if package.hashlib.sha256(text.encode()).hexdigest()!=expected:raise ValueError('Metadata SHA differs: '+str(path))
    return json.loads(text)


def merge(target,items):
    for key,value in items.items():
        if key in target and target[key]!=value:raise ValueError('Conflicting binding: '+key)
        target[key]=value


def admit_build(folder,expected):
    root=package.ROOT;build=metadata(folder/'build.json',expected)
    if not build['passed'] or build['format']!='toklet-m7-apple-package-build-v1':raise ValueError('Successful reviewed package required')
    rows=build['slices']
    if len(rows)!=3 or {s['target'] for s in rows}!={s['target'] for s in package.SPECS}:raise ValueError('Exact three-slice inventory required')
    pins={};merge(pins,build['source_pins']);merge(pins,build['package_pins'])
    for spec in package.SPECS:
        row=next(s for s in rows if s['target']==spec['target'])
        for key in ('triple','platform','minos','name','variant'):
            if row[key]!=spec[key]:raise ValueError('Wrong slice platform/minimum')
        expected_path=str((folder/(spec['target']+'-link-smoke')).relative_to(root))
        if row['link_executable']!=expected_path:raise ValueError('Remapped executable')
        library=str((folder/'target'/spec['target']/'release/libtoklet.a').relative_to(root))
        if row['library']!=library:raise ValueError('Remapped library')
        merge(pins,{expected_path:row['link_sha256'],library:row['library_sha256']})
    merge(pins,{str((folder/'build.json').relative_to(root)):expected})
    return pins


def admit_simulator(simulator):
    records=[metadata(package.ROOT/name,pin) for name,pin in OWNED_SIMULATOR.items()]
    inventory,boot=records
    if inventory['udid']!=simulator or inventory['name']!='Inference Lab iPhone 14' or inventory['state']!='Shutdown' or not inventory['isAvailable']:
        raise ValueError('Not the owned, previously shutdown lab simulator')
    if boot['command']!=['xcrun','simctl','bootstatus',simulator,'-b'] or boot['exit_code']!=0 or boot['primary_error'] or boot['cleanup_errors']:
        raise ValueError('Owned simulator boot evidence differs')
    return dict(OWNED_SIMULATOR)


def shutdown_simulator(folder,simulator,env):
    rows=[];errors=[];verified=False
    for label,command in [('sim-shutdown-v3',['xcrun','simctl','shutdown',simulator]),('sim-state-v3',['xcrun','simctl','list','devices','--json'])]:
        row=package.owned(command,folder/'logs'/(label+'.stdout'),folder/'logs'/(label+'.stderr'),env,60);rows.append(row)
        if row['primary_error'] or row['cleanup_errors']:errors.append(label+': '+str(row['primary_error'])+' '+repr(row['cleanup_errors']))
    try:
        state=json.loads(package.small(package.ROOT/rows[-1]['stdout']))
        matches=[d for devices in state['devices'].values() for d in devices if d['udid']==simulator]
        verified=len(matches)==1 and matches[0]['state']=='Shutdown'
        if not verified:errors.append('Owned simulator shutdown not verified')
    except BaseException as error:errors.append('State verification: '+type(error).__name__+': '+str(error))
    return dict(workers=rows,errors=errors,shutdown_verified=verified)


def main(folder,simulator,expected):
    folder=folder.resolve();root=package.ROOT
    if not folder.is_relative_to(root/'artifacts'):raise ValueError('Expected repository artifact folder')
    output=folder/'integration-v3.json'
    if output.exists():raise ValueError('Never replace integration evidence')
    rows=[];before={};primary=None;endpoints=[];cleanup=None;simulator_owned=False
    previous=package.signal.signal(package.signal.SIGTERM,package.interrupted)
    try:
        env,_=package.build_environment();env['TMPDIR']=str(folder/'tmp')
        merge(before,admit_build(folder,expected));merge(before,FIXTURE)
        merge(before,{str(Path(__file__).resolve().relative_to(root)):package.sha(Path(__file__))})
        for name,pin in before.items():
            if package.sha(root/name)!=pin:raise ValueError('Preflight changed: '+name)
        owned=admit_simulator(simulator);simulator_owned=True;merge(before,owned)
        fixture=str(root/'artifacts/m7-native-tiny-v2')
        commands=[('mac-cpu-v3',[str(folder/'aarch64-apple-darwin-link-smoke'),fixture]),
                  ('mac-metal-v3',[str(folder/'aarch64-apple-darwin-link-smoke'),fixture,'metal']),
                  ('sim-cpu-v3',['xcrun','simctl','spawn',simulator,str(folder/'aarch64-apple-ios-sim-link-smoke'),fixture])]
        for label,command in commands:
            row=package.owned(command,folder/'logs'/(label+'.stdout'),folder/'logs'/(label+'.stderr'),env,180);rows.append(row)
            if row['primary_error'] or row['cleanup_errors']:raise RuntimeError(row['primary_error'] or repr(row['cleanup_errors']))
            if json.loads(package.small(root/row['stdout']))!={'ids':[47,47,47,47],'abi':1,'passed':True}:raise ValueError('C oracle output differs')
    except BaseException as error:primary=type(error).__name__+': '+str(error)
    finally:
        if simulator_owned:
            try:cleanup=shutdown_simulator(folder,simulator,env)
            except BaseException as error:cleanup=dict(errors=[type(error).__name__+': '+str(error)],shutdown_verified=False)
        package.signal.signal(package.signal.SIGTERM,previous)
        for name,pin in before.items():
            try:
                if package.sha(root/name)!=pin:endpoints.append('Changed endpoint: '+name)
            except OSError as error:endpoints.append('Unavailable endpoint: '+name+': '+str(error))
    passed=primary is None and not endpoints and len(rows)==3 and cleanup is not None and cleanup['shutdown_verified'] and not cleanup['errors']
    result=dict(format='toklet-m7-apple-integration-v2',passed=passed,reviewed_build_sha256=expected,bindings=before,workers=rows,
                primary_error=primary,endpoint_errors=endpoints,simulator_cleanup=cleanup,physical_device_qualified=False,m7_complete=False)
    with output.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(passed=passed,primary_error=primary,endpoint_errors=endpoints,simulator_cleanup=cleanup)))
    return 0 if passed else 1


if __name__=='__main__':sys.exit(main(Path(sys.argv[1]),sys.argv[2],sys.argv[3]))
