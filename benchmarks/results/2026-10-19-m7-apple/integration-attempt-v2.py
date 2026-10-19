"""Run the qualified tiny C client on Mac CPU/Metal and an explicit simulator.

Use only after reviewing a successful m7_apple_package build. This establishes
integration behavior, never physical-device numerical/performance qualification.
"""
import json
from pathlib import Path
import sys
import m7_apple_package as package

FIXTURE={
    'artifacts/m7-native-tiny-v2/config.json':'84b6e6078f35006f49ae2e2a8663093cdc8d2813e2b7f28ebfa6142113e9c1d9',
    'artifacts/m7-native-tiny-v2/model.safetensors':'b01a114bf2d44f0c6cb438035e85535b658e803c76691729facd78b15eefbce2',
}


def main(folder,simulator):
    folder=folder.resolve();root=package.ROOT
    if not folder.is_relative_to(root/'artifacts'):raise ValueError('Expected repository artifact folder')
    output=folder/'integration-v2.json'
    if output.exists():raise ValueError('Never replace integration evidence')
    rows=[];before={};primary=None;endpoints=[]
    previous=package.signal.signal(package.signal.SIGTERM,package.interrupted)
    try:
        text=package.small(folder/'build.json');build=json.loads(text)
        build_pin=package.hashlib.sha256(text.encode()).hexdigest()
        if not build['passed'] or build['format']!='toklet-m7-apple-package-build-v1':raise ValueError('Successful package build required')
        before=dict(build['source_pins'],**build['package_pins'],**FIXTURE)
        for row in build['slices']:before[row['link_executable']]=row['link_sha256']
        before[str((folder/'build.json').relative_to(root))]=build_pin
        before[str(Path(__file__).resolve().relative_to(root))]=package.sha(Path(__file__))
        for name,pin in before.items():
            if package.sha(root/name)!=pin:raise ValueError('Preflight changed: '+name)
        env,_=package.build_environment();env['TMPDIR']=str(folder/'tmp')
        fixture=str(root/'artifacts/m7-native-tiny-v2')
        commands=[('mac-cpu-v2',[str(folder/'aarch64-apple-darwin-link-smoke'),fixture]),
                  ('mac-metal-v2',[str(folder/'aarch64-apple-darwin-link-smoke'),fixture,'metal']),
                  ('sim-cpu-v2',['xcrun','simctl','spawn',simulator,str(folder/'aarch64-apple-ios-sim-link-smoke'),fixture])]
        for label,command in commands:
            row=package.owned(command,folder/'logs'/(label+'.stdout'),folder/'logs'/(label+'.stderr'),env,180);rows.append(row)
            if row['primary_error'] or row['cleanup_errors']:raise RuntimeError(row['primary_error'] or repr(row['cleanup_errors']))
            parsed=json.loads(package.small(root/row['stdout']))
            if parsed!={'ids':[47,47,47,47],'abi':1,'passed':True}:raise ValueError('C oracle output differs')
    except BaseException as error:primary=type(error).__name__+': '+str(error)
    finally:
        package.signal.signal(package.signal.SIGTERM,previous)
        for name,pin in before.items():
            try:
                if package.sha(root/name)!=pin:endpoints.append('Changed endpoint: '+name)
            except OSError as error:endpoints.append('Unavailable endpoint: '+name+': '+str(error))
    passed=primary is None and not endpoints and len(rows)==3
    result=dict(format='toklet-m7-apple-integration-v1',passed=passed,bindings=before,workers=rows,
                primary_error=primary,endpoint_errors=endpoints,physical_device_qualified=False,m7_complete=False)
    with output.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(passed=passed,primary_error=primary,endpoint_errors=endpoints)))
    return 0 if passed else 1


if __name__=='__main__':sys.exit(main(Path(sys.argv[1]),sys.argv[2]))
