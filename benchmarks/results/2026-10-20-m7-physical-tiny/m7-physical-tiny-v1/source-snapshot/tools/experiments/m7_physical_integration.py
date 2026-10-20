"""Signed, bounded tiny CPU/Metal integration on one privately admitted iPhone.

The private selection binds the owner-selected team and physical device. Raw
signing/device logs stay in ignored artifacts. The public receipt contains
hashes and redacted observations, never credentials or device identifiers.
This controller does not qualify real-model memory, performance or lifecycle.
"""
import datetime
import json
from pathlib import Path
import plistlib
import re
import sys
from urllib.parse import unquote, urlparse
import m7_apple_package as package
import m7_apple_integration as integration
import m7_swift_harness_build as builder

ROOT = package.ROOT
DEVICECTL = '/Library/Developer/PrivateFrameworks/CoreDevice.framework/Versions/A/Resources/bin/devicectl'
BUNDLE = 'dev.toklet.harness'
STATES = [('load',0,0,0,0,0),('prompt',6,5,1,6,1),('continue',8,7,1,8,1),
          ('suspend',0,0,0,8,0),('replay-cancelled',0,0,0,8,3),('replay',8,8,0,8,1),
          ('after-replay',9,8,1,9,1),('reset',0,0,0,0,0),('cancel',5,4,1,5,3),('resume',6,5,1,6,1)]


def selection(path, expected):
    config = integration.metadata(path, expected)
    if set(config) != {'device_identifier','device_udid','development_team','product_type','os_version'}:
        raise ValueError('Exact private selection fields required')
    if not all(isinstance(v,str) and 0 < len(v) < 128 and '\n' not in v for v in config.values()):
        raise ValueError('Invalid private selection values')
    if not re.fullmatch(r'[A-Z0-9]{10}',config['development_team']):
        raise ValueError('Invalid owner-selected development team')
    if config['product_type'] != 'iPhone15,2' or config['os_version'] != '27.0.1':
        raise ValueError('Only the explicitly admitted iPhone14Pro/iOS27.0.1 is supported here')
    return config


def admit_device(data, config):
    d = data['result']; h = d['hardwareProperties']; p = d['deviceProperties']; c = d['connectionProperties']
    if (h['udid'] != config['device_udid'] or h['productType'] != config['product_type']
        or h['reality'] != 'physical' or p['osVersionNumber'] != config['os_version']
        or p['developerModeStatus'] != 'enabled' or p['ddiServicesAvailable'] is not True
        or c['pairingState'] != 'paired' or c['transportType'] != 'wired' or c['tunnelState'] != 'connected'):
        raise ValueError('Physical device identity/developer services/connection differs')
    return dict(product_type=h['productType'],marketing_name=h['marketingName'],reality=h['reality'],
                os_version=p['osVersionNumber'],developer_mode='enabled',ddi_services=True,transport='wired')


def admit_profile(profile, entitlements, signature, config, now=None):
    team = config['development_team']; application = team + '.' + BUNDLE
    now = now or datetime.datetime.now(datetime.timezone.utc)
    expiry = profile['ExpirationDate'].replace(tzinfo=datetime.timezone.utc)
    if (profile['TeamIdentifier'] != [team] or profile['ApplicationIdentifierPrefix'] != [team]
        or config['device_udid'] not in profile.get('ProvisionedDevices',[]) or expiry <= now
        or profile['Entitlements'].get('application-identifier') != application
        or profile['Entitlements'].get('com.apple.developer.team-identifier') != team
        or profile['Entitlements'].get('get-task-allow') is not True):
        raise ValueError('Development provisioning does not admit selected team/app/device')
    if (entitlements.get('application-identifier') != application
        or entitlements.get('com.apple.developer.team-identifier') != team
        or entitlements.get('get-task-allow') is not True
        or re.findall(r'^TeamIdentifier=(.*)$',signature,re.M) != [team]
        or re.findall(r'^Identifier=(.*)$',signature,re.M) != [BUNDLE]):
        raise ValueError('Signed application entitlements/team differs')
    return dict(development=True,device_in_profile=True,signature_team_matches=True,
                application_matches=True,expires_utc=expiry.isoformat())


def inventory(app):
    files = {}
    for p in sorted(app.rglob('*')):
        if p.is_symlink(): raise ValueError('Signed bundle contains symlink')
        if p.is_file(): files[str(p.relative_to(app))] = package.sha(p)
    required = {'TokletHarness','Info.plist','PkgInfo','Tiny/config.json','Tiny/model.safetensors',
                'embedded.mobileprovision','_CodeSignature/CodeResources'}
    if set(files) != required: raise ValueError('Unexpected signed app inventory')
    return files


def check_console(text, backend, files):
    results = [json.loads(s.removeprefix('TOKLET_SELFTEST_JSON=')) for s in text.splitlines() if s.startswith('TOKLET_SELFTEST_JSON=')]
    proofs = [json.loads(s.removeprefix('TOKLET_INSTALLATION_JSON=')) for s in text.splitlines() if s.startswith('TOKLET_INSTALLATION_JSON=')]
    if len(results) != 1 or len(proofs) != 2: raise ValueError('Expected one result and two installed bundle proofs')
    result = results[0]
    observed = [tuple(s[k] for k in ('name','history','evaluated','pending','saved','reason')) for s in result['states']]
    if (observed != STATES or result['passed'] is not True or result['backend'] != backend
        or result['tokens'] != [47]*4 or result['closed'] is not True or result['released'] is not True
        or result['physicalDeviceQualified'] is not False):
        raise ValueError('Physical tiny backend/oracle/state/close differs')
    if ([p['phase'] for p in proofs] != ['before','after']
        or any(p['simulator'] is not False or p['files'] != files for p in proofs)):
        raise ValueError('Installed physical bundle differs from signed source')
    return dict(result=result,installed_proofs=proofs)


def remote_path(value):
    path = unquote(urlparse(value).path) if value.startswith('file:') else value
    return path.removeprefix('/private') if path.startswith('/private/var/') else path


def installed_path(data):
    rows = data['result']['installedApps']
    if len(rows) != 1 or rows[0]['bundleID'] != BUNDLE: raise ValueError('Installed app identity differs')
    path = remote_path(rows[0]['installationURL'])
    if not path.startswith('/var/containers/Bundle/Application/') or not path.endswith('/TokletHarness.app'):
        raise ValueError('Unexpected installed app location')
    return path


def owned_processes(data, app):
    rows = data['result']['runningProcesses']; found = []
    if not isinstance(rows,list): raise ValueError('Malformed remote process inventory')
    for row in rows:
        executable = remote_path(row['executable'])
        if executable == app + '/TokletHarness':
            pid = row['processIdentifier']
            if type(pid) is not int or pid <= 0: raise ValueError('Invalid remote process ID')
            found.append(pid)
        else:
            raise ValueError('Unowned process matched TokletHarness query')
    return found


def check_launch(data, app, config):
    result=data['result']; process=result['process']; termination=result['terminationResult']
    pid=process['processIdentifier']
    if (data['info']['outcome']!='success' or result['deviceIdentifier']!=config['device_identifier']
        or remote_path(process['executableURL'])!=app+'/TokletHarness'
        or type(pid) is not int or pid<=0
        or termination.get('terminationReason')!='exit' or type(termination.get('exitCode')) is not int
        or termination['exitCode']!=0):
        raise ValueError('Remote application launch/termination differs')
    return dict(pid=pid,executable_matches=True,termination_reason='exit',exit_code=0)


def stop_remote(query, signal, app):
    # Re-query the exact installed executable; never signal a remembered PID
    # after it could have been reused by a different phone application.
    observed = []; sent = []; errors = []
    for sig in ('SIGTERM','SIGKILL'):
        try:pids = owned_processes(query(),app)
        except BaseException as error:
            errors.append('Query before '+sig+': '+type(error).__name__+': '+str(error));continue
        observed.append(pids)
        if not pids:return dict(absent=True,observations=observed,signals=sent,errors=errors)
        for pid in pids:
            sent.append(dict(pid=pid,signal=sig))
            try:signal(pid,sig)
            except BaseException as error:errors.append(sig+': '+type(error).__name__+': '+str(error))
    absent=False
    try:
        pids = owned_processes(query(),app); observed.append(pids);absent=not pids
        if pids:errors.append('Owned phone process remains after bounded termination')
    except BaseException as error:errors.append('Final absence query: '+type(error).__name__+': '+str(error))
    return dict(absent=absent,observations=observed,signals=sent,errors=errors)


def redact_record(record, config, remote=None):
    text=json.dumps(record)
    for key in ('device_identifier','device_udid','development_team'):
        if key in config:text=text.replace(config[key],'<private-'+key+'>')
    if remote:text=text.replace(remote,'<installed-app>')
    return json.loads(text)


def main(native, expected, private_selection, selection_sha, output):
    native=native.resolve(); output=output.resolve(); private_selection=private_selection.resolve()
    if any(not p.is_relative_to(ROOT/'artifacts') for p in (native,output,private_selection)):
        raise ValueError('Repository artifact paths required')
    output.mkdir(); logs=output/'private-logs'; logs.mkdir(); temporary=output/'tmp'; temporary.mkdir()
    before={}; commands=[]; config={}; device=None; profile_result=None; files={}; results={}; cleanup=[]
    app=None; remote=None; link=None; primary=None; endpoint_errors=[]; cleanup_errors=[]; env={}; counter=0
    previous=package.signal.signal(package.signal.SIGTERM,package.interrupted)
    def execute(command,label,timeout=90):
        row=package.owned(command,logs/(label+'.stdout'),logs/(label+'.stderr'),env,timeout); commands.append(row)
        if row['primary_error'] or row['cleanup_errors']: raise RuntimeError(label+': '+str(row['primary_error'])+' '+repr(row['cleanup_errors']))
        return ROOT/row['stdout']
    def device_command(parts,label):
        return execute([DEVICECTL,'device',*parts,'--device',config['device_identifier'],'--timeout','45','--quiet','--json-output','-'],label,60)
    def query():
        nonlocal counter
        counter+=1
        return json.loads(package.small(device_command(['info','processes','--search','TokletHarness'],'processes-'+str(counter))))
    def signal(pid,sig):
        nonlocal counter
        counter+=1
        device_command(['process','signal','--pid',str(pid),'--signal',sig],'signal-'+str(counter))
    def clean():
        result=stop_remote(query,signal,remote)
        cleanup.append(result);cleanup_errors.extend(result['errors'])
        if not result['absent']:cleanup_errors.append('Remote process absence not verified')
    try:
        env,_=package.build_environment();env['TMPDIR']=str(temporary)
        config=selection(private_selection,selection_sha)
        integration.merge(before,integration.admit_build(native,expected))
        integration.merge(before,{str(private_selection.relative_to(ROOT)):selection_sha})
        paths=[*sorted((ROOT/'apple/Harness').glob('*')),ROOT/'apple/tests/swift/WorkerChecks.swift',
               *sorted((ROOT/'apple/TokletHarness.xcodeproj').rglob('project.pbxproj')),
               *sorted((ROOT/'apple/TokletHarness.xcodeproj').rglob('*.xcscheme')),
               *sorted((ROOT/'apple/TokletHarness.xcodeproj').rglob('*.xcworkspacedata')),
               Path(__file__).resolve(),Path(__file__).with_name('test_m7_physical_integration.py'),
               Path(builder.__file__).resolve(),Path(integration.__file__).resolve()]
        integration.merge(before,{str(p.relative_to(ROOT)):package.sha(p) for p in paths if p.is_file()})
        integration.merge(before,integration.FIXTURE)
        for name,pin in before.items():
            if package.sha(ROOT/name)!=pin: raise ValueError('Changed admitted input: '+name)
        integration.merge(before,builder.prepare(native));link=builder.bind_library_link(native)
        device=admit_device(json.loads(package.small(device_command(['info','details'],'device-before'))),config)
        if query()['result']['runningProcesses']: raise ValueError('Existing harness process is not owned by this run')
        command=['xcodebuild','-project',str(ROOT/'apple/TokletHarness.xcodeproj'),'-scheme','TokletHarness','-configuration','Release',
                 '-sdk','iphoneos','-destination','id='+config['device_udid'],'-derivedDataPath',str(output/'derived'),
                 '-disableAutomaticPackageResolution','-allowProvisioningUpdates','-allowProvisioningDeviceRegistration',
                 'DEVELOPMENT_TEAM='+config['development_team'],'CODE_SIGN_STYLE=Automatic','ARCHS=arm64',
                 'SDK_STAT_CACHE_DIR='+str(output/'SDKStatCaches'),'CLANG_MODULE_CACHE_PATH='+str(output/'ModuleCache'),
                 'COMPILER_INDEX_STORE_ENABLE=NO','build']
        builder.verify_library_link(link); execute(command,'signed-release-build',900)
        app=output/'derived/Build/Products/Release-iphoneos/TokletHarness.app'
        executable=builder.inspect_app(app,'iPhoneOS')
        if package.small(execute(['xcrun','lipo','-archs',str(executable)],'architecture')).strip()!='arm64': raise ValueError('Wrong app architecture')
        package.check_load_commands(package.small(execute(['xcrun','otool','-l',str(executable)],'deployment')),package.SPECS[1])
        execute(['codesign','--verify','--strict',str(app)],'signature-verify')
        cms=execute(['security','cms','-D','-i',str(app/'embedded.mobileprovision')],'provisioning')
        entitlements=execute(['codesign','-d','--entitlements',':-',str(app)],'entitlements')
        execute(['codesign','-dvv',str(app)],'signature-details')
        signature=package.small(logs/'signature-details.stderr')
        profile_result=admit_profile(plistlib.loads(package.small(cms).encode()),plistlib.loads(package.small(entitlements).encode()),signature,config)
        files=inventory(app)
        device=admit_device(json.loads(package.small(device_command(['info','details'],'device-preinstall'))),config)
        installed=device_command(['install','app',str(app)],'install')
        remote=installed_path(json.loads(package.small(installed)))
        for backend in ('cpu','metal'):
            metadata=logs/(backend+'-launch.json')
            args=[DEVICECTL,'device','process','launch','--device',config['device_identifier'],'--terminate-existing','--console',
                  '--timeout','180','--json-output',str(metadata),BUNDLE,'--','--self-test']
            if backend=='metal':args.append('--metal')
            raw=execute(args,backend+'-self-test',210)
            launch=check_launch(json.loads(package.small(metadata)),remote,config)
            results[backend]=check_console(package.small(raw),backend,files)
            results[backend]['launch']=launch
            clean()
            if cleanup_errors:raise RuntimeError('Remote cleanup reported errors')
    except BaseException as error: primary=type(error).__name__+': '+str(error)
    finally:
        if remote is not None:
            try:clean()
            except BaseException as error:cleanup_errors.append(type(error).__name__+': '+str(error))
        if config:
            try:admit_device(json.loads(package.small(device_command(['info','details'],'device-after'))),config)
            except BaseException as error:endpoint_errors.append('Device endpoint: '+type(error).__name__+': '+str(error))
        if link:
            try:builder.verify_library_link(link)
            except (ValueError,OSError) as error:endpoint_errors.append(str(error))
        for name,pin in before.items():
            try:
                if package.sha(ROOT/name)!=pin:endpoint_errors.append('Changed endpoint: '+name)
            except OSError:endpoint_errors.append('Unavailable endpoint: '+name)
        if files:
            try:
                if inventory(app)!=files:endpoint_errors.append('Signed bundle changed')
            except (ValueError,OSError) as error:endpoint_errors.append(str(error))
        package.signal.signal(package.signal.SIGTERM,previous)
    passed=primary is None and not cleanup_errors and not endpoint_errors and set(results)=={'cpu','metal'} and len(cleanup)==3
    raw_record=dict(commands=commands,config=config,remote_app=remote,primary_error=primary,cleanup_errors=cleanup_errors,endpoint_errors=endpoint_errors)
    raw_path=output/'private-execution.json'
    raw_path.write_text(json.dumps(raw_record,indent=2)+'\n')
    log_pins={str(p.relative_to(ROOT)):package.sha(p) for p in sorted(logs.iterdir()) if p.is_file()}
    log_pins[str(raw_path.relative_to(ROOT))]=package.sha(raw_path)
    record=dict(format='toklet-m7-physical-tiny-integration-v1',passed=passed,device=device,signing=profile_result,
                reviewed_native_package_sha256=expected,private_selection_sha256=selection_sha,bindings=before,
                signed_bundle=files,results=results,remote_cleanup=cleanup,library_link=link,private_evidence_pins=log_pins,
                commands=commands,primary_error=primary,endpoint_errors=endpoint_errors,cleanup_errors=cleanup_errors,
                physical_device_qualified=False,m7_complete=False)
    record=redact_record(record,config,remote)
    (output/'integration.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(passed=passed,primary_error=record['primary_error'],cleanup_errors=record['cleanup_errors'],endpoint_errors=record['endpoint_errors'])))
    return 0 if passed else 1


if __name__=='__main__':
    sys.exit(main(Path(sys.argv[1]),sys.argv[2],Path(sys.argv[3]),sys.argv[4],Path(sys.argv[5])))
