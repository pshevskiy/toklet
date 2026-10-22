"""Explicit sealed MLX jobs on the admitted signed iPhone; no signing or downloads."""
import argparse
import json
import os
from pathlib import Path
import plistlib
import signal
import sys
import time

import m7_apple_package as package
import m7_apple_integration as integration
import m7_physical_integration as physical
import m7_physical_measurement as measurement
import m7_physical_quality as quality
import m7_physical_storage as storage
import m7_physical_tasks as tasks
import m8_mlx_jobs as jobs
import m8_physical_state as state

ROOT=package.ROOT
NATIVE=ROOT/'artifacts/m8-apple-package-v1'
NATIVE_SHA='00d3701912b3371a70485bf4cff7bd4d8217db0d99e53206274c17d47095bee5'
PREPARED=ROOT/'artifacts/m8-phone-prepared-v2'
PREPARED_SHA='30a0db2db2f12edfd335424b9b7c58694bd486494b3a1e3d1d64099a4bbb0c88'
SELECTION=ROOT/'artifacts/m7-physical-admission-v1/private-selection.json'
SELECTION_SHA='711ef5916ad223096bcb91cd93d7814f2c44056ab7494882d87af7d42a9804db'
FORMAT='toklet-m8-physical-mlx-v1'
READY_FORMAT='toklet-m8-physical-mlx-ready-v1'


def capture(path,pin,maximum=16*1024**2):
    with path.open('rb') as stream:raw=stream.read(maximum+1)
    if len(raw)>maximum or package.hashlib.sha256(raw).hexdigest()!=pin:raise ValueError('Consumed metadata differs: '+str(path))
    return raw


def metadata(path,pin,maximum=16*1024**2):
    return json.loads(capture(path,pin,maximum))


def admitted_stage(folder,expected):
    folder=folder.resolve()
    if not folder.is_relative_to(ROOT/'artifacts'):raise ValueError('Repository stage required')
    record=metadata(folder/'staged.json',expected)
    mode,config,host,pins,_=jobs.job(record['model'],record['workload'])
    if (record['format']!='toklet-m8-mlx-staged-job-v1' or record['passed'] is not True
        or mode not in ['measurement','quality','tasks','state'] or record['mode']!=mode or jobs.encoded(record['config'])!=jobs.encoded(config) or record['host']!=host
        or any(record[key] is not False for key in ['full_quality_qualified','physical_device_qualified','performance_qualified'])):
        raise ValueError('Exact supported staged MLX job required')
    for name,pin in pins.items():
        if record['bindings'].get(name)!=pin:raise ValueError('Staged recipe binding differs')
    measurement.verify_inputs(record['bindings'])
    receipt=record['job'];name=receipt['relative_path'];digest=receipt['sha256']
    if not storage.digest(digest) or name!='M7Jobs/'+digest+'.json' or jobs.encoded(metadata(folder/'transfer'/name,digest,512*1024))!=jobs.encoded(config):
        raise ValueError('Staged job bytes differ')
    payload,total=storage.inventory(folder/'transfer',config,name,digest)
    inventory={name:dict(bytes=item['bytes'],sha256=item['sha256']) for name,item in payload.items()}
    if inventory!=record['staged_files']:raise ValueError('Complete staged inventory differs')
    merged=dict(record['bindings']);integration.merge(merged,{str((folder/'staged.json').relative_to(ROOT)):expected})
    integration.merge(merged,{item['source']:item['sha256'] for item in payload.values()})
    return record,merged,payload,total


def checked_console(text,record,signed):
    config=record['config'];mode=record['mode']
    if mode=='measurement':return measurement.validate_console(text,config,signed)
    if mode=='state':return state.validate_console(text,config,signed)
    base=ROOT/record['transfer_root']
    if mode=='quality':
        raw=jobs.captured(str((base/config['tokensFile']).relative_to(ROOT)),config['tokensSHA256'],40004)
        import struct
        tokens=list(struct.unpack('<'+str(len(raw)//4)+'I',raw))
        return quality.check_console(text,config,signed,tokens,64 if record['model']=='tiny' else 151936)
    data=metadata(base/config['tasksFile'],config['tasksSHA256'],512*1024)
    return tasks.check_console(text,config,signed,data)


def admit_storage(folder,expected,context):
    folder=folder.resolve()
    if not folder.is_relative_to(ROOT/'artifacts'):raise ValueError('Repository proof required')
    proof=metadata(folder/'measurement.json',expected)
    if (proof['format']!=FORMAT or proof['passed'] is not True or proof['model']!='tiny' or proof['workload']!='measurement-short'
        or proof['mode']!='measurement' or proof['prepared_receipt']!=context['prepared_receipt'] or proof['signed_bundle']!=context['signed_bundle']
        or proof['native_package_sha256']!=NATIVE_SHA or proof['bindings'].get(str(SELECTION.relative_to(ROOT)))!=SELECTION_SHA
        or proof['primary_error'] or proof['endpoint_errors'] or proof['cleanup_errors'] or len(proof['remote_cleanup'])!=2
        or any(not row['absent'] or row['errors'] for row in proof['remote_cleanup']) or proof['physical_device_qualified'] is not False):
        raise ValueError('Same-device successful tiny MLX observation required')
    for name,pin in context['required_bindings'].items():
        if proof['bindings'].get(name)!=pin:raise ValueError('Storage context binding differs')
    if jobs.encoded(proof['config'])!=jobs.encoded(jobs.job('tiny','measurement-short')[1]):raise ValueError('Exact tiny observation config required')
    start=proof['measurement_host_start'];boot=proof['host_boot_token']
    if (set(start)!= {'wallUnixNanoseconds','monotonicNanoseconds'} or any(not storage.unsigned(v) or v==0 for v in start.values())
        or not storage.digest(boot)):raise ValueError('Bounded observation clocks required')
    measurement.verify_inputs(proof['bindings']);measurement.verify_inputs(proof['private_evidence_pins'])
    paths=[folder/'private-logs'/name for name in ['measurement.stdout','measurement.stderr','launch.json','host-boot.stdout']]
    if any(str(path.relative_to(ROOT)) not in proof['private_evidence_pins'] for path in paths):raise ValueError('Unbound storage observation')
    buffers=[capture(path,proof['private_evidence_pins'][str(path.relative_to(ROOT))]) for path in paths]
    checked=measurement.validate_console(buffers[0].decode()+'\n'+buffers[1].decode(),proof['config'],context['signed_bundle'])
    checked['launch']=physical.check_launch(json.loads(buffers[2]),context['remote'],context['selected'])
    if storage.boot_token(buffers[3].decode())!=boot:raise ValueError('Storage boot token differs')
    if checked!=proof['checked']:raise ValueError('Storage observation replay differs')
    pins=dict(proof['bindings']);integration.merge(pins,proof['private_evidence_pins'])
    integration.merge(pins,{str((folder/'measurement.json').relative_to(ROOT)):expected})
    return proof,pins


def admit_command(data,selected):
    if data['info']['outcome']!='success' or data['result']['deviceIdentifier']!=selected['device_identifier']:
        raise ValueError('Device command outcome/identity differs')
    return data


def validate_payload_listing(data,selected,name,size,*,allow_others=False):
    admit_command(data,selected);result=data['result']
    if result['domain']!='appDataContainer' or result['domainIdentifier']!=physical.BUNDLE:
        raise ValueError('Auxiliary inventory domain/count differs')
    rows=[row for row in result['files'] if row['relativePath']==name] if allow_others else result['files']
    if len(rows)!=1:raise ValueError('Auxiliary inventory count differs')
    row=rows[0]
    if (row['name']!=name or row['relativePath']!=name or row['resources']['isSymbolicLink'] is not False
        or row['resources']['isDirectory'] is not False or type(row['metadata']['size']) is not int or row['metadata']['size']!=size):
        raise ValueError('Auxiliary payload inventory/size differs')


def run(staged,staged_sha,output,*,prepare_only=False,ready=None,ready_sha=None,proof=None,proof_sha=None):
    if os.path.lexists(output):raise FileExistsError('Never overwrite an existing physical trial')
    staged=staged.resolve();output=output.resolve()
    if not output.is_relative_to(ROOT/'artifacts'):raise ValueError('Repository output required')
    if (ready is None)!=(ready_sha is None) or (proof is None)!=(proof_sha is None) or (ready and (prepare_only or proof)):
        raise ValueError('Explicit transfer/storage options conflict')
    output.mkdir();logs=output/'private-logs';logs.mkdir();commands=[];pins={};selected={};remote=None;app=None;signed={};device=None;profile=None
    checked=None;config={};record={};transfer_verified=False;remote_admitted=False;primary=None;endpoints=[];cleanup=[];cleanup_errors=[];counter=0
    storage_admission=None;host_boot=None;host_start=None;observed=None
    env,_=package.build_environment();env['TMPDIR']=str(output);previous=signal.signal(signal.SIGTERM,package.interrupted)
    def merge(extra):integration.merge(pins,extra)
    def guard():measurement.verify_inputs(pins)
    def execute(args,label,timeout=90,admit=True):
        if admit:guard()
        if label.startswith('copy-') and storage_admission:
            storage_admission['checks'].append(storage.fresh(storage_admission,host_boot,time.time_ns(),time.monotonic_ns(),label))
        row=package.owned(args,logs/(label+'.stdout'),logs/(label+'.stderr'),env,timeout);commands.append(row)
        # All raw diagnostics stay private and are pinned again at completion.
        for key in ['stdout','stderr']:
            path=ROOT/row[key]
            if path.is_file():merge({row[key]:package.sha(path)})
        if row['exit_code']!=0 or row['primary_error'] or row['cleanup_errors']:raise RuntimeError('Owned device command failed: '+label)
        if admit:guard()
        return ROOT/row['stdout']
    def command(parts,label,timeout=90,admit=True):
        return execute([physical.DEVICECTL,'device',*parts,'--device',selected['device_identifier'],'--timeout',str(timeout),'--quiet','--json-output','-'],label,timeout+20,admit)
    def consume(path):return capture(path,pins[str(path.relative_to(ROOT))]).decode()
    def current():return json.loads(consume(command(['info','apps','--bundle-id',physical.BUNDLE],'current-'+str(len(commands)))))
    def query():
        nonlocal counter
        counter+=1;return admit_command(json.loads(consume(command(['info','processes','--search','TokletHarness'],'process-'+str(counter),admit=False))),selected)
    def send(pid,sig):
        nonlocal counter
        counter+=1;admit_command(json.loads(consume(command(['process','signal','--pid',str(pid),'--signal',sig],'signal-'+str(counter),admit=False))),selected)
    def clean():
        row=physical.stop_remote(query,send,remote);cleanup.append(row);cleanup_errors.extend(row['errors'])
        if not row['absent']:cleanup_errors.append('Owned remote absence unavailable')
    try:
        selected=physical.selection(SELECTION,SELECTION_SHA);merge({str(SELECTION.relative_to(ROOT)):SELECTION_SHA})
        merge(integration.admit_build(NATIVE,NATIVE_SHA))
        app,extra,remote,signed=physical.admit_reuse(PREPARED,PREPARED_SHA,selected,NATIVE_SHA);merge(extra)
        prepared=metadata(PREPARED/'integration.json',PREPARED_SHA)
        # The cached profile/signature are exact receipt-bound bytes; recheck expiry now.
        profile_paths=[PREPARED/'private-logs'/n for n in ['provisioning.stdout','entitlements.stdout','signature-details.stderr']]
        profile_buffers=[]
        for path in profile_paths:
            pin=prepared['private_evidence_pins'][str(path.relative_to(ROOT))]
            profile_buffers.append(capture(path,pin))
            merge({str(path.relative_to(ROOT)):pin})
        profile=physical.admit_profile(plistlib.loads(profile_buffers[0]),plistlib.loads(profile_buffers[1]),profile_buffers[2].decode(),selected)
        helpers={Path(module.__file__).resolve() for module in list(sys.modules.values()) if getattr(module,'__file__',None)
                 and str(module.__file__).endswith('.py') and Path(module.__file__).resolve().is_relative_to(ROOT/'tools')}
        helpers.add(Path(__file__).resolve());merge({str(path.relative_to(ROOT)):package.sha(path) for path in helpers})
        common=dict(pins)
        record,extra,payload,total=admitted_stage(staged,staged_sha);merge(extra);config=record['config']
        record=dict(record,transfer_root=str((staged/'transfer').relative_to(ROOT)))
        context=dict(prepared_receipt=dict(path=str((PREPARED/'integration.json').relative_to(ROOT)),sha256=PREPARED_SHA),signed_bundle=signed,required_bindings=common,remote=remote,selected=selected)
        if record['model']!='tiny' and not ready and not proof:raise ValueError('Real copy requires fresh same-app storage proof')
        if record['model']=='tiny' and proof:raise ValueError('Tiny transfer needs no real-model storage proof')
        host_boot=storage.boot_token(consume(execute(['/usr/sbin/sysctl','-n','kern.bootsessionuuid'],'host-boot')))
        device=physical.admit_device(json.loads(consume(command(['info','details'],'device-before'))),selected)
        physical.launch_verified(current,lambda:True,remote,selected);remote_admitted=True;clean()
        if cleanup_errors:raise ValueError('Preflight cleanup failed')
        if ready:
            ready=ready.resolve()
            if not ready.is_relative_to(ROOT/'artifacts'):raise ValueError('Repository ready receipt required')
            saved=metadata(ready/'measurement.json',ready_sha)
            if (saved['format']!=READY_FORMAT or saved['passed'] is not True or saved['native_package_sha256']!=NATIVE_SHA
                or saved['model']!=record['model'] or saved['workload']!=record['workload'] or saved['mode']!=record['mode'] or jobs.encoded(saved['config'])!=jobs.encoded(config) or saved['staged_receipt']!=dict(path=str((staged/'staged.json').relative_to(ROOT)),sha256=staged_sha)
                or saved['signed_bundle']!=signed or saved['prepared_receipt']!=context['prepared_receipt'] or saved['checked'] is not None
                or saved['primary_error'] or saved['endpoint_errors'] or saved['cleanup_errors'] or len(saved['remote_cleanup'])!=2
                or any(not row['absent'] or row['errors'] for row in saved['remote_cleanup'])
                or any(saved[key] is not False for key in ['full_quality_qualified','physical_device_qualified','performance_qualified'])):raise ValueError('Exact successful transfer-only receipt required')
            for name,pin in common.items():
                if saved['bindings'].get(name)!=pin:raise ValueError('Ready context binding differs')
            measurement.verify_inputs(saved['bindings']);measurement.verify_inputs(saved['private_evidence_pins'])
            merge(saved['bindings']);merge(saved['private_evidence_pins']);merge({str((ready/'measurement.json').relative_to(ROOT)):ready_sha})
            if record['model']!='tiny':
                previous_admission=saved['storage_admission'];reference=previous_admission['proof'];path=(ROOT/reference['path']).resolve()
                observed,extra=admit_storage(path.parent,reference['sha256'],context);merge(extra)
                if observed['device']!=device:raise ValueError('Ready storage device differs')
                original=storage.decision(observed,path.parent,reference['sha256'],payload,total)
                labels=['copy-Model']+(['copy-corpus'] if 'tokensFile' in config else [])+(['copy-tasks'] if 'tasksFile' in config else [])+['copy-job']
                old_checks=previous_admission['checks']
                if len(old_checks)!=len(labels):raise ValueError('Original copy storage checks missing')
                original['checks']=[storage.fresh(original,host_boot,row['wallUnixNanoseconds'],row['monotonicNanoseconds'],label) for row,label in zip(old_checks,labels)]
                if original!=previous_admission:raise ValueError('Original storage admission differs')
                storage_admission=original
            transfer_verified=True
        else:
            if proof:
                proof=proof.resolve();observed,extra=admit_storage(proof,proof_sha,context);merge(extra)
                if observed['device']!=device:raise ValueError('Storage device details differ')
                storage_admission=storage.decision(observed,proof,proof_sha,payload,total)
            parent=str(Path(config['modelFolder']).parent);base=staged/'transfer'
            physical.launch_verified(current,lambda:command(['copy','to','--source',str(base/config['modelFolder']),'--destination','Documents/'+config['modelFolder'],
                '--domain-type','appDataContainer','--domain-identifier',physical.BUNDLE],'copy-Model',600),remote,selected)
            listing=json.loads(consume(command(['info','files','--subdirectory','Documents/'+parent,'--recurse','--domain-type','appDataContainer','--domain-identifier',physical.BUNDLE],'model-listing')))
            measurement.validate_listing(listing,selected['device_identifier'],dict(config,host=record['host']),parent)
            for field,label in [('tokensFile','corpus'),('tasksFile','tasks')]:
                if field not in config:continue
                directory=str(Path(config[field]).parent)
                physical.launch_verified(current,lambda:command(['copy','to','--source',str(base/directory),'--destination','Documents/'+directory,'--domain-type','appDataContainer','--domain-identifier',physical.BUNDLE],'copy-'+label,90),remote,selected)
                listing=json.loads(consume(command(['info','files','--subdirectory','Documents/'+directory,'--recurse','--domain-type','appDataContainer','--domain-identifier',physical.BUNDLE],label+'-listing')))
                validate_payload_listing(listing,selected,Path(config[field]).name,payload[config[field]]['bytes'])
            physical.launch_verified(current,lambda:command(['copy','to','--source',str(base/'M7Jobs'),'--destination','Documents/M7Jobs','--domain-type','appDataContainer','--domain-identifier',physical.BUNDLE],'copy-job',90),remote,selected)
            listing=json.loads(consume(command(['info','files','--subdirectory','Documents/M7Jobs','--recurse','--domain-type','appDataContainer','--domain-identifier',physical.BUNDLE],'job-listing')))
            job_name=record['job']['relative_path']
            validate_payload_listing(listing,selected,Path(job_name).name,payload[job_name]['bytes'],allow_others=True)
            transfer_verified=True
        if not prepare_only:
            job=record['job'];launch=logs/'launch.json';limit=7200 if record['mode']!='measurement' else 3600
            args=[physical.DEVICECTL,'device','process','launch','--device',selected['device_identifier'],'--terminate-existing','--console','--timeout',str(limit),'--json-output',str(launch),physical.BUNDLE,'--',{'measurement':'--measure','quality':'--quality','tasks':'--tasks','state':'--state'}[record['mode']],job['relative_path'],job['sha256']]
            host_start=dict(wallUnixNanoseconds=time.time_ns(),monotonicNanoseconds=time.monotonic_ns())
            raw=physical.launch_verified(current,lambda:execute(args,'measurement',limit+30),remote,selected)
            with launch.open('rb') as stream:launch_raw=stream.read(16*1024**2+1)
            if len(launch_raw)>16*1024**2:raise ValueError('Launch metadata exceeds limit')
            merge({str(launch.relative_to(ROOT)):package.hashlib.sha256(launch_raw).hexdigest()})
            launch_check=physical.check_launch(json.loads(launch_raw),remote,selected)
            with raw.open('rb') as stream:stdout=stream.read(64*1024**2+1)
            with (logs/'measurement.stderr').open('rb') as stream:stderr=stream.read(64*1024**2+1)
            if max(len(stdout),len(stderr))>64*1024**2:raise ValueError('Physical console exceeds limit')
            if package.hashlib.sha256(stdout).hexdigest()!=pins[str(raw.relative_to(ROOT))] or package.hashlib.sha256(stderr).hexdigest()!=pins[str((logs/'measurement.stderr').relative_to(ROOT))]:raise ValueError('Consumed physical console differs')
            checked=checked_console(stdout.decode()+'\n'+stderr.decode(),record,signed);checked['launch']=launch_check
    except BaseException as error:primary=type(error).__name__+': '+str(error)
    finally:
        if remote_admitted:
            try:clean()
            except BaseException as error:cleanup_errors.append(type(error).__name__+': '+str(error))
        if selected and device is not None:
            try:physical.admit_device(json.loads(consume(command(['info','details'],'device-after',admit=False))),selected)
            except BaseException as error:endpoints.append('Device endpoint: '+type(error).__name__+': '+str(error))
        try:guard()
        except BaseException as error:endpoints.append(type(error).__name__+': '+str(error))
        if app:
            try:
                if physical.inventory(app)!=signed:endpoints.append('Signed bundle endpoint differs')
            except BaseException as error:endpoints.append('Signed endpoint: '+str(error))
        signal.signal(signal.SIGTERM,previous)
    private=output/'private-execution.json'
    with private.open('x') as stream:json.dump(dict(commands=commands,primary_error=primary,endpoint_errors=endpoints,cleanup_errors=cleanup_errors),stream,indent=2);stream.write('\n')
    private_pins={str(path.relative_to(ROOT)):package.sha(path) for path in [private,*logs.iterdir()] if path.is_file()}
    passed=primary is None and not endpoints and not cleanup_errors and transfer_verified and len(cleanup)==2 and (checked is None if prepare_only else checked is not None)
    result=dict(format=READY_FORMAT if prepare_only else FORMAT,passed=passed,model=record.get('model'),workload=record.get('workload'),mode=record.get('mode'),config=config,
                native_package_sha256=NATIVE_SHA,prepared_receipt=dict(path=str((PREPARED/'integration.json').relative_to(ROOT)),sha256=PREPARED_SHA),
                staged_receipt=dict(path=str((staged/'staged.json').relative_to(ROOT)),sha256=staged_sha),signed_bundle=signed,device=device,signing=profile,bindings=pins,
                private_evidence_pins=private_pins,commands=commands,remote_cleanup=cleanup,storage_admission=storage_admission,host_boot_token=host_boot,measurement_host_start=host_start,
                checked=checked,primary_error=primary,endpoint_errors=endpoints,cleanup_errors=cleanup_errors,full_quality_qualified=False,physical_device_qualified=False,performance_qualified=False)
    result=physical.redact_record(result,selected,remote)
    with (output/'measurement.json').open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(passed=passed,primary_error=result['primary_error'],endpoint_errors=result['endpoint_errors'],cleanup_errors=result['cleanup_errors'])))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--staged',type=Path,required=True);parser.add_argument('--staged-sha',required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--ready',type=Path);parser.add_argument('--ready-sha');parser.add_argument('--proof',type=Path);parser.add_argument('--proof-sha');args=parser.parse_args()
    result=run(args.staged,args.staged_sha,args.output,prepare_only=args.prepare_only,ready=args.ready,ready_sha=args.ready_sha,proof=args.proof,proof_sha=args.proof_sha)
    sys.exit(0 if result['passed'] else 1)
