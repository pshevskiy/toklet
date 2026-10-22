"""Explicit MLX same-process iOS scene interruption; no forced foreground relaunch.

The synchronous owner drains its own launcher before remote cleanup. Use directly
or through exec: a finite parent must not kill it during cooperative restoration.
"""
import json
import os
from pathlib import Path
import plistlib
import signal
import subprocess
import sys
import time

import m7_physical_lifecycle as lifecycle
import m8_physical_mlx as phone

ROOT=phone.ROOT
FORMAT='toklet-m8-physical-mlx-lifecycle-v1'


def validate_console(text,config,signed):
    # State and lifecycle share the exact native config schema and bounded shape.
    phone.state.admit_config(config)
    return lifecycle.validate_console(text,config,signed,backend='metal-mlx')


def admit_ready(folder,expected,staged,staged_sha,record,payload,total,context,boot):
    saved=phone.metadata(folder/'measurement.json',expected)
    if (saved['format']!=phone.READY_FORMAT or saved['passed'] is not True or saved['model']!='qwen8'
        or saved['workload']!='lifecycle-512' or saved['mode']!='lifecycle' or saved['checked'] is not None
        or phone.jobs.encoded(saved['config'])!=phone.jobs.encoded(record['config'])
        or saved['native_package_sha256']!=phone.NATIVE_SHA or saved['prepared_receipt']!=context['prepared_receipt']
        or saved['signed_bundle']!=context['signed_bundle']
        or saved['staged_receipt']!=dict(path=str((staged/'staged.json').relative_to(ROOT)),sha256=staged_sha)
        or saved['primary_error'] is not None or saved['endpoint_errors']!=[] or saved['cleanup_errors']!=[]
        or any(saved[k] is not False for k in ['full_quality_qualified','physical_device_qualified','performance_qualified'])
        or len(saved['remote_cleanup'])!=2 or any(row['absent'] is not True or row['errors']!=[] for row in saved['remote_cleanup'])):
        raise ValueError('Exact successful MLX lifecycle transfer-only receipt required')
    for name,pin in context['required_bindings'].items():
        if saved['bindings'].get(name)!=pin:raise ValueError('Lifecycle ready context differs')
    phone.measurement.verify_inputs(saved['bindings']);phone.measurement.verify_inputs(saved['private_evidence_pins'])
    admission=saved['storage_admission'];reference=admission['proof'];proof_path=(ROOT/reference['path']).resolve()
    if proof_path.name!='measurement.json' or not proof_path.is_relative_to(ROOT/'artifacts'):
        raise ValueError('Bounded original tiny storage proof required')
    proof,pins=phone.admit_storage(proof_path.parent,reference['sha256'],context)
    original=phone.storage.decision(proof,proof_path.parent,reference['sha256'],payload,total)
    if len(admission['checks'])!=2:raise ValueError('Exact original copy freshness checks required')
    original['checks']=[phone.storage.fresh(original,boot,row['wallUnixNanoseconds'],row['monotonicNanoseconds'],label)
                        for row,label in zip(admission['checks'],['copy-Model','copy-job'],strict=True)]
    if original!=admission or proof['device']!=saved['device']:raise ValueError('Lifecycle original storage decision differs')
    phone.integration.merge(pins,saved['bindings']);phone.integration.merge(pins,saved['private_evidence_pins'])
    phone.integration.merge(pins,{str((folder/'measurement.json').relative_to(ROOT)):expected})
    return saved,pins,original


def collect(staged,staged_sha,ready,ready_sha,output):
    staged=staged.resolve();ready=ready.resolve();output=output.absolute()
    if any(not path.is_relative_to(ROOT/'artifacts') for path in [staged,ready,output.resolve()]):
        raise ValueError('Repository artifact paths required')
    if os.path.lexists(output):raise FileExistsError('Never overwrite a lifecycle attempt')
    output.mkdir();logs=output/'private-logs';logs.mkdir()
    pins={};commands=[];selected={};remote=None;app=None;signed={};device=None;profile=None;config={};record={};storage_admission=None
    checked=None;phase=[];cleanups=[];primary=None;endpoints=[];cleanup_errors=[];host_boot=None
    process=None;live_row=None;out_handle=None;err_handle=None;launch_attempted=False;live_pid=None;ready_value=None;remote_admitted=False
    env,_=phone.package.build_environment();env['TMPDIR']=str(output)
    previous=signal.signal(signal.SIGTERM,phone.package.interrupted)
    def merge(extra):phone.integration.merge(pins,extra)
    def guard():phone.measurement.verify_inputs(pins)
    def capture(path,pin=None):
        with path.open('rb') as stream:raw=stream.read(64*1024**2+1)
        if len(raw)>64*1024**2:raise ValueError('Bounded lifecycle log exceeded')
        digest=phone.package.hashlib.sha256(raw).hexdigest()
        if pin is not None and digest!=pin:raise ValueError('Consumed lifecycle metadata differs')
        merge({str(path.relative_to(ROOT)):digest});return raw
    def execute(args,label,timeout=90,full_guard=True):
        if full_guard:guard()
        row=phone.package.owned(args,logs/(label+'.stdout'),logs/(label+'.stderr'),env,timeout);commands.append(row)
        buffers={};row['capture_errors']=[]
        for field in ['stdout','stderr']:
            try:buffers[field]=capture(ROOT/row[field])
            except BaseException as error:row['capture_errors'].append(field+': '+type(error).__name__+': '+str(error))
        if row['exit_code']!=0 or row['primary_error'] or row['cleanup_errors'] or row['capture_errors']:
            raise ValueError('Owned lifecycle command failed: '+label)
        if full_guard:guard()
        return buffers['stdout']
    def command(parts,label,full_guard=True):
        # devicectl treats values after the positional bundle as app arguments.
        args=[phone.physical.DEVICECTL,'device',*parts[:2],'--device',selected['device_identifier'],
              '--timeout','90','--quiet','--json-output','-',*parts[2:]]
        data=json.loads(execute(args,label,full_guard=full_guard))
        if parts[:2]==['info','details']:
            if data['info']['outcome']!='success' or data['info']['commandType']!='devicectl.device.info.details':
                raise ValueError('Successful typed device details required')
        else:phone.admit_command(data,selected)
        return data
    def current(label,full_guard=True):return command(['info','apps','--bundle-id',phone.physical.BUNDLE],label,full_guard)
    def query(label):
        data=command(['info','processes','--search','TokletHarness'],label,False)
        if data['info']['commandType']!='devicectl.device.info.processes':raise ValueError('Exact process query required')
        return data
    def same_pid(label):
        pids=phone.physical.owned_processes(query(label),remote)
        if len(pids)!=1 or live_pid is not None and pids!=[live_pid]:raise ValueError('Same live MLX lifecycle PID required')
        phase.append(dict(phase=label,pid=pids[0],observed_host_monotonic_nanoseconds=time.monotonic_ns()));return pids[0]
    def cleanup():
        counter=0
        def query_cleanup():
            nonlocal counter
            counter+=1;return query('cleanup-process-'+str(counter))
        def send(pid,sig):
            nonlocal counter
            counter+=1;data=command(['process','signal','--pid',str(pid),'--signal',sig],'cleanup-signal-'+str(counter),False)
            result=data['result'];process_result=result['process'];value=result['signal']
            if (data['info']['commandType']!='devicectl.device.process.signal'
                or type(process_result['processIdentifier']) is not int or process_result['processIdentifier']!=pid
                or phone.physical.remote_path(process_result['executable'])!=remote+'/TokletHarness'
                or set(value)!={'name','value'} or value['name']!=sig or type(value['value']) is not int
                or value['value']!={'SIGTERM':15,'SIGKILL':9}[sig]):raise ValueError('Exact cleanup signal result required')
        result=phone.physical.stop_remote(query_cleanup,send,remote);cleanups.append(result);cleanup_errors.extend(result['errors'])
        if not result['absent']:cleanup_errors.append('Lifecycle remote absence unverified')
    try:
        selected=phone.physical.selection(phone.SELECTION,phone.SELECTION_SHA);merge({str(phone.SELECTION.relative_to(ROOT)):phone.SELECTION_SHA})
        merge(phone.integration.admit_build(phone.NATIVE,phone.NATIVE_SHA))
        app,extra,remote,signed=phone.physical.admit_reuse(phone.PREPARED,phone.PREPARED_SHA,selected,phone.NATIVE_SHA);merge(extra)
        prepared=phone.metadata(phone.PREPARED/'integration.json',phone.PREPARED_SHA);buffers=[]
        for name in ['provisioning.stdout','entitlements.stdout','signature-details.stderr']:
            path=phone.PREPARED/'private-logs'/name;buffers.append(capture(path,prepared['private_evidence_pins'][str(path.relative_to(ROOT))]))
        profile=phone.physical.admit_profile(plistlib.loads(buffers[0]),plistlib.loads(buffers[1]),buffers[2].decode(),selected)
        merge(phone.shared_helper_bindings());common=dict(pins)
        helpers={Path(m.__file__).resolve() for m in list(sys.modules.values()) if getattr(m,'__file__',None)
                 and str(m.__file__).endswith('.py') and Path(m.__file__).resolve().is_relative_to(ROOT/'tools')}
        merge({str(path.relative_to(ROOT)):phone.package.sha(path) for path in helpers})
        record,extra,payload,total=phone.admitted_stage(staged,staged_sha);merge(extra);config=record['config']
        if record['mode']!='lifecycle' or record['model']!='qwen8' or record['workload']!='lifecycle-512':
            raise ValueError('Explicit selected eight-bit lifecycle512 stage required')
        phone.state.admit_config(config)
        host_boot=phone.storage.boot_token(execute(['/usr/sbin/sysctl','-n','kern.bootsessionuuid'],'host-boot').decode())
        context=dict(prepared_receipt=dict(path=str((phone.PREPARED/'integration.json').relative_to(ROOT)),sha256=phone.PREPARED_SHA),signed_bundle=signed,required_bindings=common,remote=remote,selected=selected)
        saved,extra,storage_admission=admit_ready(ready,ready_sha,staged,staged_sha,record,payload,total,context,host_boot);merge(extra)
        device=phone.physical.admit_device(command(['info','details'],'device-before'),selected)
        if device!=saved['device']:raise ValueError('Lifecycle same-device observation differs')
        phone.physical.launch_verified(lambda:current('current-install-before'),lambda:True,remote,selected);remote_admitted=True
        if phone.physical.owned_processes(query('preflight-process'),remote):raise ValueError('Natural preflight absence required; no lifecycle reset')
        cleanups.append(dict(absent=True,observations=[[]],signals=[],errors=[]));guard()
        job=record['job'];metadata=logs/'launch.json'
        args=[phone.physical.DEVICECTL,'device','process','launch','--device',selected['device_identifier'],
              '--console','--timeout','900','--json-output',str(metadata),phone.physical.BUNDLE,'--','--lifecycle',job['relative_path'],job['sha256']]
        out_handle=(logs/'lifecycle.stdout').open('xb');err_handle=(logs/'lifecycle.stderr').open('xb')
        live_row=dict(command=args,exit_code=None,primary_error=None,cleanup_errors=[],capture_errors=[],pid=None,
                      stdout=str((logs/'lifecycle.stdout').relative_to(ROOT)),stderr=str((logs/'lifecycle.stderr').relative_to(ROOT)))
        commands.append(live_row)
        process=subprocess.Popen(args,cwd=ROOT,env=env,stdout=out_handle,stderr=err_handle,start_new_session=True)
        live_row['pid']=process.pid;launch_attempted=True;deadline=time.monotonic()+300
        while ready_value is None:
            ready_value=lifecycle.peek_ready(logs/'lifecycle.stdout')
            if ready_value is not None:break
            if process.poll() is not None:raise ValueError('Lifecycle launcher exited before active READY')
            if time.monotonic()>=deadline:raise TimeoutError('Active READY deadline')
            time.sleep(.02)
        phase.append(dict(phase='ready-consumed',observed_host_monotonic_nanoseconds=time.monotonic_ns(),ready=ready_value))
        # No whole-model input hashing between active READY and OS activation.
        live_pid=same_pid('ready-process')
        lifecycle.activation(command(['process','launch','--activate','com.apple.Preferences'],'activate-settings',False),selected)
        phase.append(dict(phase='settings-activated',observed_host_monotonic_nanoseconds=time.monotonic_ns()))
        time.sleep(2);same_pid('background-process')
        phone.physical.launch_verified(lambda:current('current-install-return',False),lambda:True,remote,selected)
        lifecycle.activation(command(['process','launch','--activate',phone.physical.BUNDLE],'activate-harness',False),selected,remote,live_pid)
        phase.append(dict(phase='harness-activated',pid=live_pid,observed_host_monotonic_nanoseconds=time.monotonic_ns()))
        live_row['exit_code']=process.wait(timeout=600)
        if live_row['exit_code']!=0:live_row['primary_error']='tool exit '+str(live_row['exit_code']);raise ValueError('Lifecycle console child failed')
        guard();text=capture(logs/'lifecycle.stdout').decode()+'\n'+capture(logs/'lifecycle.stderr').decode()
        checked=validate_console(text,config,signed)
        checked['launch']=phone.physical.check_launch(json.loads(capture(metadata)),remote,selected)
        if checked['ready']!=ready_value or checked['launch']['pid']!=live_pid:raise ValueError('Original process/READY identity differs')
    except BaseException as error:primary=type(error).__name__+': '+str(error)
    finally:
        # Reap the host launcher before remote cleanup so it cannot launch late.
        if process is not None:
            try:phone.package.drain_group(process,live_row['cleanup_errors'],15)
            except BaseException as error:live_row['cleanup_errors'].append('Drain: '+type(error).__name__+': '+str(error))
            try:live_row['exit_code']=process.wait(timeout=15)
            except BaseException as error:live_row['cleanup_errors'].append('Reap: '+type(error).__name__+': '+str(error))
            if live_row['exit_code']!=0 and live_row['primary_error'] is None:live_row['primary_error']='tool exit '+str(live_row['exit_code'])
        if remote_admitted and launch_attempted:
            try:cleanup()
            except BaseException as error:cleanup_errors.append(type(error).__name__+': '+str(error))
        for handle in [out_handle,err_handle]:
            if handle is not None:
                try:handle.close()
                except BaseException as error:cleanup_errors.append('Capture close: '+type(error).__name__+': '+str(error))
        for path in sorted(logs.iterdir()):
            if path.is_file():
                try:capture(path)
                except BaseException as error:endpoints.append(type(error).__name__+': '+str(error))
        if device is not None:
            try:
                if phone.physical.admit_device(command(['info','details'],'device-after',False),selected)!=device:raise ValueError('Device endpoint differs')
            except BaseException as error:endpoints.append('Device endpoint: '+str(error))
        try:guard()
        except BaseException as error:endpoints.append(type(error).__name__+': '+str(error))
        if app is not None:
            try:
                if phone.physical.inventory(app)!=signed:raise ValueError('Signed app endpoint differs')
            except BaseException as error:endpoints.append('Signed endpoint: '+str(error))
        signal.signal(signal.SIGTERM,previous)
    private=output/'private-execution.json'
    private_raw=(json.dumps(dict(commands=commands,primary_error=primary,endpoint_errors=list(endpoints),cleanup_errors=list(cleanup_errors)),indent=2,allow_nan=False)+'\n').encode()
    with private.open('xb') as stream:stream.write(private_raw)
    private_name=str(private.relative_to(ROOT));private_pin=phone.package.hashlib.sha256(private_raw).hexdigest()
    merge({private_name:private_pin});private_pins={private_name:private_pin}
    for path in sorted(logs.iterdir()):
        if not path.is_file():continue
        name=str(path.relative_to(ROOT))
        if name in pins:private_pins[name]=pins[name]
        else:endpoints.append('Final log has no consumed binding: '+name)
    try:guard()
    except BaseException as error:endpoints.append('Final closure: '+type(error).__name__+': '+str(error))
    # Keep the original failure/command ledger immutable. Late causes have their
    # own serialized, hash-bound journal; log pins remain the consumed digests.
    late=output/'private-final-guard.json'
    late_raw=(json.dumps(dict(primary_error=primary,endpoint_errors=list(endpoints),cleanup_errors=list(cleanup_errors)),indent=2,allow_nan=False)+'\n').encode()
    with late.open('xb') as stream:stream.write(late_raw)
    late_name=str(late.relative_to(ROOT));late_pin=phone.package.hashlib.sha256(late_raw).hexdigest()
    merge({late_name:late_pin});private_pins[late_name]=late_pin
    natural=dict(absent=True,observations=[[]],signals=[],errors=[])
    passed=primary is None and not endpoints and not cleanup_errors and checked is not None and cleanups==[natural,natural] and all(row['exit_code']==0 and not row['primary_error'] and not row['cleanup_errors'] and not row['capture_errors'] for row in commands)
    result=dict(format=FORMAT,passed=passed,bindings=pins,private_evidence_pins=private_pins,config=config,device=device,signing=profile,signed_bundle=signed,
                native_package_sha256=phone.NATIVE_SHA,prepared_receipt=dict(path=str((phone.PREPARED/'integration.json').relative_to(ROOT)),sha256=phone.PREPARED_SHA),
                staged_receipt=dict(path=str((staged/'staged.json').relative_to(ROOT)),sha256=staged_sha),transfer_receipt=dict(path=str((ready/'measurement.json').relative_to(ROOT)),sha256=ready_sha),
                host_boot_token=host_boot,storage_admission=storage_admission,checked=checked,phase_observations=phase,commands=commands,remote_cleanup=cleanups,
                primary_error=primary,endpoint_errors=endpoints,cleanup_errors=cleanup_errors,actual_os_lifecycle_qualified=False,physical_device_qualified=False,performance_qualified=False,
                scope='Selected native MLX raw actual-scene lifecycle observation; independent replay required; warning remains synthetic')
    if selected:result=phone.physical.redact_record(result,selected,remote)
    else:result.update(primary_error='Initial admission failed; see hash-bound private ledger',endpoint_errors=['Endpoint admission failed; see private ledger'] if endpoints else [],commands=[])
    with (output/'lifecycle.json').open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(passed=passed,primary_error=result['primary_error'],endpoint_errors=result['endpoint_errors'],cleanup_errors=result['cleanup_errors'])));return result


if __name__=='__main__':
    if len(sys.argv)!=6:raise SystemExit('Expected staged stagedSHA ready readySHA output')
    receipt=collect(Path(sys.argv[1]),sys.argv[2],Path(sys.argv[3]),sys.argv[4],Path(sys.argv[5]))
    sys.exit(0 if receipt['passed'] else 1)
