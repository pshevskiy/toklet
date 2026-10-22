"""Explicitly reset only the freshly identified installed development harness.

This is a preflight reset authorized by the physical test request, never a claim
that an already running process was launched/owned by the new trial. No install,
signing, unlock or application-data deletion is performed.
"""
import argparse
import json
import os
import signal
from pathlib import Path

import m7_apple_package as package
import m7_physical_integration as physical

ROOT=package.ROOT


def response(data,selected,command):
    key='identifier' if command=='info.details' else 'deviceIdentifier'
    if (data['info']['outcome']!='success' or data['info']['commandType']!='devicectl.device.'+command
        or data['result'][key]!=selected['device_identifier']):
        raise ValueError('Reset command outcome/device differs')
    return data


def installed(data,selected):
    response(data,selected,'info.apps')
    rows=data['result']['apps']
    if type(rows) is not list or len(rows)!=1:raise ValueError('One installed harness required')
    remote=physical.remote_path(rows[0]['url']).rstrip('/')
    if '..' in remote.split('/') or not remote.startswith('/var/containers/Bundle/Application/') or not remote.endswith('/TokletHarness.app'):
        raise ValueError('Unexpected installed harness location')
    physical.launch_verified(lambda:data,lambda:True,remote,selected)
    return remote


def reset(current,query,send,selected):
    remote=installed(current(),selected)
    def check():
        if installed(current(),selected)!=remote:raise ValueError('Installed harness changed during reset')
        return physical.owned_processes(response(query(),selected,'info.processes'),remote)
    def inventory():
        # stop_remote consumes the complete exact-path list, not remembered PIDs.
        pids=check()
        return dict(result=dict(runningProcesses=[dict(executable=remote+'/TokletHarness',processIdentifier=pid) for pid in pids]))
    def signal(pid,sig):
        if pid not in check():raise ValueError('Reset PID changed before signal')
        data=response(send(pid,sig),selected,'process.signal')['result']
        process=data['process']
        if (type(process['processIdentifier']) is not int or process['processIdentifier']!=pid
            or physical.remote_path(process['executable'])!=remote+'/TokletHarness'
            or type(data['signal']['value']) is not int
            or data['signal']!={'name':sig,'value':15 if sig=='SIGTERM' else 9}):
            raise ValueError('Reset signal target or signal differs')
    result=physical.stop_remote(inventory,signal,remote)
    if installed(current(),selected)!=remote:raise ValueError('Installed harness endpoint changed')
    return result,remote


def run(selection,selection_sha,output):
    if os.path.lexists(output):raise FileExistsError('Never overwrite a reset output')
    selection=selection.resolve();output=output.resolve()
    if any(not p.is_relative_to(ROOT/'artifacts') for p in [selection,output]):raise ValueError('Repository artifacts required')
    output.mkdir();logs=output/'private-logs';logs.mkdir();env,_=package.build_environment()
    pins={};commands=[];captures=[];capture_errors=[];selected={};remote=None;result=None;primary=None;endpoints=[]
    previous=signal.signal(signal.SIGTERM,package.interrupted)
    def guard():
        for name,pin in pins.items():
            if package.sha(ROOT/name)!=pin:raise ValueError('Reset input changed: '+name)
    def publish(path,value):
        raw=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
        with path.open('xb') as stream:stream.write(raw)
        pins[str(path.relative_to(ROOT))]=package.hashlib.sha256(raw).hexdigest()
    def command(parts,label):
        guard()
        row=package.owned([physical.DEVICECTL,'device',*parts,'--device',selected['device_identifier'],
            '--timeout','45','--quiet','--json-output','-'],logs/(label+'.stdout'),logs/(label+'.stderr'),env,60)
        commands.append(row);buffers={};errors=[]
        for key in ['stdout','stderr']:
            name=row[key]
            try:
                with (ROOT/name).open('rb') as stream:raw=stream.read(16*1024**2+1)
                if len(raw)>16*1024**2:raise ValueError('Reset diagnostic exceeds bound')
                digest=package.hashlib.sha256(raw).hexdigest();pins[name]=digest;buffers[key]=raw
                captures.append(dict(path=name,sha256=digest,bytes=len(raw)))
            except BaseException as error:
                errors.append(dict(path=name,error=type(error).__name__+': '+str(error)))
        capture_errors.extend(errors)
        if row['primary_error'] or row['cleanup_errors']:raise RuntimeError('Reset owned failure: '+label+': '+str(row['primary_error'])+' '+repr(row['cleanup_errors']))
        if errors:raise ValueError('Reset diagnostic capture failed: '+label)
        guard()
        return response(json.loads(buffers['stdout']),selected,'.'.join(parts[:2]))
    def current():return command(['info','apps','--bundle-id',physical.BUNDLE],'apps-'+str(len(commands)))
    def query():return command(['info','processes','--search','TokletHarness'],'processes-'+str(len(commands)))
    def send(pid,sig):return command(['process','signal','--pid',str(pid),'--signal',sig],'signal-'+str(len(commands)))
    try:
        selected=physical.selection(selection,selection_sha)
        paths=[selection,Path(__file__).resolve(),Path(package.__file__).resolve(),Path(physical.__file__).resolve(),
               Path(physical.integration.__file__).resolve(),Path(physical.builder.__file__).resolve()]
        pins={str(p.relative_to(ROOT)):package.sha(p) for p in paths}
        pins[str(selection.relative_to(ROOT))]=selection_sha
        physical.admit_device(command(['info','details'],'device-before'),selected)
        result,remote=reset(current,query,send,selected)
        if not result['absent'] or result['errors']:raise ValueError('Reset absence unavailable or termination error retained')
        physical.admit_device(command(['info','details'],'device-after'),selected)
    except BaseException as error:primary=type(error).__name__+': '+str(error)
    finally:
        signal.signal(signal.SIGTERM,previous)
        try:guard()
        except BaseException as error:endpoints.append(type(error).__name__+': '+str(error))
    # Immutable original failure ledger. Evidence pins describe consumed bytes,
    # never a fresh re-hash that could bless a replacement after admission.
    private=output/'private-execution.json'
    publish(private,dict(commands=commands,captures=captures,capture_errors=capture_errors,
                         result=result,remote=remote,primary_error=primary,endpoint_errors=list(endpoints)))
    try:guard()
    except BaseException as error:endpoints.append('After private publication: '+type(error).__name__+': '+str(error))
    record=physical.redact_record(dict(format='toklet-m8-explicit-harness-reset-v1',passed=primary is None and not endpoints and not capture_errors,
        preflight_reset=True,previous_process_launched_by_this_trial=False,bindings=dict(pins),
        commands=commands,capture_errors=capture_errors,result=result,primary_error=primary,endpoint_errors=list(endpoints)),selected,remote)
    public=output/'reset.json';publish(public,record)
    late=[]
    try:guard()
    except BaseException as error:late.append('After public publication: '+type(error).__name__+': '+str(error))
    if late:
        # Preserve the original public/private bytes; a late-cause journal and
        # returned failure supersede the draft success. Callers must recheck pins.
        journal=output/'private-late-failure.json';publish(journal,dict(primary_error=primary,endpoint_errors=endpoints,late_errors=late,capture_errors=capture_errors,original_public_sha256=pins[str(public.relative_to(ROOT))]))
        record=dict(record,passed=False,endpoint_errors=[*endpoints,*late],late_failure_receipt=dict(path=str(journal.relative_to(ROOT)),sha256=pins[str(journal.relative_to(ROOT))]))
    print(json.dumps(dict(passed=record['passed'],primary_error=record['primary_error'],endpoint_errors=record['endpoint_errors'])))
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('selection',type=Path);parser.add_argument('selection_sha');parser.add_argument('output',type=Path)
    args=parser.parse_args();raise SystemExit(0 if run(args.selection,args.selection_sha,args.output)['passed'] else 1)
