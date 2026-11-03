"""Shared evidence and graceful VM-window lifetime helpers."""
from pathlib import Path
import hashlib
import json
import os
import signal
import stat
import subprocess
import time
import podman_machine_lifecycle as lifecycle
ROOT=Path(__file__).resolve().parents[2]
def require(condition,message):
    if not condition:raise ValueError(message)

def verify_file(path, expected):
    def fields(v):return v.st_dev,v.st_ino,v.st_mode,v.st_size,v.st_mtime_ns,v.st_ctime_ns
    fd = os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    with os.fdopen(fd,'rb') as file:
        before = os.fstat(file.fileno());digest = hashlib.sha256()
        size = expected['bytes']
        require(type(size) is int and 0<=size<=4*1024**3
            and stat.S_ISREG(before.st_mode) and before.st_size==size, 'regular admitted evidence file required')
        remaining = size
        while remaining:
            body = file.read(min(1024**2,remaining))
            require(body, 'admitted evidence truncated during hashing')
            remaining -= len(body);digest.update(body)
        require(not file.read(1) and digest.hexdigest()==expected['sha256'] and fields(os.fstat(file.fileno()))==fields(before)
            and fields(Path(path).lstat())==fields(before), 'collection/source evidence changed')


def selected(inventory):
    matches = [v for v in inventory if v.get('Name')==lifecycle.MACHINE]
    require(len(matches)==1, 'unique existing grader machine required')
    return matches[0]


def write_json(path, value):
    with Path(path).open('x') as file:json.dump(value,file,indent=2,ensure_ascii=False,allow_nan=False);file.write('\n')


def finish_decision(record, pending, guard):
    if (pending is None or record.get('failure') or record.get('restoration_failure')
            or record.get('restored_stopped') is not True or record.get('pipe_lifetime_finished') is not True):
        return
    try:
        guard();record['decision'] = pending;record['completed'] = True
    except BaseException as error:record['failure'] = dict(type=type(error).__name__,message=str(error)[:4096])


def supervise_window(command, stdout, stderr, env, observe, group_alive, timeout=3600):
    """Retain the lifetime owner through graceful cancellation and VM shutdown.

    The deadline requests SIGTERM, never SIGKILL. Continue waiting for the whole
    worker group, including cleanup after the direct time-wrapper process exits.
    Independently verify the original stopped VM state after the owner exits.
    """
    record = dict(command=command,exit_code=None,pid=None,primary_error=None,cleanup_errors=[],
        signals=[],lifetime_policy='graceful TERM only; retain owner until group exit; independently verify VM stopped')
    process = None;old = {}
    try:
        before = observe();record['machine_before'] = before;machine = selected(before)
        require(machine.get('Running') is False and machine.get('Starting') is False, 'outer window requires stopped VM')
        with Path(stdout).open('xb') as out,Path(stderr).open('xb') as err:
            process = subprocess.Popen(command,cwd=ROOT,env=env,stdout=out,stderr=err,start_new_session=True)
            record['pid'] = process.pid
            try:
                record['exit_code'] = process.wait(timeout=timeout)
                if record['exit_code']!=0:record['primary_error'] = 'tool exit '+str(record['exit_code'])
            except BaseException as error:
                record['primary_error'] = type(error).__name__+': '+str(error)
                old = {s:signal.signal(s,signal.SIG_IGN)for s in(signal.SIGINT,signal.SIGTERM)}
                try:os.killpg(process.pid,signal.SIGTERM);record['signals'].append('SIGTERM')
                except ProcessLookupError:pass
            finally:
                # No bounded forced-drain helper here: it could kill the VM
                # lifetime owner while its restoration finally block is active.
                if not old:old = {s:signal.signal(s,signal.SIG_IGN)for s in(signal.SIGINT,signal.SIGTERM)}
                while process.poll() is None or group_alive(process.pid):time.sleep(0.1)
                record['exit_code'] = process.wait()
    except BaseException as error:
        if record['primary_error'] is None:record['primary_error'] = type(error).__name__+': '+str(error)
    finally:
        try:
            after = observe();record['machine_after'] = after;machine = selected(after)
            require(machine.get('Running') is False and machine.get('Starting') is False,
                            'outer VM restoration is unverified')
            record['independent_stopped_verification'] = True
        except BaseException as error:record['cleanup_errors'].append(type(error).__name__+': '+str(error))
        for sig,handler in old.items():signal.signal(sig,handler)
        record.update(stdout=str(Path(stdout).relative_to(ROOT)),stderr=str(Path(stderr).relative_to(ROOT)))
    return record


