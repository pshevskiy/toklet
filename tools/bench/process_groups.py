"""Bounded live process-group snapshots without spawning setuid ps on macOS.

This module only observes groups. Callers retain ownership of their Popen
session leader and decide whether signaling is authorized. Snapshots are not
a continuous guarantee about processes created after the query.
"""
import ctypes
import errno
from functools import lru_cache
import subprocess
import sys

MAX_GROUP_MEMBERS = 8192
PROC_PGRP_ONLY = 2
PROC_PIDT_SHORTBSDINFO = 13
SZOMB = 5


class _BsdShortInfo(ctypes.Structure):
    # macOS SDK sys/proc_info.h: proc_bsdshortinfo, 64 bytes.
    _fields_ = [(name, ctypes.c_uint32) for name in ('pid', 'ppid', 'pgid', 'status')] + [
        ('comm', ctypes.c_char * 16),
    ] + [(name, ctypes.c_uint32) for name in
         ('flags', 'uid', 'gid', 'ruid', 'rgid', 'svuid', 'svgid', 'reserved')]


def _group_id(value):
    if type(value) is not int or not 0 < value <= 0x7fffffff:
        raise ValueError('Positive signed-32-bit process group required')
    return value


@lru_cache(maxsize=1)
def _native_functions():
    if ctypes.sizeof(_BsdShortInfo) != 64 or ctypes.sizeof(ctypes.c_int32) != 4:
        raise ValueError('Unsupported macOS process-info ABI')
    library = ctypes.CDLL('/usr/lib/libproc.dylib', use_errno=True)
    listing = library.proc_listpids
    listing.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_int]
    listing.restype = ctypes.c_int
    info = library.proc_pidinfo
    info.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
                     ctypes.c_void_p, ctypes.c_int]
    info.restype = ctypes.c_int
    return library, listing, info


def _darwin_live_members(group, listing, info):
    # One extra slot makes an exactly full result fail closed instead of
    # silently accepting a possibly truncated group. Zero entries can result
    # from processes exiting during the kernel snapshot.
    buffer = (ctypes.c_int32 * (MAX_GROUP_MEMBERS + 1))()
    ctypes.set_errno(0)
    used = listing(PROC_PGRP_ONLY, group, ctypes.byref(buffer), ctypes.sizeof(buffer))
    error = ctypes.get_errno()
    if type(used) is not int or used < 0:
        raise OSError(error or errno.EIO, 'Process-group listing failed')
    if used == 0 and error not in (0, errno.ESRCH):
        raise OSError(error, 'Process-group listing failed')
    if used % 4 or used >= ctypes.sizeof(buffer):
        raise ValueError('Truncated or invalid process-group listing')
    pids = [buffer[index] for index in range(used // 4) if buffer[index] != 0]
    if any(pid < 0 for pid in pids) or len(pids) != len(set(pids)):
        raise ValueError('Invalid process-group PID inventory')
    live = []
    for pid in pids:
        value = _BsdShortInfo()
        ctypes.set_errno(0)
        count = info(pid, PROC_PIDT_SHORTBSDINFO, 0,
                     ctypes.byref(value), ctypes.sizeof(value))
        error = ctypes.get_errno()
        if count == 0 and error == errno.ESRCH:
            continue
        if type(count) is not int or count != ctypes.sizeof(value):
            raise OSError(error or errno.EIO, 'Process-info read failed for PID '+str(pid))
        if value.pid != pid:
            raise ValueError('Process-info PID differs')
        # A process may leave the queried group between the two observations.
        if value.pgid == group and value.status != SZOMB:
            live.append(pid)
    return sorted(live)


def live_group_members(group):
    """Return observed non-zombie members; reject incomplete/error snapshots."""
    group = _group_id(group)
    if sys.platform == 'darwin':
        _, listing, info = _native_functions()  # Cached owner keeps libproc loaded.
        return _darwin_live_members(group, listing, info)
    snapshot = subprocess.check_output(['ps', '-axo', 'pid=,pgid=,stat='], text=True)
    rows = snapshot.splitlines()
    if len(rows) > MAX_GROUP_MEMBERS:
        raise ValueError('Process table exceeds discovery bound')
    return [int(row[0]) for line in rows if len(row := line.split()) == 3
            and int(row[1]) == group and not row[2].startswith('Z')]
