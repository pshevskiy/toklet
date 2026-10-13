"""Public macOS libproc v4 ABI from the SDK's sys/resource.h/libproc.h.

Read a still-owned idle child, never an arbitrary PID. Native charge counters
may omit shared/clean pages; these do not replace numerical admission accounting.
"""
import ctypes
import os
import sys

FIELDS = "user_time system_time pkg_idle_wkups interrupt_wkups pageins wired_size resident_size phys_footprint proc_start_abstime proc_exit_abstime child_user_time child_system_time child_pkg_idle_wkups child_interrupt_wkups child_pageins child_elapsed_abstime diskio_bytesread diskio_byteswritten cpu_time_qos_default cpu_time_qos_maintenance cpu_time_qos_background cpu_time_qos_utility cpu_time_qos_legacy cpu_time_qos_user_initiated cpu_time_qos_user_interactive billed_system_time serviced_system_time logical_writes lifetime_max_phys_footprint instructions cycles billed_energy serviced_energy interval_max_phys_footprint runnable_time".split()


class Usage(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_ubyte * 16)] + [(name, ctypes.c_uint64) for name in FIELDS]


def usage(pid=None):
    if sys.platform != "darwin":
        raise RuntimeError("native usage requires macOS")
    assert ctypes.sizeof(Usage) == 296
    lib = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    lib.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    lib.proc_pid_rusage.restype = ctypes.c_int
    value = Usage()
    if lib.proc_pid_rusage(os.getpid() if pid is None else pid, 4, ctypes.byref(value)) != 0:
        raise OSError(ctypes.get_errno(), "proc_pid_rusage v4 failed")
    return {name: getattr(value, name) for name in ("resident_size", "phys_footprint", "lifetime_max_phys_footprint")}
