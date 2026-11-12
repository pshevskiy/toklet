"""Portable process-info bounds, failure and lifecycle controls; no model/GPU."""
import ctypes
import errno
import unittest
from unittest.mock import patch

import process_groups as groups


class FakeDarwin:
    def __init__(self, pids=(11,), states=None, errors=None):
        self.pids = pids
        self.states = states or {pid: (91, 2) for pid in pids if pid > 0}
        self.errors = errors or {}
        self.used = None
        self.info_bytes = 64
        self.wrong_pid = False

    def listing(self, kind, group, buffer, size):
        if kind != groups.PROC_PGRP_ONLY or group != 91:
            raise ValueError('Wrong listing request')
        target = ctypes.cast(buffer, ctypes.POINTER(ctypes.c_int32))
        for index, pid in enumerate(self.pids[:size//4]):
            target[index] = pid
        return len(self.pids)*4 if self.used is None else self.used

    def info(self, pid, flavor, arg, buffer, size):
        if flavor != 13 or arg != 0 or size != 64:
            raise ValueError('Wrong process-info request')
        if pid in self.errors:
            ctypes.set_errno(self.errors[pid])
            return 0
        target = ctypes.cast(buffer, ctypes.POINTER(groups._BsdShortInfo)).contents
        target.pid = pid + int(self.wrong_pid)
        target.pgid, target.status = self.states[pid]
        return self.info_bytes

    def query(self):
        return groups._darwin_live_members(91, self.listing, self.info)


class ProcessGroups(unittest.TestCase):
    def test_sdk_layout_and_named_fields(self):
        self.assertEqual(ctypes.sizeof(groups._BsdShortInfo), 64)
        self.assertEqual(groups._BsdShortInfo.pid.offset, 0)
        self.assertEqual(groups._BsdShortInfo.pgid.offset, 8)
        self.assertEqual(groups._BsdShortInfo.status.offset, 12)
        self.assertEqual(groups._BsdShortInfo.comm.offset, 16)
        self.assertEqual(groups._BsdShortInfo.reserved.offset, 60)

    def test_invalid_group_rejected_before_observation(self):
        with patch.object(groups, '_native_functions') as native, \
                patch.object(groups.subprocess, 'check_output') as ps:
            for value in [True, False, 0, -1, 2**31, '91', 1.5]:
                with self.assertRaises(ValueError):
                    groups.live_group_members(value)
            native.assert_not_called()
            ps.assert_not_called()

    def test_live_zombie_departed_and_zero_entries(self):
        fake = FakeDarwin((14, 0, 12, 11), {14: (91, 3), 12: (91, 5), 11: (92, 2)})
        self.assertEqual(fake.query(), [14])

    def test_reaped_pid_is_an_explicit_absence(self):
        fake = FakeDarwin((11, 12), errors={11: errno.ESRCH})
        self.assertEqual(fake.query(), [12])

    def test_live_permission_and_io_errors_fail_closed(self):
        for error in [errno.EPERM, errno.EACCES, errno.EIO]:
            with self.subTest(error=error), self.assertRaises(OSError) as raised:
                FakeDarwin(errors={11: error}).query()
            self.assertEqual(raised.exception.errno, error)

    def test_exactly_full_and_oversize_buffers_rejected(self):
        for size in [(groups.MAX_GROUP_MEMBERS+1)*4, (groups.MAX_GROUP_MEMBERS+2)*4]:
            fake = FakeDarwin()
            fake.used = size
            with self.assertRaises(ValueError):
                fake.query()

    def test_bad_byte_count_and_listing_error_rejected(self):
        for size in [1, 3, True, -1]:
            fake = FakeDarwin()
            fake.used = size
            with self.assertRaises((OSError, ValueError)):
                fake.query()

    def test_empty_group_and_listing_permission_failure(self):
        self.assertEqual(FakeDarwin(()).query(), [])
        fake = FakeDarwin(())
        def denied(*_):
            ctypes.set_errno(errno.EPERM)
            return 0
        with self.assertRaises(PermissionError):
            groups._darwin_live_members(91, denied, fake.info)

    def test_duplicate_and_negative_inventory_rejected(self):
        for pids in [(11, 11), (-1,)]:
            with self.assertRaises(ValueError):
                FakeDarwin(pids).query()

    def test_short_info_and_wrong_pid_rejected(self):
        for count in [0, 63, 65, True]:
            fake = FakeDarwin()
            fake.info_bytes = count
            with self.assertRaises(OSError):
                fake.query()
        fake = FakeDarwin()
        fake.wrong_pid = True
        with self.assertRaises(ValueError):
            fake.query()

    def test_darwin_route_never_spawns_ps(self):
        fake = FakeDarwin()
        with patch.object(groups.sys, 'platform', 'darwin'), \
                patch.object(groups, '_native_functions', return_value=(object(), fake.listing, fake.info)), \
                patch.object(groups.subprocess, 'check_output') as ps:
            self.assertEqual(groups.live_group_members(91), [11])
            ps.assert_not_called()

    def test_other_platform_retains_zombie_filtered_fallback(self):
        with patch.object(groups.sys, 'platform', 'linux'), \
                patch.object(groups.subprocess, 'check_output',
                             return_value='11 91 S\n12 91 Z\n13 92 R\n') as ps:
            self.assertEqual(groups.live_group_members(91), [11])
            ps.assert_called_once_with(['ps', '-axo', 'pid=,pgid=,stat='], text=True)


if __name__ == '__main__':
    unittest.main()
