import ctypes
import unittest
from unittest.mock import patch
import native_usage


class Call:
    def __init__(self, result): self.result = result
    def __call__(self, pid, flavor, output):
        assert flavor == 4
        value = ctypes.cast(output,ctypes.POINTER(native_usage.Usage)).contents
        value.resident_size = 100
        value.phys_footprint = 200
        value.lifetime_max_phys_footprint = 300
        return self.result


class NativeUsage(unittest.TestCase):
    def test_sdk_v4_layout(self):
        self.assertEqual(ctypes.sizeof(native_usage.Usage),296)
        self.assertEqual(native_usage.Usage.phys_footprint.offset,72)
        self.assertEqual(native_usage.Usage.lifetime_max_phys_footprint.offset,240)
    def test_checked_public_call(self):
        library = type("Library",(),{"proc_pid_rusage":Call(0)})()
        with patch.object(native_usage.ctypes,"CDLL",return_value=library), patch.object(native_usage.sys,"platform","darwin"):
            self.assertEqual(native_usage.usage(),{"resident_size":100,"phys_footprint":200,"lifetime_max_phys_footprint":300})
    def test_failed_call_has_no_memory_claim(self):
        library = type("Library",(),{"proc_pid_rusage":Call(-1)})()
        with patch.object(native_usage.ctypes,"CDLL",return_value=library), patch.object(native_usage.sys,"platform","darwin"), patch.object(native_usage.ctypes,"get_errno",return_value=1):
            with self.assertRaises(OSError): native_usage.usage()


if __name__ == "__main__": unittest.main()
