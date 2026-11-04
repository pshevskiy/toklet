import copy
import unittest
import linux_process_cpu_guest as subject


class IdentityAndClockControls(unittest.TestCase):
    def test_stat_name_parentheses_do_not_shift_start_identity(self):
        fields = ['S']+[str(i) for i in range(4, 53)]
        value = subject.stat_identity('99 (name with ) spaces) '+' '.join(fields))
        self.assertEqual(value, dict(start_ticks=22, ppid=4))
        with self.assertRaises(ValueError):
            subject.stat_identity('99 missing comm')

    def test_integer_clock_boundary_and_one_nanosecond_short(self):
        self.assertEqual(subject.cpu_nanoseconds(4, 0), 4_000_000_000)
        self.assertEqual(subject.cpu_nanoseconds(3, 999_999_999), 3_999_999_999)
        for pair in [(-1, 0), (0, -1), (0, 10**9), (True, 0), (61, 0)]:
            with self.subTest(pair=pair), self.assertRaises(ValueError):
                subject.cpu_nanoseconds(*pair)

    def test_signed_linux_process_clock_identity_excludes_per_thread_bit(self):
        self.assertEqual(subject.process_clock_id(42, 0), -344)
        self.assertEqual(subject.process_clock_id(42, 2), -342)
        self.assertEqual(subject.process_clock_id(42, 0) & 7, 0)
        for pair in [(0, 0), (True, 0), (42, 4), (42, True), (2**28, 0)]:
            with self.subTest(pair=pair), self.assertRaises(ValueError):
                subject.process_clock_id(*pair)

    def test_changed_process_or_namespace_identity_rejects(self):
        original = dict(pid=42, tgid=42, threads=1, start_ticks=100, namespace='pid:[2]',
                        cgroup='0::/libpod-lease.scope', nspid=[42, 1], tracer_pid=0)
        attached = copy.deepcopy(original)
        attached['tracer_pid'] = 55
        subject.stable_identity(original, attached)
        for key, value in [('pid', 43), ('tgid', 43), ('threads', 2), ('start_ticks', 101),
                           ('namespace', 'pid:[3]'), ('cgroup', 'other'), ('nspid', [43, 1])]:
            wrong = copy.deepcopy(attached)
            wrong[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                subject.stable_identity(original, wrong)
