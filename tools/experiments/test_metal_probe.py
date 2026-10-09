"""Offline admission and owned-process timeout checks, without Apple APIs."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import metal_probe as probe


class MetalChecks(unittest.TestCase):
    def test_existing_evidence_and_build_are_rejected_before_toolchain_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, build = root / "probe.json", root / "build"
            with patch.object(probe, "command") as command:
                for existing in (output, output.with_suffix(".json.stderr.txt"), build):
                    existing.touch()
                    with self.assertRaisesRegex(ValueError, "overwrite"):
                        probe.run(output, build)
                    existing.unlink()
                command.assert_not_called()

    def test_timeout_kills_only_the_new_process_group_and_reaps_its_child(self):
        calls = []
        real_killpg = os.killpg

        def killpg(pid, sig):
            calls.append((pid, sig))
            return real_killpg(pid, sig)

        with tempfile.TemporaryFile(mode="w+") as stderr, patch.object(probe.os, "killpg", side_effect=killpg):
            with self.assertRaises(subprocess.TimeoutExpired):
                probe.bounded([sys.executable, "-c", "import time; time.sleep(10)"], stderr, 0.05)
        self.assertEqual(len(calls), 1)
        pid, sig = calls[0]
        self.assertNotEqual(pid, os.getpid())
        self.assertEqual(sig, signal.SIGKILL)
        with self.assertRaises(ChildProcessError):
            os.waitpid(pid, os.WNOHANG)


if __name__ == "__main__":
    unittest.main()
