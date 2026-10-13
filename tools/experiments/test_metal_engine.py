"""Tiny owned processes exercise collector failures; no GPU/model payload work."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import metal_engine as engine


class Collector(unittest.TestCase):
    def exercise(self, script, error, timeout=False):
        launch_real = subprocess.Popen
        children = []
        def launch(*args, **kwargs):
            self.assertTrue(kwargs["start_new_session"])
            child = launch_real([sys.executable,"-c",script],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
            children.append(child)
            return child
        def digest(path):
            return "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe" if path.name == "model.safetensors" else "bound-test-identity"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"result.json"
            started = time.monotonic()
            with patch.object(engine,"digest",side_effect=digest), patch.object(engine.subprocess,"Popen",side_effect=launch), patch.object(engine.subprocess,"check_output",return_value="test"):
                if timeout:
                    with patch.object(engine.Transport,"read",side_effect=TimeoutError("test timed out")):
                        with self.assertRaisesRegex(TimeoutError,error): engine.run(Path("/tmp/fake-binary"),output)
                else:
                    with self.assertRaisesRegex(ValueError,error): engine.run(Path("/tmp/fake-binary"),output)
            self.assertLess(time.monotonic()-started,5)
            self.assertIsNotNone(children[0].poll())
            result = json.loads(output.read_text())
            self.assertIn("error",result)
            self.assertTrue(result["identities_unchanged"])
            self.assertTrue(output.with_suffix(".raw.jsonl").exists())
            self.assertLessEqual(output.with_suffix(".stderr").stat().st_size,1024*1024)
    def test_unexpected_event_reaps_child_and_retains_failure(self):
        self.exercise('print(\'{"event":"unexpected"}\',flush=True); import time; time.sleep(60)',"unexpected")
    def test_eof_reaps_child(self): self.exercise("pass","ended unexpectedly")
    def test_oversized_line_reaps_child(self): self.exercise('print("x"*65537,flush=True)',"exceeds bounded")
    def test_timeout_reaps_child(self): self.exercise("import time; time.sleep(60)","timed out",True)
    def test_stderr_flood_is_capped_and_reaped(self):
        self.exercise('import os,time; os.write(2,b"x"*(2*1024*1024)); time.sleep(60)',"stderr exceeds")
    def test_exited_leader_descendant_holding_pipes_is_cleaned(self):
        self.exercise('import os,time; p=os.fork();\nif p==0: time.sleep(60)\nelse: print(\'{"event":"unexpected"}\',flush=True); os._exit(0)',"unexpected")
    def test_existing_evidence_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"result.json"; output.write_text("retained")
            with patch.object(engine,"digest",return_value="fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"), patch.object(engine.subprocess,"Popen") as launch:
                with self.assertRaisesRegex(ValueError,"already exists"): engine.run(Path("/tmp/fake-binary"),output)
                launch.assert_not_called()
            self.assertEqual(output.read_text(),"retained")


if __name__ == "__main__": unittest.main()
