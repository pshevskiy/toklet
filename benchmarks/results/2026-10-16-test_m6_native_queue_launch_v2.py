"""Ownership tests with mocked launchers and one bounded CPU-only signal fixture."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import runpy
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock,patch
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'artifacts/m6-native-queue-launch-v2.py'
QUEUE=ROOT/'artifacts/m6-remaining-native-queue-v2.py'
spec=importlib.util.spec_from_file_location('resume_queue',QUEUE)
queue=importlib.util.module_from_spec(spec);spec.loader.exec_module(queue)

class Launch(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
  self.root=Path(self.temp.name).resolve();self.artifacts=self.root/'artifacts';self.artifacts.mkdir()
  self.launcher=self.artifacts/SOURCE.name;self.launcher.write_bytes(SOURCE.read_bytes())
  (self.artifacts/'m6-remaining-native-queue-v2.py').write_text('mock queue source\n')
  self.child=Mock(pid=33001)
 def run_launcher(self):
  with contextlib.redirect_stdout(io.StringIO()):runpy.run_path(str(self.launcher),run_name='__main__')
 def test_detached_owned_supervision(self):
  with patch('subprocess.Popen',return_value=self.child) as popen:self.run_launcher()
  args,kwargs=popen.call_args
  self.assertEqual(args[0],[str(self.root/'.venv/bin/python'),str(self.artifacts/'m6-remaining-native-queue-v2.py')])
  self.assertIs(kwargs['start_new_session'],True);self.assertEqual(kwargs['stdin'],subprocess.DEVNULL);self.assertEqual(kwargs['stderr'],subprocess.STDOUT);self.assertEqual(kwargs['cwd'],self.root)
  record=json.loads((self.artifacts/'m6-remaining-native-queue-v2.owner.json').read_text());self.assertEqual(record['pid'],33001);self.assertEqual(record['pgid'],33001);self.assertIs(record['qualification'],False)
  self.child.wait.assert_not_called()
 def test_existing_owner_blocks_duplicate_launch(self):
  (self.artifacts/'m6-remaining-native-queue-v2.owner.json').write_text('existing evidence')
  with patch('subprocess.Popen') as popen:
   with self.assertRaises(ValueError):self.run_launcher()
   popen.assert_not_called()
 def test_metadata_failure_gracefully_cleans_created_group(self):
  original=OSError('metadata failure')
  with patch('subprocess.Popen',return_value=self.child),patch('json.dump',side_effect=original),patch('os.killpg') as killpg:
   with self.assertRaises(OSError) as raised:self.run_launcher()
   self.assertIs(raised.exception,original);killpg.assert_called_once_with(33001,signal.SIGINT);self.child.wait.assert_called_once_with(timeout=180)
 def test_cleanup_failures_preserve_primary_error(self):
  original=OSError('publication failure');self.child.wait.side_effect=subprocess.TimeoutExpired('mock',180)
  with patch('subprocess.Popen',return_value=self.child),patch('json.dump',side_effect=original),patch('os.killpg',side_effect=PermissionError('signal failure')):
   with self.assertRaises(OSError) as raised:self.run_launcher()
   self.assertIs(raised.exception,original);notes=' '.join(original.__notes__);self.assertIn('interrupt failed',notes);self.assertIn('wait failed',notes)
 def test_file_close_failures_use_cleanup_boundary(self):
  original_open=Path.open
  for ending in ('.owner.json','.supervisor.log'):
   with self.subTest(ending=ending):
    for path in self.artifacts.glob('m6-remaining-native-queue-v2.*'):
     if path.suffix!='.py':path.unlink()
    original=OSError('close failure');self.child.reset_mock()
    class FailClose:
     def __init__(self,stream):self.stream=stream
     def __enter__(self):return self.stream.__enter__()
     def __exit__(self,*args):self.stream.__exit__(*args);raise original
    def opened(path,*args,**kwargs):
     stream=original_open(path,*args,**kwargs)
     return FailClose(stream) if str(path).endswith(ending) else stream
    with patch.object(Path,'open',opened),patch('subprocess.Popen',return_value=self.child),patch('os.killpg') as killpg:
     with self.assertRaises(OSError) as raised:self.run_launcher()
     self.assertIs(raised.exception,original);killpg.assert_called_once_with(33001,signal.SIGINT);self.child.wait.assert_called_once_with(timeout=180)
 def test_supervisor_waits_for_collector_cleanup_on_interrupt(self):
  child=Mock(pid=33002);interruption=KeyboardInterrupt();child.wait.side_effect=[interruption,-2]
  with patch.object(queue.subprocess,'Popen',return_value=child):
   with self.assertRaises(KeyboardInterrupt) as raised:queue.run_collector(['mock'],io.BytesIO())
   self.assertIs(raised.exception,interruption);self.assertEqual(child.wait.call_count,2);self.assertEqual(child.wait.call_args.kwargs,dict(timeout=120))
 def test_supervisor_cleanup_wait_failure_preserves_interrupt(self):
  child=Mock(pid=33002);interruption=KeyboardInterrupt();child.wait.side_effect=[interruption,subprocess.TimeoutExpired('mock',120)]
  with patch.object(queue.subprocess,'Popen',return_value=child):
   with self.assertRaises(KeyboardInterrupt) as raised:queue.run_collector(['mock'],io.BytesIO())
   self.assertIs(raised.exception,interruption);self.assertIn('verify owned worker absence',' '.join(interruption.__notes__))
 def test_real_group_interrupt_reaps_separate_session_child(self):
  # Three CPU-only processes model queue -> collector -> separately owned worker.
  # The signal targets queue+collector; only collector finally reaches the worker.
  pidfile=self.root/'worker.pid';marker=self.root/'collector.cleaned';collector=self.root/'collector.py';wrapper=self.root/'wrapper.py';log=self.root/'signal.log'
  collector.write_text("import subprocess,sys,time\nfrom pathlib import Path\nw=subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)'],start_new_session=True)\ntry:\n Path(sys.argv[1]).write_text(str(w.pid))\n while True:time.sleep(1)\nfinally:\n w.terminate();w.wait(timeout=5)\n Path(sys.argv[2]).write_text('cleaned')\n")
  wrapper.write_text("import importlib.util,sys\ns=importlib.util.spec_from_file_location('q',sys.argv[1]);q=importlib.util.module_from_spec(s);s.loader.exec_module(q)\nwith open(sys.argv[5],'wb') as log:\n try:q.run_collector([sys.executable,sys.argv[2],sys.argv[3],sys.argv[4]],log)\n except KeyboardInterrupt:raise SystemExit(130)\n")
  process=subprocess.Popen([sys.executable,str(wrapper),str(QUEUE),str(collector),str(pidfile),str(marker),str(log)],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
  try:
   deadline=time.monotonic()+5
   while not pidfile.exists() and time.monotonic()<deadline:time.sleep(.01)
   self.assertTrue(pidfile.exists(),'CPU collector did not become ready')
   worker=int(pidfile.read_text());os.killpg(process.pid,signal.SIGINT)
   self.assertEqual(process.wait(timeout=10),130);self.assertEqual(marker.read_text(),'cleaned')
   with self.assertRaises(ProcessLookupError):os.kill(worker,0)
  finally:
   if process.poll() is None:
    os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
   if pidfile.exists():
    try:os.killpg(int(pidfile.read_text()),signal.SIGKILL)
    except ProcessLookupError:pass

if __name__=='__main__':unittest.main()
