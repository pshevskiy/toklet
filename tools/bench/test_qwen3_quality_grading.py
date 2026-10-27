"""Model-idle and evidence admission controls; no VM or generated code runs."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import qwen3_quality_grading as subject


def complete():
    return dict(passed=True,primary_error=None,endpoint_errors=[],
        observations=dict(scope=dict(full_model_collections_complete=True)),
        commands=[dict(pid=100001+i,exit_code=0,primary_error=None,cleanup_errors=[])for i in range(7)])


class GradingAdmission(unittest.TestCase):
    def test_outer_timeout_or_interrupt_retains_owner_without_forced_kill(self):
        from unittest.mock import Mock
        import subprocess
        import signal
        stopped=[dict(Name=subject.lifecycle.MACHINE,Running=False,Starting=False)]
        for error in (subprocess.TimeoutExpired('window',1),KeyboardInterrupt(),-15):
            process=Mock(pid=12345);process.wait.side_effect=[error,-15];process.poll.return_value=-15
            alive=Mock(side_effect=[True,True,False])
            with tempfile.TemporaryDirectory(dir=subject.ROOT/'artifacts') as temporary:
                root=Path(temporary)
                with patch.object(subject.subprocess,'Popen',return_value=process),patch.object(subject.os,'killpg') as send,patch.object(subject.time,'sleep'):
                    record=subject.supervise_window(['window'],root/'out',root/'err',{},lambda:stopped,alive,1)
                if type(error) is int:send.assert_not_called()
                else:send.assert_called_once_with(12345,signal.SIGTERM)
                self.assertEqual(record['signals'],[]if type(error)is int else['SIGTERM']);self.assertEqual(alive.call_count,3)
                self.assertTrue(record['independent_stopped_verification']);self.assertEqual(record['exit_code'],-15)
                self.assertIsNotNone(record['primary_error'])

    def test_outer_window_rejects_running_vm_before_or_after_worker(self):
        from unittest.mock import Mock
        stopped=[dict(Name=subject.lifecycle.MACHINE,Running=False,Starting=False)]
        running=[dict(Name=subject.lifecycle.MACHINE,Running=True,Starting=False)]
        with tempfile.TemporaryDirectory(dir=subject.ROOT/'artifacts') as temporary:
            root=Path(temporary);process=Mock(pid=12345);process.wait.return_value=0;process.poll.return_value=0
            with patch.object(subject.subprocess,'Popen',return_value=process) as spawn:
                record=subject.supervise_window(['window'],root/'out',root/'err',{},lambda:running,lambda _:False)
                spawn.assert_not_called();self.assertTrue(record['cleanup_errors'])
            states=iter([stopped,running])
            with patch.object(subject.subprocess,'Popen',return_value=process):
                record=subject.supervise_window(['window'],root/'out',root/'err',{},lambda:next(states),lambda _:False)
                self.assertEqual(record['exit_code'],0);self.assertTrue(record['cleanup_errors'])

    def test_every_model_process_and_group_must_be_absent(self):
        with patch.object(subject.os,'kill',side_effect=ProcessLookupError) as query:
            self.assertEqual(subject.require_idle(complete()),list(range(100001,100008)))
            self.assertEqual(query.call_count,14)
        with patch.object(subject.os,'kill',return_value=None):
            with self.assertRaisesRegex(ValueError,'remains live'):subject.require_idle(complete())

    def test_partial_failed_or_unowned_collections_cannot_start_grading(self):
        for kind in ('partial','phase','exit','cleanup','duplicate'):
            record=complete()
            if kind=='partial':record['passed']=False
            elif kind=='phase':record['commands'].pop()
            elif kind=='exit':record['commands'][0]['exit_code']=1
            elif kind=='cleanup':record['commands'][0]['cleanup_errors']=['unverified']
            else:record['commands'][1]['pid']=record['commands'][0]['pid']
            with patch.object(subject.os,'kill',side_effect=ProcessLookupError):
                with self.assertRaises(ValueError):subject.require_idle(record)

    def test_streamed_evidence_checks_payload_extent_and_symlinks(self):
        import hashlib
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);path=root/'evidence';path.write_bytes(b'original')
            pin=dict(bytes=8,sha256=hashlib.sha256(b'original').hexdigest())
            subject.verify_file(path,pin)
            path.write_bytes(b'changed!')
            with self.assertRaises(ValueError):subject.verify_file(path,pin)
            alias=root/'alias';alias.symlink_to(path)
            with self.assertRaises(OSError):subject.verify_file(alias,pin)

    def test_growing_file_reads_only_admitted_bytes_plus_one(self):
        import hashlib
        real_fdopen=subject.os.fdopen;requests=[]
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'growing';path.write_bytes(b'original')
            class Growing:
                def __init__(self,file):self.file=file
                def __enter__(self):return self
                def __exit__(self,*_):self.file.close()
                def fileno(self):return self.file.fileno()
                def read(self,count):
                    requests.append(count)
                    if len(requests)==1:
                        with path.open('ab')as file:file.write(b'grow'*10000)
                    return self.file.read(count)
            with patch.object(subject.os,'fdopen',side_effect=lambda fd,mode:Growing(real_fdopen(fd,mode))):
                with self.assertRaises(ValueError):subject.verify_file(path,dict(bytes=8,sha256=hashlib.sha256(b'original').hexdigest()))
            self.assertEqual(requests,[8,1])

    def test_post_decision_failure_cannot_publish_completion(self):
        from unittest.mock import Mock
        pending=dict(decision='retain_experimental_candidate')
        record=dict(completed=False,restored_stopped=True,pipe_lifetime_finished=True,
            failure=dict(type='ValueError',message='post-grading image changed'))
        guard=Mock();subject.finish_decision(record,pending,guard)
        guard.assert_not_called();self.assertFalse(record['completed']);self.assertNotIn('decision',record)
        record.pop('failure');guard=Mock(side_effect=ValueError('post-grading source changed'))
        subject.finish_decision(record,pending,guard)
        self.assertFalse(record['completed']);self.assertNotIn('decision',record);self.assertIn('failure',record)


if __name__=='__main__':unittest.main()
