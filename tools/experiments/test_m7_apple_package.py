import importlib.util
import os
import plistlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

spec=importlib.util.spec_from_file_location('package',Path(__file__).with_name('m7_apple_package.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class PackageTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        p=patch.object(m,'ROOT',self.root);p.start();self.addCleanup(p.stop)

    def run_command(self,command,timeout=1):
        return m.owned(command,self.root/'out',self.root/'err',dict(os.environ),timeout)

    def test_real_timeout_reaps_owned_child(self):
        row=self.run_command([sys.executable,'-c','import time;time.sleep(30)'],.05)
        self.assertIn('TimeoutExpired',row['primary_error'])
        self.assertIsNotNone(row['exit_code']);self.assertEqual(row['cleanup_errors'],[])

    def test_descendant_ignoring_term_is_drained_after_parent_exits(self):
        marker=self.root/'child-ready'
        child="import signal,time,pathlib;signal.signal(signal.SIGTERM,signal.SIG_IGN);pathlib.Path("+repr(str(marker))+ ").write_text('ready');time.sleep(30)"
        parent="import subprocess,sys,time,pathlib;p=subprocess.Popen([sys.executable,'-c',"+repr(child)+"]);\nwhile not pathlib.Path("+repr(str(marker))+").exists():time.sleep(.01)\ntime.sleep(30)"
        row=m.owned([sys.executable,'-c',parent],self.root/'out',self.root/'err',dict(os.environ),.5,.5)
        self.assertTrue(marker.exists())
        self.assertIn('TimeoutExpired',row['primary_error']);self.assertEqual(row['cleanup_errors'],[])
        self.assertFalse(m.group_exists(row['pid']))

    def test_exit_race_is_reaped_and_primary_preserved(self):
        p=Mock(pid=12345);p.wait.side_effect=[subprocess.TimeoutExpired('primary',1),0]
        with patch.object(m.subprocess,'Popen',return_value=p),patch.object(m,'group_exists',return_value=False):
            row=self.run_command(['fixture'])
        self.assertIn('primary',row['primary_error']);self.assertEqual(row['exit_code'],0)
        self.assertEqual(p.wait.call_args_list[-1].kwargs,{'timeout':15})

    def test_final_group_and_reap_failure_are_preserved(self):
        p=Mock(pid=12345)
        p.wait.side_effect=[subprocess.TimeoutExpired('primary',1),subprocess.TimeoutExpired('kill',.01)]
        with patch.object(m.subprocess,'Popen',return_value=p),patch.object(m,'group_exists',return_value=True),patch.object(m.os,'killpg') as kill:
            row=m.owned(['fixture'],self.root/'out',self.root/'err',dict(os.environ),1,.01)
        self.assertIn('primary',row['primary_error']);self.assertEqual(len(row['cleanup_errors']),2)
        self.assertEqual([x.args[1] for x in kill.call_args_list],[signal.SIGTERM,signal.SIGKILL])

    def test_environment_is_allowlisted_and_configuration_rejected(self):
        with patch.object(Path,'home',return_value=self.root),patch.dict(os.environ,{'RUSTFLAGS':'unadmitted','CARGO_PROFILE_RELEASE_LTO':'fat'}):
            env,absent=m.build_environment()
            self.assertNotIn('RUSTFLAGS',env);self.assertNotIn('CARGO_PROFILE_RELEASE_LTO',env)
            self.assertIn(str(self.root/'.cargo/config.toml'),absent)
            (self.root/'.cargo').mkdir();(self.root/'.cargo/config.toml').write_text('[build]')
            with self.assertRaises(ValueError):m.build_environment()

    def test_bundle_validates_platform_copies_and_paths(self):
        bundle=self.root/'Toklet.xcframework';bundle.mkdir()
        header=self.root/'toklet.h';header.write_bytes(b'header')
        rows=[];slices=[]
        for index,specification in enumerate(m.SPECS):
            folder=bundle/str(index);folder.mkdir();(folder/'Headers').mkdir()
            (folder/'libtoklet.a').write_bytes(b'archive');(folder/'Headers/toklet.h').write_bytes(b'header')
            rows.append(dict(LibraryIdentifier=str(index),LibraryPath='libtoklet.a',HeadersPath='Headers',SupportedPlatform=specification['name'],SupportedPlatformVariant=specification['variant'],SupportedArchitectures=['arm64']))
            slices.append(dict(specification,library_sha256=m.sha(folder/'libtoklet.a')))
        def write(): (bundle/'Info.plist').write_bytes(plistlib.dumps(dict(AvailableLibraries=rows)))
        write();m.check_bundle(bundle,slices,{'toklet.h':m.sha(header)})
        for field,value in [('SupportedPlatform','bad'),('LibraryPath','../../toklet.h')]:
            original=rows[0][field];rows[0][field]=value;write()
            with self.assertRaises(ValueError):m.check_bundle(bundle,slices,{'toklet.h':m.sha(header)})
            rows[0][field]=original
        write()
        for path in (bundle/'0/libtoklet.a',bundle/'0/Headers/toklet.h'):
            original=path.read_bytes();path.write_bytes(b'changed')
            with self.assertRaises(ValueError):m.check_bundle(bundle,slices,{'toklet.h':m.sha(header)})
            path.write_bytes(original)

    def test_platform_and_minimum_are_exact(self):
        text='cmd LC_BUILD_VERSION\n cmdsize 32\n platform 2\n minos 18.0\n sdk 27.1'
        m.check_load_commands(text,m.SPECS[1])
        for other in (m.SPECS[0],m.SPECS[2],dict(m.SPECS[1],minos='17.0')):
            with self.assertRaises(ValueError):m.check_load_commands(text,other)

    def test_symbol_lines_are_bounded(self):
        p=self.root/'symbols';p.write_text('00000 T _toklet_abi_version\n')
        self.assertEqual(m.native_symbols(p),{'toklet_abi_version'})
        p.write_text('x'*65537)
        with self.assertRaises(ValueError):m.native_symbols(p)

    def test_sigterm_unwinds_for_cleanup(self):
        with self.assertRaises(InterruptedError):m.interrupted(signal.SIGTERM,None)

    def test_metadata_is_one_bounded_read(self):
        stream=Mock();stream.read.return_value=b'x'*(16*1024**2+1)
        stream.__enter__=Mock(return_value=stream);stream.__exit__=Mock(return_value=False)
        with patch.object(Path,'open',return_value=stream),self.assertRaises(ValueError):m.small(self.root/'metadata')
        stream.read.assert_called_once_with(16*1024**2+1)


if __name__=='__main__':unittest.main()
