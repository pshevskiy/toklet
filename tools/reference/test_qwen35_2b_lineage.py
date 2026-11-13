"""Actual CLI cold failures retain a manifest and exit unsuccessfully."""
import hashlib
import io
import json
from pathlib import Path
import ssl
import struct
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import qwen35_2b_lineage as lineage
from test_qwen35_2b_layout import fixture


def context():
    return SimpleNamespace(verify_mode=ssl.CERT_REQUIRED,check_hostname=True)


def ca_arguments(root):
    # Positive unit tests mock SSL parsing: no CA certificate is vendored.
    path=root/'test-ca.pem';raw=b'synthetic test input, never a trusted certificate'
    path.write_bytes(raw)
    return ['--ca-file',str(path),'--ca-sha256',hashlib.sha256(raw).hexdigest()]


class Lineage(unittest.TestCase):
    def test_numeric_rejection_keeps_closed_authentication_evidence_and_returns_one(self):
        saved=lineage.layout.saved.SavedFile
        class FakeSource:
            def __new__(cls,path,size=None):
                if Path(path).name not in ('config.json','model.safetensors'):return saved(path,size)
                return super().__new__(cls)
            def __init__(self,path,size=None):
                self.path=Path(path);self.before=[1,2,size,3,4]
            def read(self,count,offset=0):
                if self.path.name=='config.json':return fixture('source-config.json')
                return struct.pack('<Q',lineage.layout.SOURCE_HEADER_BYTES) if offset==0 else fixture('source-header.bin')
            def digest(self):return lineage.layout.SOURCE_SHA256
            def check(self):return self.before
            def __enter__(self):return self
            def __exit__(self,*_):pass
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);raw=(json.dumps(lineage.parent.protocol())+'\n').encode()
            path=root/'protocol';path.write_bytes(raw)
            args=['--environment-root',str(root),'--runtime-manifest',str(root/'runtime'),'--runtime-sha256','1'*64,
                  '--source-dir',str(root/'source'),'--protocol',str(path),'--protocol-sha256',hashlib.sha256(raw).hexdigest(),
                  '--fresh-evidence',str(root/'evidence'),*ca_arguments(root)]
            with patch.object(lineage.layout.saved,'SavedFile',FakeSource), \
                    patch.object(lineage.runtime,'bind_environment'), \
                    patch.object(lineage.runtime,'admit',return_value=({},lambda:{})), \
                    patch.dict(sys.modules,{'numpy':object()}), \
                    patch.object(lineage.parent,'numerical_callbacks',return_value=(None,None)), \
                    patch.object(lineage.ssl,'create_default_context',return_value=context()), \
                    patch.object(lineage,'open_parent',return_value=io.BytesIO()), \
                    patch.object(lineage.parent,'authenticate_stream',return_value=([{'passed':False}],
                        {'whole_SHA256_authenticated':True,'EOF_checked':True})):
                self.assertEqual(lineage.main(args),1)
            report=json.loads((root/'evidence/manifest.json').read_bytes())
            self.assertFalse(report['passed'])
            self.assertTrue(report['parent']['whole_SHA256_authenticated'])
            self.assertTrue(report['selected_source']['whole_SHA256_authenticated_after'])
            self.assertTrue(report['transport']['TLS_verified_for_completed_stream'])
            self.assertTrue((root/'evidence/records.jsonl').exists())

    def test_invalid_protocol_publishes_failure_without_runtime_or_network(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); protocol = root/'protocol.json'; protocol.write_bytes(b'{}')
            output = root/'evidence'
            args = ['--environment-root', str(root), '--runtime-manifest', str(root/'runtime'),
                    '--runtime-sha256', '1'*64, '--source-dir', str(root/'model'),
                    '--protocol', str(protocol), '--protocol-sha256', hashlib.sha256(b'{}').hexdigest(),
                    '--fresh-evidence', str(output),*ca_arguments(root)]
            with patch.object(lineage.runtime, 'admit', side_effect=AssertionError('runtime must stay cold')), \
                    patch.object(lineage, 'open_parent', side_effect=AssertionError('network must stay cold')):
                self.assertEqual(lineage.main(args), 1)
            result = json.loads((output/'manifest.json').read_bytes())
            self.assertFalse(result['passed'])
            self.assertEqual(result['error_type'], 'ValueError')
            self.assertFalse(result['historical_publisher_lineage_proved'])
            self.assertFalse(result['model_numerics_quality_fit_performance_adoption_qualified'])

    def test_CA_wrong_hash_rejects_before_context_creation(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'ca.pem';path.write_bytes(b'untrusted input')
            with patch.object(lineage.ssl,'create_default_context') as create:
                with self.assertRaises(ValueError):lineage.load_ca(path,'1'*64)
                create.assert_not_called()

    def test_CA_invalid_PEM_rejects_with_real_stdlib_parser_without_network(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'ca.pem';raw=b'not a PEM certificate';path.write_bytes(raw)
            with patch.object(lineage.urllib.request,'build_opener',side_effect=AssertionError('No network')):
                with self.assertRaises((ssl.SSLError,ValueError)):
                    lineage.load_ca(path,hashlib.sha256(raw).hexdigest())

    def test_CA_bytes_hash_EOF_and_identity_are_reused_without_path_replacement(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);ca_arguments(root);path=root/'test-ca.pem';raw=path.read_bytes()
            with patch.object(lineage.ssl,'create_default_context',return_value=context()) as create:
                actual,record=lineage.load_ca(path,hashlib.sha256(raw).hexdigest())
                create.assert_called_once_with(cadata=raw.decode('ascii'))
            self.assertTrue(actual.check_hostname)
            self.assertEqual(lineage.verify_ca(record),record)
            self.assertEqual(record['before_identity'],record['after_identity'])
            path.rename(root/'retained-original.pem');path.write_bytes(raw)
            with self.assertRaises(ValueError):lineage.verify_ca(record)

    def test_CA_symlink_oversize_or_unverified_context_reject(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);path=root/'ca.pem';path.write_bytes(b'bounded')
            link=root/'link.pem';link.symlink_to(path)
            with self.assertRaises(OSError):lineage.load_ca(link,hashlib.sha256(b'bounded').hexdigest())
            path.write_bytes(bytes(lineage.parent.MAX_CA_BYTES+1))
            with self.assertRaises(ValueError):lineage.load_ca(path,hashlib.sha256(path.read_bytes()).hexdigest())
            path.write_bytes(b'bounded')
            for unsafe in (SimpleNamespace(verify_mode=ssl.CERT_NONE,check_hostname=True),
                           SimpleNamespace(verify_mode=ssl.CERT_REQUIRED,check_hostname=False)):
                with patch.object(lineage.ssl,'create_default_context',return_value=unsafe):
                    with self.assertRaises(ValueError):lineage.load_ca(path,hashlib.sha256(b'bounded').hexdigest())

    def test_CA_arguments_are_required_before_any_input_is_opened(self):
        args=['--environment-root','/synthetic','--runtime-manifest','/synthetic/runtime','--runtime-sha256','1'*64,
              '--source-dir','/synthetic/model','--protocol','/synthetic/protocol','--protocol-sha256','2'*64,
              '--fresh-evidence','/synthetic/output']
        with patch('sys.stderr',io.StringIO()),patch.object(lineage.files,'capture_named') as capture:
            with self.assertRaises(SystemExit) as failure:lineage.main(args)
            self.assertEqual(failure.exception.code,2)
            capture.assert_not_called()

    def test_HTTPS_handler_uses_pinned_context_and_rejects_changed_verification(self):
        ctx=context()
        response=SimpleNamespace(status=200,headers={'Content-Length':str(lineage.parent.PARENT['bytes'])},
            geturl=lambda:'https://example.invalid/verified',close=lambda:None)
        opener=SimpleNamespace(open=lambda *a,**k:response)
        with patch.object(lineage.urllib.request,'HTTPSHandler') as handler, \
                patch.object(lineage.urllib.request,'build_opener',return_value=opener):
            self.assertIs(lineage.open_parent(ctx),response)
            handler.assert_called_once_with(context=ctx)
        ctx.check_hostname=False
        with patch.object(lineage.urllib.request,'build_opener') as build:
            with self.assertRaises(ValueError):lineage.open_parent(ctx)
            build.assert_not_called()


if __name__ == '__main__':
    unittest.main()
