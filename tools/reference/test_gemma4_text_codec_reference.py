"""Offline request, capture, cold-admission and driver-failure regressions."""
import hashlib
import io
from importlib.machinery import ModuleSpec, ExtensionFileLoader, SourceFileLoader
import json
from pathlib import Path
import tempfile
import signal
import unittest
from unittest.mock import patch

import gemma4_text_codec_reference as d


class CodecReference(unittest.TestCase):
    def test_strict_subset_options_roles_literals_and_control_bounds(self):
        options=dict(enable_thinking=False,add_generation_prompt=True)
        good=[dict(role='user',content=' \tПривет 世界🙂\n')]
        d.validate(good,options,['<bos>'])
        for messages,settings in [([],options),(good*64,options),
                ([dict(role='system',content='Hi')],options),
                ([dict(role='user',content=[])],options),
                ([dict(role='user',content=' \n')],options),
                ([dict(role='user',content='<bos>')],options),
                ([dict(role='user',content='\u001cHi')],options),
                ([dict(role='user',content='x'*d.MAX_TEXT)],options),
                ([dict(good[0],reasoning='secret')],options),
                (good,dict(options,enable_thinking=0)),
                (good,dict(options,tools=[])),(good,dict(options,add_generation_prompt=False))]:
            with self.subTest(messages=messages[:1],settings=settings):
                with self.assertRaises(ValueError):d.validate(messages,settings,['<bos>'])

    def test_exact_jsonl_bytes_descriptors_and_prewrite_bounds(self):
        stream=io.BytesIO();capture=d.Capture(stream)
        values=[dict(kind='test',text='é\n🙂'),dict(kind='test',ids=[1,2])]
        for value in values:capture.add(value)
        raw=stream.getvalue();cursor=0
        for value,item in zip(values,capture.records):
            body=raw[cursor:cursor+item['bytes']];self.assertEqual(json.loads(body),value)
            self.assertEqual(item,dict(offset=cursor,bytes=len(body),sha256=hashlib.sha256(body).hexdigest(),completed=True))
            cursor+=len(body)
        self.assertEqual(cursor,capture.bytes)
        with self.assertRaises(ValueError):capture.add({'oversize':'x'*d.MAX_TEXT})
        self.assertEqual(stream.getvalue(),raw);self.assertEqual(len(capture.records),2)
        with patch.object(d,'MAX_RECORDS',2):
            with self.assertRaises(ValueError):capture.add({})
        with patch.object(d,'MAX_CAPTURE',capture.bytes):
            with self.assertRaises(ValueError):capture.add({})
        self.assertEqual(stream.getvalue(),raw)

    def test_short_write_and_flush_failure_preserve_attempted_descriptor(self):
        class Short(io.BytesIO):
            def write(self,raw):return super().write(raw[:-1])
        class Flush(io.BytesIO):
            def flush(self):raise OSError('injected flush failure')
        for stream in (Short(),Flush()):
            capture=d.Capture(stream)
            with self.assertRaises(OSError):capture.add({'retained':'é'})
            self.assertEqual(len(capture.records),1)
            self.assertFalse(capture.records[0]['completed'])
            self.assertEqual(capture.bytes,0);self.assertTrue(stream.getvalue())

    def test_cold_bad_inventory_identity_version_and_preimport_reject(self):
        runtime=d.runtime
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'source.py';source.write_text('pass\n')
            def manifest(files):
                raw=(json.dumps(dict(format='toklet-text-reference-runtime-v1',versions=runtime.VERSIONS,files=files))+'\n').encode()
                q=root/'manifest.json';q.write_bytes(raw);return q,hashlib.sha256(raw).hexdigest()
            pin=dict(bytes=source.stat().st_size,sha256=hashlib.sha256(source.read_bytes()).hexdigest())
            with patch.object(runtime,'paths',return_value=[source]):
                q,sha=manifest({})
                with self.assertRaisesRegex(ValueError,'inventory'):runtime.admit(q,sha,[])
                q,sha=manifest({str(source):dict(pin,sha256='0'*64)})
                with self.assertRaisesRegex(ValueError,'identity'):runtime.admit(q,sha,[])
                q,sha=manifest({str(source):dict(pin,bytes=pin['bytes']+1)})
                with self.assertRaisesRegex(ValueError,'byte count'):runtime.admit(q,sha,[])
                q,sha=manifest({str(source):pin})
                with patch.object(runtime,'version',return_value='wrong'):
                    with self.assertRaisesRegex(ValueError,'versions'):runtime.admit(q,sha,[])
                with patch.dict(runtime.sys.modules,{'tokenizers':object()}):
                    with self.assertRaisesRegex(ValueError,'already imported'):runtime.admit(q,sha,[])

    def test_actual_driver_retains_partial_records_and_primary_on_endpoint_error(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();(root/'artifacts').mkdir()
            def failed(capture,_):
                capture.add(dict(kind='partial',ids=[1,2,3]))
                raise InterruptedError('injected after capture')
            def endpoint():raise RuntimeError('injected endpoint failure')
            with patch.object(d,'ROOT',root),patch.object(d,'produce',side_effect=failed),\
                    patch.object(d.runtime.common,'captured',return_value=b'fixture'):
                self.assertEqual(d.run(root/'artifacts'/'output',{},endpoint),1)
            output=root/'artifacts/output';obs=json.loads((output/'observation.json').read_bytes())
            raw=(output/'records.jsonl').read_bytes()
            self.assertFalse(obs['passed']);self.assertEqual(len(obs['records']),1)
            self.assertTrue(obs['records'][0]['completed']);self.assertEqual(json.loads(raw)['ids'],[1,2,3])
            self.assertIn('InterruptedError: injected after capture',obs['primary_error'])
            self.assertTrue(any('endpoint failure' in s for s in obs['endpoint_errors']))
            self.assertEqual(obs['raw'],dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))

    def test_cold_source_shadow_namespace_and_wrong_loaders_reject(self):
        runtime=d.runtime;site=Path('/test-site');specs={};inventory=[]
        for name in runtime.PACKAGES:
            path=site/name/'__init__.py';inventory.append(path)
            spec=ModuleSpec(name,SourceFileLoader(name,str(path)),origin=str(path),is_package=True)
            spec.submodule_search_locations=[str(path.parent)];specs[name]=spec
        for name in ('tokenizers.tokenizers','markupsafe._speedups'):
            path=site/name.split('.')[0]/(name.split('.')[1]+'.so');inventory.append(path)
            specs[name]=ModuleSpec(name,ExtensionFileLoader(name,str(path)),origin=str(path))
        def selected():
            with patch.object(runtime,'find_spec',side_effect=lambda n:specs[n]),\
                    patch.object(runtime.PathFinder,'find_spec',side_effect=lambda n,_:specs[n]):
                return runtime.selected_imports(site,inventory)
        self.assertEqual(set(selected()),set(specs))
        for name,field,value in [('jinja2','origin','/shadow/jinja2/__init__.py'),
                ('jinja2','loader',None),('jinja2','submodule_search_locations',['/namespace']),
                ('tokenizers.tokenizers','loader',SourceFileLoader('fake','/fake.py')),
                ('markupsafe._speedups','origin','/shadow/_speedups.so')]:
            with patch.object(specs[name],field,value):
                with self.assertRaisesRegex(ValueError,'resolution/loader'):selected()

    def test_actual_driver_raw_read_and_publication_errors_restore_signals(self):
        real_open=Path.open
        for point in ('raw','publication'):
            with self.subTest(point=point),tempfile.TemporaryDirectory() as folder:
                root=Path(folder).resolve();(root/'artifacts').mkdir()
                previous={v:signal.getsignal(v) for v in (signal.SIGINT,signal.SIGTERM)}
                def failed(capture,_):
                    capture.add(dict(kind='partial'));raise OSError('original producer failure')
                def opening(path,*args,**kwargs):
                    mode=args[0] if args else kwargs.get('mode','r')
                    if (point=='raw' and path.name=='records.jsonl' and mode=='rb') or (point=='publication' and path.name=='observation.json' and mode=='x'):
                        raise OSError('injected '+point+' failure')
                    return real_open(path,*args,**kwargs)
                with patch.object(d,'ROOT',root),patch.object(d,'produce',side_effect=failed),\
                        patch.object(d.runtime.common,'captured',return_value=b'fixture'),\
                        patch.object(Path,'open',opening):
                    if point=='publication':
                        with self.assertRaisesRegex(OSError,'publication'):d.run(root/'artifacts/output',{},lambda:{})
                    else:self.assertEqual(d.run(root/'artifacts/output',{},lambda:{}),1)
                self.assertEqual(previous,{v:signal.getsignal(v) for v in previous})
                if point=='raw':
                    obs=json.loads((root/'artifacts/output/observation.json').read_bytes())
                    self.assertEqual(obs['primary_error'],'OSError: original producer failure')
                    self.assertTrue(any('raw failure' in s for s in obs['endpoint_errors']))


if __name__=='__main__':unittest.main()
