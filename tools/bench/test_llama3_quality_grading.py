"""Llama adapter admission controls; no VM or generated code executes."""
import copy
import hashlib
import types
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import llama3_quality_grading as subject


def complete():
    return dict(passed=True,primary_error=None,endpoint_errors=[],observations=dict(scope=dict(full_model_collections_complete=True)),
                commands=[dict(name=name,pid=100001+i,pgid=100001+i,exit_code=0,natural_exit=True,reaped=True,primary_error=None,cleanup_errors=[])for i,name in enumerate(['native','original'])])


class LlamaGradingAdmission(unittest.TestCase):
    def test_both_named_natural_exits_and_absent_groups_are_required(self):
        with patch.object(subject.os,'kill',side_effect=ProcessLookupError)as query:
            self.assertEqual(subject.require_idle(complete()),[100001,100002]);self.assertEqual(query.call_count,4)
        for kind in ['partial','name','exit','natural','reaped','group','duplicate']:
            record=complete()
            if kind=='partial':record['observations']['scope']['full_model_collections_complete']=False
            elif kind=='name':record['commands'][0]['name']='other'
            elif kind=='exit':record['commands'][1]['exit_code']=1
            elif kind=='natural':record['commands'][1]['natural_exit']=False
            elif kind=='reaped':record['commands'][1]['reaped']=False
            elif kind=='group':record['commands'][1]['pgid']=99
            else:record['commands'][1]['pid']=record['commands'][0]['pid']
            with patch.object(subject.os,'kill',side_effect=ProcessLookupError):
                with self.assertRaises(ValueError):subject.require_idle(record)
        with patch.object(subject.os,'kill',return_value=None):
            with self.assertRaises(ValueError):subject.require_idle(complete())
    def test_bind_rejects_symlink_and_changed_payload(self):
        with tempfile.TemporaryDirectory(dir=subject.ROOT/'artifacts')as name:
            folder=Path(name);path=folder/'payload';path.write_bytes(b'original');bindings={};subject.bind(bindings,path)
            alias=folder/'alias';alias.symlink_to(path)
            with self.assertRaises(ValueError):subject.bind({},alias)
            pin=bindings[str(path.relative_to(subject.ROOT))];path.write_bytes(b'changed!')
            with self.assertRaises(ValueError):subject.verify_file(path,pin)

    def test_decoder_constructs_from_exact_verified_buffer(self):
        with tempfile.TemporaryDirectory()as name:
            folder=Path(name);assets={'tokenizer.json':b'verified-tokenizer','tokenizer_config.json':b'config','special_tokens_map.json':b'special'}
            for key,value in assets.items():(folder/key).write_bytes(value)
            manifest=dict(tokenizer_pins={key:hashlib.sha256(value).hexdigest()for key,value in assets.items()},preparation_provider_sha256={})
            real_fdopen=subject.os.fdopen;changed=False
            class SwappingFile:
                def __init__(self,file):self.file=file
                def __enter__(self):return self.file
                def __exit__(self,*_):
                    nonlocal changed
                    self.file.close()
                    if not changed:(folder/'tokenizer.json').write_bytes(b'replaced-after-verification');changed=True
            class Tokenizer:
                @staticmethod
                def from_buffer(raw):
                    self.assertEqual(raw,assets['tokenizer.json']);return Tokenizer()
                def decode(self,ids,skip_special_tokens):return ' literal \n'
            cases=[dict(id=f'case-{i}',category='prose')for i in range(100)]
            records=[dict(id=c['id'],category=c['category'],visible_token_ids=[7])for c in cases]
            with patch.object(subject.inputs.codec_policy,'MODEL',folder),patch.object(subject.inputs.codec_policy,'closure',return_value={}),patch.object(subject.importlib.metadata,'version',return_value='0.22.2'),patch.object(subject.os,'fdopen',side_effect=lambda fd,mode:SwappingFile(real_fdopen(fd,mode))),patch.dict('sys.modules',{'tokenizers':types.SimpleNamespace(Tokenizer=Tokenizer)}):
                decoded=subject.decode(cases,records,manifest)
            self.assertEqual(decoded[0]['visible_text'],' literal \n')
